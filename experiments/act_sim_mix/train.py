"""Operator-run 30,000-update mixed ACT training.  No hardware access.

Domain composition is fixed at real 50% / nominal simulation 25% / corrected
simulation 25% per batch (4/2/2 at batch size 8).  Validation frames never
enter gradients, sampling, or normalization; the frozen real-data processors
are used for both domains.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .data import CAMS, HORIZON, EpisodeDataset, MixedSampler, REAL_ROOT, digest, load_datasets
from .sim_collect import write_json
from .metrics import PredictionMetrics, phase_labels, validation_brief

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROTOCOL = HERE / "config/train_protocol.json"


def load_policy(model: Path):
    from lerobot.configs import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    cfg = PreTrainedConfig.from_pretrained(model)
    cfg.pretrained_path = Path(model)
    cfg.device = "cuda"
    policy = get_policy_class(cfg.type).from_pretrained(
        config=cfg, pretrained_name_or_path=model
    )
    pre, post = make_pre_post_processors(
        cfg,
        pretrained_path=model,
        preprocessor_overrides={"device_processor": {"device": "cuda"}},
    )
    return policy, pre, post


def model_binding(model: Path) -> dict:
    config = json.loads((model / "config.json").read_text())
    return {
        "path": str(model),
        "chunk_size": config["chunk_size"],
        "n_action_steps": config["n_action_steps"],
        "files": {p.name: digest(p) for p in sorted(model.iterdir()) if p.is_file()},
    }


def data_binding(converted: Path) -> dict:
    split = json.loads((converted / "split.json").read_text())
    return {
        "converted": str(converted),
        "split_sha256": digest(converted / "split.json"),
        "conversion_report_sha256": digest(converted / "conversion_report.json"),
        "real_train_manifest_sha256": digest(
            REAL_ROOT / "train/meta/visual_expert_manifest.json"
        ),
        "real_validation_manifest_sha256": digest(
            REAL_ROOT / "validation/meta/visual_expert_manifest.json"
        ),
        "sim_train_manifest_sha256": digest(
            converted / "train/meta/visual_expert_manifest.json"
        ),
        "sim_validation_manifest_sha256": digest(
            converted / "validation/meta/visual_expert_manifest.json"
        ),
        "sim_test_manifest_sha256": digest(
            converted / "test/meta/visual_expert_manifest.json"
        ),
        "splits": split,
    }


def code_binding() -> dict:
    files = [PROTOCOL, HERE / "train.py", HERE / "data.py", HERE / "adapt_chunk.py", HERE / "metrics.py", HERE / "paths.py", HERE / "contracts.py"]
    return {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else 'external_train_protocol': digest(p) for p in files}


def preflight(protocol: dict) -> dict:
    model = ROOT / protocol["initial_model"]
    converted = ROOT / protocol["converted_dataset"]
    if not model.is_dir():
        raise FileNotFoundError(model)
    if not (converted / "split.json").is_file():
        raise FileNotFoundError(converted / "split.json")
    binding = model_binding(model)
    if binding["chunk_size"] != 100 or binding["n_action_steps"] != 50:
        raise ValueError("adapted model must have chunk_size=100 and n_action_steps=50")
    real, normal, recovery = load_datasets(converted)
    real_val = EpisodeDataset(REAL_ROOT / "validation", domain="real")
    sim_val = EpisodeDataset(converted / "validation", domain="sim")
    sim_test = EpisodeDataset(converted / "test", domain="sim")
    normal.episodes = [e for e in normal.episodes if e["domain"] == "normal"]
    recovery.episodes = [e for e in recovery.episodes if e["domain"] == "recovery"]
    for ds in (normal, recovery):
        offset = 0
        for e in ds.episodes:
            e["offset"] = offset
            offset += e["length"]
        ds.ends = np.cumsum([e["length"] for e in ds.episodes])
        ds.total = int(ds.ends[-1]) if len(ds.ends) else 0
    split = json.loads((converted / "split.json").read_text())
    counts = {
        "real_train_episodes": len(real.episodes),
        "real_train_frames": len(real),
        "real_validation_episodes": len(real_val.episodes),
        "real_validation_frames": len(real_val),
        "sim_train_normal_episodes": len(normal.episodes),
        "sim_train_normal_frames": len(normal),
        "sim_train_recovery_episodes": len(recovery.episodes),
        "sim_train_recovery_frames": len(recovery),
        "sim_validation_episodes": len(sim_val.episodes),
        "sim_validation_frames": len(sim_val),
        "sim_test_episodes": len(sim_test.episodes),
        "sim_test_frames": len(sim_test),
    }
    expected = protocol.get("expected_counts", {
        "sim_train_normal_episodes": 80,
        "sim_train_recovery_episodes": 40,
        "sim_validation_episodes": 15,
        "sim_test_episodes": 15,
    })
    for key in ("real_train_frames", "real_validation_frames", "sim_train_normal_frames", "sim_train_recovery_frames"):
        if counts[key] == 0:
            raise ValueError(f"Empty required training/validation domain: {key}")
    for key in expected:
        if counts[key] != expected[key]:
            raise ValueError(f"{key}={counts[key]} expected {expected[key]}")
    group_train = {e["group"] for e in normal.episodes} | {
        e["group"] for e in recovery.episodes
    }
    group_val = {e["group"] for e in sim_val.episodes}
    group_test = {e["group"] for e in sim_test.episodes}
    if group_train & group_val or group_train & group_test or group_val & group_test:
        raise ValueError("scenario family leakage across converted splits")
    scenario_sets = []
    for name in ("train", "validation", "test"):
        manifest = json.loads((converted / name / "meta/visual_expert_manifest.json").read_text())
        keys = set()
        for episode in manifest["episodes"]:
            if not episode.get("scenario_keys"):
                raise ValueError("Missing canonical scenario identity; use the v2 converter")
            keys.update(episode["scenario_keys"])
        scenario_sets.append(keys)
    if any(scenario_sets[i] & scenario_sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("cross-domain layout/seed leakage across converted splits")
    free = shutil.disk_usage(HERE).free / 2**30
    if free < 12:
        raise RuntimeError("need at least 12 GiB free for retained checkpoints")
    bound_data = data_binding(converted)
    bound_data["metric_phase_sources"] = {
        name: {str(e["number"]): phase_labels(ds, e)[1] for e in ds.episodes}
        for name, ds in (("real_validation", real_val), ("sim_validation", sim_val))
    }
    return {
        "model": binding,
        "data": bound_data,
        "code": code_binding(),
        "counts": counts,
        "scenario_families_disjoint": True,
        "free_disk_gib": free,
        "hardware_accessed": False,
        "protocol": protocol,
    }


def capture_rng(rng: np.random.Generator) -> dict:
    return dict(
        python=random.getstate(),
        numpy=np.random.get_state(),
        torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        sampler=rng.bit_generator.state,
    )


def restore_rng(state: dict, rng: np.random.Generator) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("CUDA device count changed; exact resume unavailable")
        torch.cuda.set_rng_state_all([v.cpu() for v in state["cuda"]])
    rng.bit_generator.state = state["sampler"]


def save_checkpoint(path: Path, policy, optimizer, rng, sampler: MixedSampler, step: int, recipe: dict, model_dir: Path) -> None:
    staging = path.with_name(path.name + ".incomplete")
    if path.exists() or staging.exists():
        raise FileExistsError(path)
    staging.mkdir()
    policy.save_pretrained(staging)
    for p in model_dir.glob("policy_*"):
        shutil.copy2(p, staging / p.name)
        if digest(p) != digest(staging / p.name):
            raise RuntimeError("processor copy mismatch")
    counts = {k: v.copy() for k, v in sampler.counts.items()}
    torch.save(
        dict(
            step=step,
            optimizer=optimizer.state_dict(),
            rng=capture_rng(rng),
            counts={k: torch.from_numpy(v) for k, v in counts.items()},
            domain_draws=sampler.domain_draws,
        ),
        staging / "training_state.pt",
    )
    write_json(
        staging / "manifest.json",
        dict(
            step=step,
            recipe=recipe,
            files={p.name: digest(p) for p in staging.iterdir() if p.is_file()},
        ),
    )
    staging.rename(path)


def validate_checkpoint(path: Path, recipe: dict) -> dict:
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest["recipe"]["contract"] != recipe["contract"]:
        raise ValueError("resume contract changed")
    for name, expected in manifest["files"].items():
        if digest(path / name) != expected:
            raise ValueError("checkpoint integrity failure: " + name)
    return manifest


def evaluate(policy, pre, post, datasets: dict[str, EpisodeDataset]) -> dict:
    # Observe the existing ACT forward result: no second model inference and
    # no change to native loss, gradients, or optimizer behavior.
    was_training = policy.training
    policy.eval()
    out = {}
    captured = []
    def capture(_module, _inputs, output):
        captured.append(output[0].detach())
    hook = policy.model.register_forward_hook(capture)
    try:
        with torch.inference_mode():
            for name, ds in datasets.items():
                if len(ds) == 0:
                    out[name] = {"frames": 0, "loss": None}
                    continue
                diagnostics = PredictionMetrics(ds, int(policy.config.chunk_size))
                total, count = 0.0, 0
                evaluation_started = progress_at = time.monotonic()
                print("VALIDATION START", name, "frames=" + str(len(ds)), flush=True)
                for start in range(0, len(ds), 8):
                    samples = [ds[i] for i in range(start, min(start + 8, len(ds)))]
                    batch = {
                        key: torch.stack([s[key] for s in samples])
                        for key in ("observation.state", "action", "action_is_pad", *CAMS)
                    }
                    proc = pre(batch)
                    captured.clear()
                    loss, _ = policy.forward(proc)
                    if not torch.isfinite(loss) or len(captured) != 1:
                        raise RuntimeError("Invalid ACT validation forward or diagnostic capture")
                    actions = captured.pop()
                    physical = post(actions.reshape(-1, actions.shape[-1])).reshape(actions.shape)
                    diagnostics.update(physical.cpu().numpy(),
                        [int(s["episode_index"]) for s in samples], [int(s["frame_index"]) for s in samples])
                    total += float(loss) * len(samples)
                    count += len(samples)
                    now = time.monotonic()
                    if count == len(ds) or now - progress_at >= 20:
                        elapsed = now - evaluation_started
                        print("VALIDATION_PROGRESS", name, json.dumps({
                            "frames": count, "total_frames": len(ds),
                            "elapsed_s": round(elapsed, 1),
                            "estimated_remaining_s": round(elapsed / count * (len(ds) - count), 1),
                        }), flush=True)
                        progress_at = now
                out[name] = {"frames": count, "loss": total / count, "diagnostics": diagnostics.report()}
                print("VALIDATION", name, json.dumps(validation_brief(out[name])), flush=True)
    finally:
        hook.remove()
        policy.train(was_training)
    return out


def main() -> int:
    global PROTOCOL
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL, help="Training recipe; paths are relative to the checkout")
    parser.add_argument("--output", type=Path, help="New run directory")
    parser.add_argument("--resume", type=Path, help="Complete checkpoint from this experiment")
    parser.add_argument("--check", action="store_true", help="Data/model/hash check only")
    parser.add_argument("--steps", type=int, help="Override total additional steps (smoke only)")
    parser.add_argument("--eval-every", type=int, help="Override checkpoint interval (smoke only)")
    args = parser.parse_args()
    PROTOCOL = args.protocol.resolve()
    protocol = json.loads(PROTOCOL.read_text())
    if args.steps is not None:
        protocol["additional_steps"] = args.steps
    if args.eval_every is not None:
        protocol["eval_every"] = args.eval_every
    check = preflight(protocol)
    if args.check:
        print(json.dumps(check, indent=2))
        print("CHECK OK: no model loaded, no optimizer updates, no hardware.")
        return 0
    if args.output is None:
        parser.error("--output is required for training")
    if args.output.exists():
        parser.error("output exists; preserve it and choose a new directory")
    contract = {
        "model": check["model"],
        "data": check["data"],
        "code": check["code"],
        "protocol": protocol,
        "environment": dict(
            torch=torch.__version__,
            numpy=np.__version__,
            python=platform.python_version(),
            cuda=torch.version.cuda,
        ),
    }
    recipe = {
        "contract": contract,
        "output": str(args.output.resolve()),
        "initial_model": str(ROOT / protocol["initial_model"]),
        "resumed_from": str(args.resume.resolve()) if args.resume else None,
        "hardware_accessed": False,
        "model_deployed": False,
        "physical_success_verified": False,
    }
    manifest = None
    if args.resume:
        manifest = validate_checkpoint(args.resume.resolve(), recipe)
        recipe["baseline_run"] = manifest["recipe"].get("baseline_run", str(args.output.resolve()))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this fixed recipe")
    random.seed(protocol["seed"])
    np.random.seed(protocol["seed"])
    torch.manual_seed(protocol["seed"])
    torch.cuda.manual_seed_all(protocol["seed"])
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    rng = np.random.default_rng(protocol["seed"])
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    model_dir = ROOT / protocol["initial_model"]
    policy, pre, post = load_policy(args.resume.resolve() if manifest else model_dir)
    policy.config.optimizer_lr_backbone = protocol["lr"]
    optimizer = torch.optim.AdamW(
        policy.get_optim_params(),
        lr=protocol["lr"],
        weight_decay=protocol["weight_decay"],
    )
    converted = ROOT / protocol["converted_dataset"]
    real, normal, recovery = load_datasets(converted)
    normal.episodes = [e for e in normal.episodes if e["domain"] == "normal"]
    recovery.episodes = [e for e in recovery.episodes if e["domain"] == "recovery"]
    for ds in (normal, recovery):
        offset = 0
        for e in ds.episodes:
            e["offset"] = offset
            offset += e["length"]
        ds.ends = np.cumsum([e["length"] for e in ds.episodes])
        ds.total = int(ds.ends[-1]) if len(ds.ends) else 0
    sampler = MixedSampler(real, normal, recovery, batch_size=protocol["batch_size"])
    start_step = 0
    if manifest:
        state = torch.load(args.resume / "training_state.pt", map_location="cpu", weights_only=False)
        start_step = int(state["step"])
        if start_step <= 0 or start_step > protocol["additional_steps"]:
            raise ValueError("invalid resume step")
        optimizer.load_state_dict(state["optimizer"])
        restore_rng(state["rng"], rng)
        sampler.counts = {k: v.numpy().copy() for k, v in state["counts"].items()}
        sampler.domain_draws = state["domain_draws"]
    write_json(output / "recipe.json", recipe)
    if manifest:
        evaluation = evaluate(
            policy,
            pre,
            post,
            {
                "real_validation": EpisodeDataset(REAL_ROOT / "validation", domain="real"),
                "sim_validation": EpisodeDataset(converted / "validation", domain="sim"),
            },
        )
        write_json(
            output / f"summary_{start_step:06d}.json",
            {"step": start_step, "evaluation": evaluation, "resume_re_evaluation": True},
        )
    else:
        baseline = evaluate(
            policy,
            pre,
            post,
            {
                "real_validation": EpisodeDataset(REAL_ROOT / "validation", domain="real"),
                "sim_validation": EpisodeDataset(converted / "validation", domain="sim"),
            },
        )
        write_json(output / "summary_000000.json", {"step": 0, "evaluation": baseline})
        recipe["baseline_summary_sha256"] = digest(output / "summary_000000.json")
        write_json(output / "recipe.json", recipe)
    val_sets = {
        "real_validation": EpisodeDataset(REAL_ROOT / "validation", domain="real"),
        "sim_validation": EpisodeDataset(converted / "validation", domain="sim"),
    }
    torch.cuda.reset_peak_memory_stats()
    began = time.monotonic()
    try:
        with (output / "train.jsonl").open("x") as log:
            for step in range(start_step + 1, protocol["additional_steps"] + 1):
                batch = sampler.batch(rng)
                model_batch = {
                    key: batch[key]
                    for key in ("observation.state", "action", "action_is_pad", *CAMS)
                }
                proc = pre(model_batch)
                policy.train()
                optimizer.zero_grad(set_to_none=True)
                loss, details = policy.forward(proc)
                if not torch.isfinite(loss):
                    raise RuntimeError("nonfinite loss")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(
                    policy.parameters(), protocol["grad_clip"], error_if_nonfinite=True
                )
                optimizer.step()
                row = dict(
                    step=step,
                    loss=float(loss.detach()),
                    gradient_norm=float(norm),
                    lr=optimizer.param_groups[0]["lr"],
                    elapsed_s=time.monotonic() - began,
                    domains=batch["_domains"],
                    indices=batch["index"].tolist(),
                    episodes=batch["episode_index"].tolist(),
                    frames=batch["frame_index"].tolist(),
                    **{k: v for k, v in details.items() if isinstance(v, (int, float))},
                )
                log.write(json.dumps(row) + "\n")
                log.flush()
                if step <= start_step + 2 or step % 50 == 0:
                    print(json.dumps({k: v for k, v in row.items() if k not in ("indices", "episodes", "frames", "domains")}), flush=True)
                if step % protocol["eval_every"] == 0 or step == protocol["additional_steps"]:
                    ckpt = output / f"checkpoint_{step:06d}"
                    save_checkpoint(ckpt, policy, optimizer, rng, sampler, step, recipe, model_dir)
                    print("CHECKPOINT COMPLETE:", ckpt, flush=True)
                    evaluation = evaluate(policy, pre, post, val_sets)
                    write_json(
                        output / f"summary_{step:06d}.json",
                        dict(
                            step=step,
                            evaluation=evaluation,
                            domain_draws=sampler.domain_draws,
                            peak_gpu_mib=torch.cuda.max_memory_allocated() / 2**20,
                        ),
                    )
                    np.savez(
                        output / f"sample_counts_{step:06d}.npz",
                        **{k: v for k, v in sampler.counts.items()},
                    )
    except BaseException as exc:
        write_json(
            output / "interrupted.json",
            dict(
                error=type(exc).__name__,
                message=str(exc),
                note="Resume only a complete checkpoint with manifest.json; old outputs are preserved.",
            ),
        )
        raise
    write_json(
        output / "completed.json",
        dict(
            additional_steps=protocol["additional_steps"],
            resumed_from_step=start_step,
            elapsed_s=time.monotonic() - began,
            hardware_accessed=False,
            model_deployed=False,
            physical_success_verified=False,
        ),
    )
    print("DONE: training budget complete. Review adjacent checkpoints before any deployment.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
