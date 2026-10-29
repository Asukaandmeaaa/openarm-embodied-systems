"""Adapt the deployed 50-step ACT checkpoint to a 100-step chunk.

This is a structural adaption followed by a new fine-tuning experiment, not a
seamless resume of the original training.  Only the learned decoder position
embedding depends on chunk_size; it is expanded by copying rows 0..49 exactly
and initialising rows 50..99 from row 49.  Every other tensor is copied
unchanged.  Load is strict except for that documented tensor.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import torch

from .sim_collect import digest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEFAULT_SOURCE = ROOT / "models/act50/pretrained_model"
DEFAULT_OUTPUT = ROOT / "models/act100_init/pretrained_model"


def write_json(path: Path, value: dict) -> None:
    path = Path(path)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def load_policy(path: Path):
    from lerobot.configs import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class

    cfg = PreTrainedConfig.from_pretrained(path)
    cfg.pretrained_path = Path(path)
    cfg.device = "cpu"
    policy = get_policy_class(cfg.type).from_pretrained(
        config=cfg, pretrained_name_or_path=path
    )
    return policy, cfg


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--n-action-steps", type=int, default=50)
    parser.add_argument("--check", action="store_true", help="Only inspect the source config")
    args = parser.parse_args()
    if args.chunk_size != 100 or args.n_action_steps != 50:
        parser.error("this audited adapter supports exactly 50 -> 100 with n_action_steps=50")
    source = args.source.resolve()
    output = args.output.resolve()
    if not (source / "config.json").is_file():
        raise SystemExit("source checkpoint missing config.json")
    source_config = json.loads((source / "config.json").read_text())
    source_hashes = {
        p.name: digest(p) for p in sorted(source.iterdir()) if p.is_file()
    }
    if args.check:
        print(
            json.dumps(
                {
                    "source": str(source),
                    "chunk_size": source_config["chunk_size"],
                    "n_action_steps": source_config["n_action_steps"],
                    "hashes": source_hashes,
                },
                indent=2,
            )
        )
        return 0
    if output.exists():
        raise SystemExit("output exists; preserve it and choose a new directory")
    if source_config.get("chunk_size") != 50:
        raise SystemExit("source is not the expected 50-step checkpoint")
    policy, cfg = load_policy(source)
    old = policy.model.decoder_pos_embed
    if tuple(old.weight.shape) != (50, policy.config.dim_model):
        raise SystemExit("unexpected decoder_pos_embed shape")
    new = torch.nn.Embedding(args.chunk_size, policy.config.dim_model)
    with torch.no_grad():
        new.weight[:50].copy_(old.weight)
        new.weight[50:].copy_(old.weight[-1:].expand(args.chunk_size - 50, -1))
    policy.model.decoder_pos_embed = new
    # The VAE encoder uses a fixed sinusoidal table sized by chunk_size.  It is
    # not learned; recompute it with the official helper so rows 0..51 remain
    # the same and the new rows use the same sinusoidal definition.
    from lerobot.policies.act.modeling_act import create_sinusoidal_pos_embedding

    old_vae = policy.model.vae_encoder_pos_enc
    new_vae = create_sinusoidal_pos_embedding(
        args.chunk_size + 2, policy.config.dim_model
    ).unsqueeze(0)
    if tuple(old_vae.shape) != (1, 52, policy.config.dim_model):
        raise SystemExit("unexpected vae_encoder_pos_enc shape")
    if not torch.equal(new_vae[:, :52], old_vae):
        raise RuntimeError("recomputed sinusoidal prefix differs from source")
    policy.model.vae_encoder_pos_enc = new_vae
    policy.config.chunk_size = args.chunk_size
    policy.config.n_action_steps = args.n_action_steps
    policy.config.temporal_ensemble_coeff = None
    policy.config.device = source_config.get('device', 'cuda')
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(output.name + ".incomplete")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()
    policy.save_pretrained(staging)
    for p in source.glob("policy_*"):
        shutil.copy2(p, staging / p.name)
        if digest(p) != digest(staging / p.name):
            raise RuntimeError("processor copy mismatch")
    # Record exact parameter-level changes and verify nothing else moved.
    from safetensors.torch import load_file

    before = load_file(str(source / "model.safetensors"))
    after = load_file(str(staging / "model.safetensors"))
    if set(before) != set(after):
        raise RuntimeError("parameter key set changed")
    changed = []
    for key in before:
        if before[key].shape != after[key].shape:
            if key == "model.decoder_pos_embed.weight":
                method = "source rows 0..49 copied; rows 50..99 repeat source row 49"
            elif key == "model.vae_encoder_pos_enc":
                method = "official sinusoidal table recomputed for 102 tokens; rows 0..51 byte-identical to source"
            else:
                raise RuntimeError(f"undocumented shape change in {key}")
            changed.append(
                {
                    "parameter": key,
                    "source_shape": list(before[key].shape),
                    "output_shape": list(after[key].shape),
                    "initialization": method,
                }
            )
        elif not torch.equal(before[key], after[key]):
            raise RuntimeError(f"unexpected value change in {key}")
    if [c["parameter"] for c in changed] != [
        "model.decoder_pos_embed.weight",
        "model.vae_encoder_pos_enc",
    ]:
        raise RuntimeError("unexpected adaptation set: " + json.dumps(changed))
    if not torch.equal(after["model.decoder_pos_embed.weight"][:50], before["model.decoder_pos_embed.weight"]):
        raise RuntimeError("decoder embedding prefix changed")
    if not torch.equal(after["model.decoder_pos_embed.weight"][50:], before["model.decoder_pos_embed.weight"][-1:].expand(50, -1)):
        raise RuntimeError("decoder embedding extension differs from documented repeat")
    manifest = {
        "source": str(source),
        "source_sha256": source_hashes,
        "output": str(output),
        "structural_adaptation": True,
        "experiment_note": "new fine-tuning experiment, not a seamless resume of the original training",
        "source_chunk_size": 50,
        "output_chunk_size": args.chunk_size,
        "n_action_steps": args.n_action_steps,
        "changed_parameters": changed,
        "unchanged_parameters": len(before) - len(changed),
        "normalization": "frozen real-data processors copied byte-for-byte; no sim statistics used",
        "output_files": {p.name: digest(p) for p in sorted(staging.iterdir()) if p.is_file()},
    }
    write_json(staging / "chunk_adaptation.json", manifest)
    staging.rename(output)
    # Reload and run one synthetic forward + native loss; no hardware.
    reload_policy, cfg = load_policy(output)
    if cfg.chunk_size != args.chunk_size:
        raise RuntimeError("reloaded chunk_size mismatch")
    batch = {
        "observation.state": torch.zeros(1, 16),
        "observation.images.ceiling": torch.full((1, 3, 480, 640), 0.5),
        "observation.images.wrist_right": torch.full((1, 3, 480, 640), 0.5),
        "action": torch.zeros(1, args.chunk_size, 16),
        "action_is_pad": torch.zeros(1, args.chunk_size, dtype=torch.bool),
    }
    reload_policy.eval()
    with torch.inference_mode():
        out = reload_policy.predict_action_chunk(batch)
        if tuple(out.shape) != (1, args.chunk_size, 16) or not torch.isfinite(out).all():
            raise RuntimeError("synthetic prediction shape/finite check failed")
        reload_policy.train()
        loss, _ = reload_policy.forward(batch)
        if not torch.isfinite(loss):
            raise RuntimeError("synthetic native loss is not finite")
    write_json(
        output.parent / "verification.json",
        {
            "hardware_accessed": False,
            "prediction_shape": list(out.shape),
            "native_loss": float(loss),
            "reload_from_disk": True,
            "future_actions_not_used_for_inference": True,
        },
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
