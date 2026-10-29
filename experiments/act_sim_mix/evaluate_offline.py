"""Offline prediction evaluation on the supplied real development episodes and the
simulation validation/test splits.  This is not closed-loop success.

Reports arm/gripper prediction error by horizon and, for real validation, the
same grasp/release onset timing diagnostics used by the existing ACT review.
Run once per model; compare reports side by side.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from .data import CAMS, EpisodeDataset, REAL_ROOT, digest
from .sim_collect import write_json
from .metrics import PredictionMetrics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
from .phase_reference import SavedDataset


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
    return policy, pre, post, cfg


def predict_dataset(policy, pre, post, ds: EpisodeDataset, horizon: int, batch_size: int = 8):
    preds = [np.zeros((len(e["a"]), horizon, 16), dtype=np.float32) for e in ds.episodes]
    policy.eval()
    with torch.inference_mode():
        for start in range(0, len(ds), batch_size):
            items = [ds[i] for i in range(start, min(start + batch_size, len(ds)))]
            batch = {
                key: torch.stack([item[key] for item in items])
                for key in ("observation.state", *CAMS)
            }
            proc = pre(batch)
            out = policy.predict_action_chunk(proc)
            flat = post(out.reshape(-1, out.shape[-1])).reshape(out.shape)
            flat = flat.cpu().numpy()
            for item, row in zip(items, flat):
                e, frame = ds.locate(int(item["index"]))
                if row.shape != (horizon, 16):
                    raise ValueError("prediction horizon mismatch")
                preds[e["number"]][frame] = row
    return preds


def first_three(flags):
    hits = np.flatnonzero(flags[:-2] & flags[1:-1] & flags[2:])
    return int(hits[0]) if len(hits) else None


def real_metrics(preds, ds: EpisodeDataset, horizon: int, phase_ds: SavedDataset) -> dict:
    horizons = [h for h in (0, 10, 25, 49, 99) if h < horizon]
    arm = {str(h): [] for h in horizons}
    grip = {str(h): [] for h in horizons}
    details = []
    for pred, e in zip(preds, phase_ds.episodes):
        n = len(pred)
        a = e["a"]
        row = {"episode": e["number"], "source": str(e["source"]), "frames": n}
        for h in horizons:
            if n - h <= 0:
                continue
            err = np.abs(pred[: n - h, h, :7] - a[h:, :7]) * 180 / np.pi
            gerr = np.abs(pred[: n - h, h, 7] - a[h:, 7])
            arm[str(h)].append(float(np.mean(err)))
            grip[str(h)].append(float(np.mean(gerr)))
            row[f"arm_mae_deg_h{h}"] = float(np.mean(err))
            row[f"gripper_mae_rad_h{h}"] = float(np.mean(gerr))
        timing = {}
        for phase in ("close", "open"):
            p = e["phases"][phase]
            hdict = {}
            for h in horizons:
                queries = np.arange(max(0, p["lo"] - h), min(n - h, p["hi"] - h))
                if len(queries) == 0:
                    continue
                progress = (pred[queries, h, 7] - p["g0"]) / (p["g1"] - p["g0"])
                delays = {}
                for fraction in (0.1, 0.5, 0.9):
                    hit = first_three(progress >= fraction)
                    delays[str(fraction)] = (
                        None
                        if hit is None
                        else float((queries[hit] + h - p["crossings"][str(fraction)]) / 30)
                    )
                hdict[str(h)] = delays
            timing[phase] = hdict
        row["timing_s"] = timing
        details.append(row)
    summary = {
        "frames": int(sum(len(p) for p in preds)),
        "episodes": len(preds),
        "horizon": horizon,
        "arm_mae_deg_by_horizon": {
            h: float(np.mean(v)) for h, v in arm.items() if v
        },
        "gripper_mae_rad_by_horizon": {
            h: float(np.mean(v)) for h, v in grip.items() if v
        },
        "validation_limit": "development validation is not an untouched final test set",
    }
    # Median 10% onset delay per phase and horizon (h0 only is the deployable
    # current-step diagnostic; other horizons are target-time aligned).
    summary["onset_delay_median_s"] = {}
    for phase in ("close", "open"):
        summary["onset_delay_median_s"][phase] = {}
        for h in horizons:
            vals = [
                row["timing_s"][phase].get(str(h), {}).get("0.1")
                for row in details
                if row["timing_s"][phase].get(str(h), {}).get("0.1") is not None
            ]
            summary["onset_delay_median_s"][phase][str(h)] = (
                float(np.median(vals)) if vals else None
            )
    summary["details"] = details
    return summary


def sim_metrics(preds, ds: EpisodeDataset, horizon: int) -> dict:
    horizons = [h for h in (0, 10, 25, 49, 99) if h < horizon]
    out = {"episodes": len(preds), "horizon": horizon, "arm_mae_deg_by_horizon": {}, "gripper_mae_rad_by_horizon": {}}
    for h in horizons:
        arm, grip = [], []
        for pred, e in zip(preds, ds.episodes):
            n = len(pred)
            if n - h <= 0:
                continue
            arm.append(float(np.mean(np.abs(pred[: n - h, h, :7] - e["a"][h:, :7]) * 180 / np.pi)))
            grip.append(float(np.mean(np.abs(pred[: n - h, h, 7] - e["a"][h:, 7]))))
        out["arm_mae_deg_by_horizon"][str(h)] = float(np.mean(arm)) if arm else None
        out["gripper_mae_rad_by_horizon"][str(h)] = float(np.mean(grip)) if grip else None
    out["by_domain"] = {}
    for domain in ("normal", "recovery"):
        rows = [(p, e) for p, e in zip(preds, ds.episodes) if e["domain"] == domain]
        if not rows:
            out["by_domain"][domain] = None
            continue
        arm = []
        for pred, e in rows:
            n = len(pred)
            arm.append(float(np.mean(np.abs(pred[:, 0, :7] - e["a"][:, :7]) * 180 / np.pi)))
        out["by_domain"][domain] = {"episodes": len(rows), "arm_mae_deg_h0": float(np.mean(arm))}
    return out


def prediction_diagnostics(preds, ds, horizon):
    metrics = PredictionMetrics(ds, horizon)
    for pred, episode in zip(preds, ds.episodes):
        for start in range(0, len(pred), 64):
            frames = np.arange(start, min(start + 64, len(pred)))
            metrics.update(pred[frames], np.full(len(frames), episode["number"]), frames)
    return metrics.report()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--converted", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--real-only", action="store_true")
    parser.add_argument("--sim-split", choices=("validation", "test", "all"), default="validation",
                        help="Use validation for selection; test only for the frozen final candidate")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("output exists; choose a new directory")
    model = args.model.resolve()
    policy, pre, post, cfg = load_policy(model)
    horizon = int(cfg.chunk_size)
    if horizon != args.horizon:
        raise SystemExit(f"model chunk_size={horizon}, --horizon={args.horizon}")
    args.output.mkdir(parents=True)
    report = {
        "model": str(model),
        "model_files": {p.name: digest(p) for p in sorted(model.iterdir()) if p.is_file()},
        "chunk_size": horizon,
        "n_action_steps": int(cfg.n_action_steps),
        "hardware_accessed": False,
        "physical_success_verified": False,
        "note": "Saved-input prediction error; not closed-loop task success.",
    }
    real_val = EpisodeDataset(REAL_ROOT / "validation", domain="real")
    phase_ds = SavedDataset("validation", images=False)
    preds = predict_dataset(policy, pre, post, real_val, horizon, args.batch_size)
    report["real_validation"] = real_metrics(preds, real_val, horizon, phase_ds)
    report["real_validation"]["diagnostics"] = prediction_diagnostics(preds, real_val, horizon)
    np.savez(args.output / "real_validation_predictions.npz", **{
        f"episode_{i:02d}": p for i, p in enumerate(preds)
    })
    for split in (() if args.real_only else (("validation", "test") if args.sim_split == "all" else (args.sim_split,))):
        ds = EpisodeDataset(args.converted / split, domain="sim")
        p = predict_dataset(policy, pre, post, ds, horizon, args.batch_size)
        report[f"sim_{split}"] = sim_metrics(p, ds, horizon)
        report[f"sim_{split}"]["diagnostics"] = prediction_diagnostics(p, ds, horizon)
        np.savez(args.output / f"sim_{split}_predictions.npz", **{
            f"episode_{i:03d}": x for i, x in enumerate(p)
        })
        print(split, json.dumps(report[f"sim_{split}"], indent=2), flush=True)
    write_json(args.output / "report.json", report)
    print(json.dumps(report["real_validation"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
