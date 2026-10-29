"""Audit, group-split and convert simulation episodes to LeRobot v3.

The raw collection is never modified.  Conversion writes only the expert-valid
suffix of each corrective episode; the erroneous prefix stays in the raw
episode for diagnosis and is excluded from training labels by construction.

Optional real data is audited read-only; its supplied split is preserved and its statistics
are not recomputed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq

from .sim_collect import CAMERAS, FPS, IMAGE_HEIGHT, IMAGE_WIDTH, TASK, digest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
from .paths import REAL_ROOT
REAL = REAL_ROOT
CAMS = ("observation.images.ceiling", "observation.images.wrist_right")


def write_json(path: Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def records(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def scenario_keys(meta: dict, run: dict) -> list[str]:
    """Identify shared layouts across normal/recovery domains, independent of labels."""
    def key(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()
    scene = run["scene_sha256"]
    keys = ["seed:" + key([scene, meta["seed"], meta.get("jitter", run.get("jitter"))])]
    if meta.get("randomization"):
        layout = {k: v for k, v in meta["randomization"].items() if k != "rng_seed"}
        keys.append("layout:" + key([scene, layout]))
    return keys


def real_source_keys(entry: dict) -> list[str]:
    """Use provenance, since collection-local episode names are not unique."""
    keys = []
    if entry.get("source_episode_path"):
        keys.append("path:" + str(Path(entry["source_episode_path"]).expanduser().resolve()))
    if entry.get("source_trajectory_trial_id"):
        keys.append("trial:" + entry["source_trajectory_trial_id"])
    for stream in ("metadata.json", "demonstration_action.jsonl"):
        value = entry.get("source_sha256", {}).get(stream)
        if value:
            keys.append("sha256:" + stream + ":" + value)
    if not keys:
        raise ValueError("real episode lacks globally identifiable source provenance")
    return keys


def real_audit() -> dict:
    report: dict = {"root": str(REAL), "splits": {}}
    for split in ("train", "validation"):
        meta = json.loads((REAL / split / "meta/visual_expert_manifest.json").read_text())
        frames = 0
        images = 0
        source_keys = set()
        for number, entry in enumerate(meta["episodes"]):
            keys = set(real_source_keys(entry))
            if source_keys & keys:
                raise ValueError(f"real {split}: duplicate source episode {number}")
            source_keys.update(keys)
            path = REAL / split / f"data/chunk-000/file-{number:03d}.parquet"
            table = pq.read_table(path, columns=["action", "observation.state", *CAMS])
            count = len(table)
            if count != entry["frame_count"]:
                raise ValueError(f"real {split} episode {number} frame count mismatch")
            for key in CAMS:
                raw = table[key][0].as_py()["bytes"]
                image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
                if image is None or image.shape != (IMAGE_HEIGHT, IMAGE_WIDTH, 3):
                    raise ValueError(f"real {split} episode {number} bad {key}")
                images += 1
            frames += count
        report["splits"][split] = {
            "episodes": len(meta["episodes"]),
            "frames": frames,
            "sample_images_checked": images,
            "joint_order": meta["joint_order"],
            "image_size_wh": meta["image_size_wh"],
            "gripper_policy": meta["gripper_policy"],
            "source_episodes": [e["source_episode"] for e in meta["episodes"]],
            "source_identity_keys": sorted(source_keys),
        }
    report["total_episodes"] = sum(v["episodes"] for v in report["splits"].values())
    report["total_frames"] = sum(v["frames"] for v in report["splits"].values())
    report["overlapping_source_keys"] = sorted(
        set(report["splits"]["train"]["source_identity_keys"])
        & set(report["splits"]["validation"]["source_identity_keys"])
    )
    report["train_validation_disjoint"] = not report["overlapping_source_keys"]
    report["validation_limit"] = (
        "Treat validation episodes as development data. A repeatedly inspected "
        "validation set is not an untouched final test set."
    )
    report["normalization"] = (
        "Training reuses the frozen real-data ACTION MEAN_STD and STATE IDENTITY "
        "processors. Real or simulation validation/test frames never update them."
    )
    return report


def audit_collection(output: Path) -> dict:
    output = output.resolve()
    run = json.loads((output / "run.json").read_text())
    index = records(output / "episodes/index.jsonl")
    rows = []
    for entry in index:
        directory = output / entry["path"]
        meta = json.loads((directory / "metadata.json").read_text())
        quality = json.loads((directory / "quality.json").read_text())
        replay = json.loads((directory / "replay.json").read_text())
        with np.load(directory / "numeric.npz") as z:
            state = z["state"]
            action = z["action"]
            valid = z["valid"]
            times = z["time"]
            qpos = z["qpos"]
        if state.shape != (meta["frames"], 16) or action.shape != (meta["frames"], 16):
            raise ValueError(f"{entry['path']}: numeric shape mismatch")
        if qpos.shape != (meta["frames"], 25) or not np.isfinite(qpos).all():
            raise ValueError(f"{entry['path']}: qpos missing or invalid")
        if np.any(~np.isfinite(state)) or np.any(~np.isfinite(action)):
            raise ValueError(f"{entry['path']}: non-finite numeric data")
        if np.max(np.abs(np.diff(times) - 1 / FPS)) > 1e-9:
            raise ValueError(f"{entry['path']}: timestamps are not exact 30 Hz")
        valid_from = int(meta["valid_from_frame"])
        if valid_from is None or not valid[valid_from:].all() or valid[:valid_from].any():
            raise ValueError(f"{entry['path']}: invalid valid mask")
        for role in CAMERAS:
            files = sorted((directory / "cameras" / role).glob("*.jpg"))
            if len(files) != meta["frames"]:
                raise ValueError(f"{entry['path']}: {role} image count mismatch")
            image = cv2.imread(str(files[0]), cv2.IMREAD_COLOR)
            if image is None or image.shape != (IMAGE_HEIGHT, IMAGE_WIDTH, 3):
                raise ValueError(f"{entry['path']}: bad {role} image")
        if not entry["success"] or not replay["success"]:
            raise ValueError(f"{entry['path']}: success/replay flag false")
        rows.append(
            {
                "record_id": str(directory),
                "scenario_keys": scenario_keys(meta, run),
                "seed": int(meta["seed"]),
                "episode_id": entry["episode_id"],
                "collection_root": str(output),
                "path": entry["path"],
                "domain": entry["domain"],
                "group_id": entry["group_id"],
                "family_id": entry["family_id"],
                "recovery_type": entry.get("recovery_type"),
                "frames": meta["frames"],
                "valid_from_frame": valid_from,
                "valid_frames": int(valid.sum()),
                "duration_s": meta["duration_s"],
                "success": meta["success"],
                "replay_success": replay["success"],
                "takeover_reason": meta.get("takeover_reason"),
                "sha256_numeric": digest(directory / "numeric.npz"),
                "sha256_metadata": digest(directory / "metadata.json"),
            }
        )
    normal = [r for r in rows if r["domain"] == "normal"]
    recovery = [r for r in rows if r["domain"] == "recovery"]
    recovery_counts = {
        t: sum(1 for r in recovery if r["recovery_type"] == t)
        for t in (
            "closed_approach",
            "lateral_offset",
            "bad_height",
            "recoverable_drop",
            "pre_release_hold",
        )
    }
    report = {
        "run": run,
        "episodes": len(rows),
        "normal": len(normal),
        "recovery": len(recovery),
        "recovery_counts": recovery_counts,
        "normal_frames": sum(r["frames"] for r in normal),
        "recovery_frames": sum(r["frames"] for r in recovery),
        "recovery_valid_frames": sum(r["valid_frames"] for r in recovery),
        "normal_groups": len({r["group_id"] for r in normal}),
        "recovery_families": len({r["family_id"] for r in recovery}),
        "rows": rows,
        "target": {"normal": 100, "recovery": 50, "recovery_per_type": 10, "recovery_families": 10},
        "complete": bool(
            len(normal) == 100
            and len(recovery) == 50
            and all(v == 10 for v in recovery_counts.values())
            and len({r["family_id"] for r in recovery}) == 10
        ),
    }
    write_json(output / "audit_report.json", report)
    return report


def combined_audit(collections: list[Path]) -> dict:
    if len({Path(c).resolve() for c in collections}) != len(collections):
        raise ValueError("Duplicate collection input")
    reports = [audit_collection(Path(c)) for c in collections]
    if len(reports) == 1:
        return reports[0]
    first = reports[0]["run"]
    for report in reports[1:]:
        run = report["run"]
        for key in ("scene_sha256", "fps", "image_size_hw", "code_sha256", "task"):
            if run.get(key) != first.get(key):
                raise ValueError(f"collection configuration differs in {key}")
    rows = [row for report in reports for row in report["rows"]]
    if len({row["record_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate source episode")
    normal = [r for r in rows if r["domain"] == "normal"]
    recovery = [r for r in rows if r["domain"] == "recovery"]
    counts = {
        t: sum(1 for r in recovery if r["recovery_type"] == t)
        for t in ("closed_approach", "lateral_offset", "bad_height", "recoverable_drop", "pre_release_hold")
    }
    combined = {
        "collections": [str(Path(c).resolve()) for c in collections],
        "run": first,
        "episodes": len(rows),
        "normal": len(normal),
        "recovery": len(recovery),
        "recovery_counts": counts,
        "normal_frames": sum(r["frames"] for r in normal),
        "recovery_frames": sum(r["frames"] for r in recovery),
        "recovery_valid_frames": sum(r["valid_frames"] for r in recovery),
        "normal_groups": len({r["group_id"] for r in normal}),
        "recovery_families": len({r["family_id"] for r in recovery}),
        "rows": rows,
        "target": {"normal": 100, "recovery": 50, "recovery_per_type": 10, "recovery_families": 10},
        "complete": bool(
            len(normal) == 100 and len(recovery) == 50
            and all(v == 10 for v in counts.values())
            and len({r["family_id"] for r in recovery}) == 10
        ),
    }
    return combined


def make_split(audit: dict, seed: int = 20261028) -> dict:
    """Assign connected scene families jointly, retaining exact domain counts."""
    if not audit["complete"]:
        raise ValueError("Collection is incomplete; refusing to split/convert")
    rows = audit["rows"]
    normal = [r for r in rows if r["domain"] == "normal"]
    recovery = [r for r in rows if r["domain"] == "recovery"]
    groups = sorted({("normal", r["group_id"]) for r in normal}
                    | {("recovery", r["family_id"]) for r in recovery})
    if sum(g[0] == "normal" for g in groups) != 100 or sum(g[0] == "recovery" for g in groups) != 10:
        raise ValueError("unexpected group/family counts")
    parent = {g: g for g in groups}

    def find(g):
        while parent[g] != g:
            parent[g] = parent[parent[g]]
            g = parent[g]
        return g

    owner = {}
    for row in rows:
        group = (row["domain"], row["group_id"] if row["domain"] == "normal" else row["family_id"])
        if not row.get("scenario_keys"):
            raise ValueError("Missing canonical scenario keys; re-audit sources")
        for key in row["scenario_keys"]:
            if key in owner:
                parent[find(group)] = find(owner[key])
            else:
                owner[key] = group
    connected = {}
    for group in groups:
        connected.setdefault(find(group), []).append(group)
    components = sorted(connected.values())
    random.Random(seed).shuffle(components)
    # State is normal/recovery-family counts in validation and test. Training
    # receives the remainder. DP avoids seed shopping and never relaxes counts.
    target = (10, 1, 10, 1)
    states = {(0, 0, 0, 0)}
    history = []
    for component in components:
        n = sum(g[0] == "normal" for g in component)
        r = len(component) - n
        layer = {}
        for state in sorted(states):
            for name, delta in (("train", (0, 0, 0, 0)),
                                ("validation", (n, r, 0, 0)),
                                ("test", (0, 0, n, r))):
                dest = tuple(a + b for a, b in zip(state, delta))
                if all(a <= b for a, b in zip(dest, target)) and dest not in layer:
                    layer[dest] = (state, name)
        states = set(layer)
        history.append(layer)
    if target not in states:
        raise ValueError("Shared layouts make exact 80/10/10 and 40/5/5 impossible; preserve sources and revise collection, never split a family")
    assignments = {}
    state = target
    for component, layer in zip(reversed(components), reversed(history)):
        previous, name = layer[state]
        for group in component:
            assignments[group] = name
        state = previous
    split = {"schema": "act_sim_mix_split_v2", "seed": seed,
             "rule": "joint connected family/layout/seed split; no cross-domain scenario leakage",
             "normal": {}, "recovery": {}, "episodes": {"normal": {}, "recovery": {}}}
    key_splits = {}
    for name in ("train", "validation", "test"):
        for domain, source in (("normal", normal), ("recovery", recovery)):
            field = "group_id" if domain == "normal" else "family_id"
            selected = [r for r in source if assignments[(domain, r[field])] == name]
            split[domain][name] = sorted({r[field] for r in selected})
            split["episodes"][domain][name] = sorted(
                (r["record_id"], r[field]) if domain == "normal" else
                (r["record_id"], r[field], r["recovery_type"]) for r in selected)
            for row in selected:
                for key in row["scenario_keys"]:
                    if key in key_splits and key_splits[key] != name:
                        raise ValueError("Canonical scene leakage")
                    key_splits[key] = name
        if len(split["episodes"]["normal"][name]) != {"train": 80, "validation": 10, "test": 10}[name]:
            raise ValueError("normal split count mismatch")
        selected_recovery = [r for r in recovery if assignments[("recovery", r["family_id"])] == name]
        expected = {"train": 8, "validation": 1, "test": 1}[name]
        for kind in ("closed_approach", "lateral_offset", "bad_height", "recoverable_drop", "pre_release_hold"):
            if sum(r["recovery_type"] == kind for r in selected_recovery) != expected:
                raise ValueError(f"Recovery family/type coverage mismatch: {name}/{kind}")
    split["scenario_split"] = key_splits
    split["scenario_keys_disjoint"] = True
    return split


def convert_split(
    collection: Path,
    rows: list[dict],
    output: Path,
    *,
    split_name: str,
) -> dict:
    """Write a LeRobot-v3-compatible compact Parquet split.

    The schema matches the existing real dataset: 16-D float32 action/state and
    image columns stored as ``struct<bytes, path>``.  Original JPEG bytes from
    the raw collection are embedded without transcoding, avoiding the PNG
    expansion of LeRobotDataset.create while keeping the same training reader.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    output.mkdir(parents=True)
    (output / "data/chunk-000").mkdir(parents=True)
    (output / "meta").mkdir()
    manifest_episodes = []
    global_index = 0
    image_struct = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
    for episode_number, row in enumerate(rows):
        directory = Path(row["collection_root"]) / row["path"]
        with np.load(directory / "numeric.npz") as z:
            state = z["state"]
            action = z["action"]
            valid = z["valid"]
        start = int(row["valid_from_frame"])
        if not valid[start:].all():
            raise ValueError("valid mask is not a suffix")
        state = state[start:]
        action = action[start:]
        count = len(state)
        images = {}
        for role in CAMERAS:
            values = []
            folder = directory / "cameras" / role
            for offset in range(start, start + count):
                path = folder / f"{offset:06d}.jpg"
                raw = path.read_bytes()
                if len(raw) < 100:
                    raise ValueError(f"empty JPEG {path}")
                values.append({"bytes": raw, "path": ""})
            images[f"observation.images.{role}"] = pa.array(values, type=image_struct)
        table = pa.table(
            {
                "observation.state": pa.array(
                    state.tolist(), type=pa.list_(pa.float32(), 16)
                ),
                "action": pa.array(action.tolist(), type=pa.list_(pa.float32(), 16)),
                **images,
                "timestamp": pa.array(np.arange(count) / FPS, type=pa.float64()),
                "frame_index": pa.array(np.arange(count), type=pa.int64()),
                "episode_index": pa.array(
                    np.full(count, episode_number), type=pa.int64()
                ),
                "index": pa.array(
                    np.arange(global_index, global_index + count), type=pa.int64()
                ),
                "task_index": pa.array(np.zeros(count), type=pa.int64()),
            }
        )
        pq.write_table(table, output / f"data/chunk-000/file-{episode_number:03d}.parquet")
        manifest_episodes.append(
            {
                **row,
                "converted_episode_index": episode_number,
                "converted_start_frame": start,
                "converted_frames": count,
                "converted_timestamp_start_s": 0.0,
            }
        )
        global_index += count
        print(
            f"{split_name}: saved {row['path']} -> episode {episode_number} "
            f"({count} frames)",
            flush=True,
        )
    manifest = {
        "fps": FPS,
        "image_size_wh": [IMAGE_WIDTH, IMAGE_HEIGHT],
        "joint_order": [
            "openarm_right_joint1",
            "openarm_right_joint2",
            "openarm_right_joint3",
            "openarm_right_joint4",
            "openarm_right_joint5",
            "openarm_right_joint6",
            "openarm_right_joint7",
            "openarm_right_finger_joint1",
            "openarm_left_joint1",
            "openarm_left_joint2",
            "openarm_left_joint3",
            "openarm_left_joint4",
            "openarm_left_joint5",
            "openarm_left_joint6",
            "openarm_left_joint7",
            "openarm_left_finger_joint1",
        ],
        "cameras": [f"observation.images.{role}" for role in CAMERAS],
        "image_storage": "original JPEG bytes embedded in Parquet",
        "action_source": "MuJoCo position-actuator target issued at each 30 Hz sample",
        "state_source": "MuJoCo feedback at the same 30 Hz sample",
        "timestamp_alignment": "exact 30 Hz grid; physics sampled on the next 1 ms step",
        "split": split_name,
        "episodes": manifest_episodes,
    }
    write_json(output / "meta/visual_expert_manifest.json", manifest)
    write_json(
        output / "meta/info.json",
        {
            "codebase_version": "v3.0",
            "fps": FPS,
            "total_episodes": len(manifest_episodes),
            "total_frames": global_index,
            "total_tasks": 1,
            "robot_type": "openarm",
            "splits": [split_name],
            "features": {
                "observation.state": {"dtype": "float32", "shape": [16]},
                "action": {"dtype": "float32", "shape": [16]},
                **{
                    f"observation.images.{role}": {
                        "dtype": "image",
                        "shape": [IMAGE_HEIGHT, IMAGE_WIDTH, 3],
                    }
                    for role in CAMERAS
                },
            },
            "note": (
                "Compact local schema matching the real dataset reader. "
                "Generated without LeRobotDataset.create to avoid PNG re-encoding; "
                "not used to compute ACT normalization statistics."
            ),
        },
    )
    return {
        "split": split_name,
        "episodes": len(manifest_episodes),
        "frames": global_index,
        "manifest": manifest_episodes,
    }


def cmd_audit(args) -> int:
    collections = [p.resolve() for p in args.collection]
    audit = combined_audit(collections)
    real = real_audit() if getattr(args, "real_root", None) else None
    if real is not None:
        write_json(collections[0] / "real_audit.json", real)
        audit["real"] = {k: real[k] for k in ("total_episodes", "total_frames", "train_validation_disjoint", "validation_limit")}
    report_path = args.report or collections[0] / "combined_audit_report.json"
    write_json(report_path, audit)
    print(
        json.dumps(
            {
                k: audit[k]
                for k in (
                    "episodes",
                    "normal",
                    "recovery",
                    "recovery_counts",
                    "normal_frames",
                    "recovery_valid_frames",
                    "complete",
                )
            },
            indent=2,
        )
    )
    if getattr(args, "pilot", False):
        if not audit["rows"] or (audit["recovery"] and any(v < 1 for v in audit["recovery_counts"].values())):
            print("INCOMPLETE PILOT: every requested recovery type needs at least one valid episode")
            return 5
        return 0
    return 0 if audit["complete"] else 5


def cmd_convert(args) -> int:
    collections = [p.resolve() for p in args.collection]
    output = args.output.resolve()
    if output.exists():
        raise SystemExit("conversion output exists; choose a new directory")
    audit = combined_audit(collections)
    split = make_split(audit, args.split_seed)
    real = real_audit() if getattr(args, "real_root", None) else None
    if real is not None and not real["train_validation_disjoint"]:
        raise ValueError("real train/validation sources overlap")
    rows = {r["record_id"]: r for r in audit["rows"]}
    output.mkdir(parents=True)
    write_json(output / "split.json", split)
    if real is not None:
        write_json(output / "real_audit.json", real)
    report = {"collections": [str(c) for c in collections], "split_seed": args.split_seed, "splits": {}}
    for split_name in ("train", "validation", "test"):
        entries = split["episodes"]["normal"][split_name] + split["episodes"]["recovery"][split_name]
        selected = [rows[e[0]] for e in sorted(entries)]
        report["splits"][split_name] = convert_split(
            collections[0], selected, output / split_name, split_name=split_name
        )
    # Numeric range check only; normalization stays frozen on the real data.
    real_stats = json.loads((REAL / "train/meta/stats.json").read_text()) if real is not None else None
    train_rows = [rows[e[0]] for e in split["episodes"]["normal"]["train"] + split["episodes"]["recovery"]["train"]]
    samples = []
    for row in train_rows:
        with np.load(Path(row["collection_root"]) / row["path"] / "numeric.npz") as z:
            samples.append(z["action"][int(row["valid_from_frame"]) :])
    sim_action = np.concatenate(samples, axis=0)
    sim_min = sim_action.min(axis=0).tolist()
    sim_max = sim_action.max(axis=0).tolist()
    real_min = real_stats["action"]["min"] if real_stats else None
    real_max = real_stats["action"]["max"] if real_stats else None
    outside = [
        i
        for i in range(16)
        if real_stats and (sim_min[i] < real_min[i] - 1e-6 or sim_max[i] > real_max[i] + 1e-6)
    ]
    report["range_check"] = {
        "scope": "simulation train valid frames vs real train action min/max; diagnostic only" if real_stats else "simulation train valid frames only; no real dataset supplied",
        "sim_action_min": sim_min,
        "sim_action_max": sim_max,
        "real_action_min": real_min,
        "real_action_max": real_max,
        "axes_outside_real_range": outside,
        "note": "Frozen real MEAN_STD processors are still used; no sim or validation statistics are fitted.",
    }
    if outside:
        print("WARNING: simulation action axes outside real min/max:", outside, flush=True)
    report["real"] = {
        "train_episodes": real["splits"]["train"]["episodes"],
        "train_frames": real["splits"]["train"]["frames"],
        "validation_episodes": real["splits"]["validation"]["episodes"],
        "validation_frames": real["splits"]["validation"]["frames"],
        "validation_limit": real["validation_limit"],
    } if real is not None else None
    report["normalization"] = real["normalization"] if real else "No processors fitted; supply an existing training-only real-data processor for mixed training."
    report["image_size_hw"] = [IMAGE_HEIGHT, IMAGE_WIDTH]
    report["fps"] = FPS
    report["task"] = TASK
    write_json(output / "conversion_report.json", report)
    print(
        json.dumps(
            {
                "splits": {
                    k: {"episodes": v["episodes"], "frames": v["frames"]}
                    for k, v in report["splits"].items()
                },
                "real": report["real"],
            },
            indent=2,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("audit")
    audit.add_argument("--collection", type=Path, action="append", required=True)
    audit.add_argument("--report", type=Path)
    audit.add_argument("--real-root", type=Path, help="Optional read-only audit of an existing train/validation real dataset")
    audit.add_argument("--pilot", action="store_true", help="Audit a small pilot; does not authorize conversion of incomplete batches")
    convert = sub.add_parser("convert")
    convert.add_argument("--collection", type=Path, action="append", required=True)
    convert.add_argument("--output", type=Path, required=True)
    convert.add_argument("--split-seed", type=int, default=20261028)
    convert.add_argument("--real-root", type=Path, help="Optional real-data audit and numeric range comparison")
    return parser


def main() -> int:
    global REAL
    args = build_parser().parse_args()
    if args.real_root:
        REAL = args.real_root.expanduser().resolve()
    if args.command == "audit":
        return cmd_audit(args)
    return cmd_convert(args)


if __name__ == "__main__":
    sys.exit(main())
