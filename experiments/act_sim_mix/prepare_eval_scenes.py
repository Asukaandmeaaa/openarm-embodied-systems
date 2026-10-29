"""Freeze validation or final-test scenes without mixing model selection and test.

Default: 10 home + 5 recovery validation rollouts. Use --split test only after
candidate selection; --split all is a final combined report, never a selection set.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .contracts import digest, robot_asset_binding
from .sim_collect import write_json
from .sim_expert.world import SCENE, ROOT
from .paths import recorded_path


def runtime_binding() -> dict:
    here = Path(__file__).resolve().parent
    paths = [here / "prepare_eval_scenes.py", here / "closed_loop.py", here / "metrics.py",
             here / "sim_expert/world.py", here / "sim_expert/mapping.py",
             here / "sim_reference.py", here / "paths.py", here / "contracts.py", here / "policy_limits.py",
             here / "config/deployment_command_envelope_20261029.json"]
    return {**{str(path.relative_to(ROOT)): digest(path) for path in paths}, **robot_asset_binding()}


def source_directory(entry: dict, fallback: Path | None = None) -> Path:
    root = entry.get("collection_root")
    if root is None and fallback is None:
        raise ValueError("Source has no collection_root; pass --collection explicitly")
    return recorded_path(root if root is not None else fallback).resolve() / entry["path"]


def build_scene_file(converted: Path, *, split: str = "validation", collection: Path | None = None) -> dict:
    split_names = ("validation", "test") if split == "all" else (split,)
    scenes = []
    for name in split_names:
        manifest = json.loads((converted / name / "meta/visual_expert_manifest.json").read_text())
        for entry in manifest["episodes"]:
            directory = source_directory(entry, collection)
            raw = json.loads((directory / "metadata.json").read_text())
            scene = {"id": f"{name}_{entry['domain']}_{entry['converted_episode_index']:03d}",
                     "kind": "home" if entry["domain"] == "normal" else "recovery",
                     "split": name, "seed": int(raw["seed"]),
                     "source_episode": entry["path"], "collection_root": str(directory.parent.parent),
                     "group_id": entry["group_id"], "jitter": raw["jitter"],
                     "scenario_keys": entry["scenario_keys"],
                     "metadata_sha256": digest(directory / "metadata.json"),
                     "numeric_sha256": digest(directory / "numeric.npz")}
            if scene["kind"] == "home":
                scene["start"] = "deployment_zero"
            else:
                scene["recovery_type"] = entry["recovery_type"]
                scene["takeover_frame"] = int(entry["valid_from_frame"])
                with np.load(directory / "numeric.npz") as z:
                    frame = scene["takeover_frame"]
                    if "qpos" not in z.files or not 0 <= frame < len(z["qpos"]):
                        raise ValueError("Missing/invalid takeover qpos")
                    scene["velocity_source"] = "recorded_qvel" if "qvel" in z.files else "legacy_zero_assumption"
            scenes.append(scene)
    expected = len(split_names)
    home = sum(s["kind"] == "home" for s in scenes)
    recovery = len(scenes) - home
    if home != 10 * expected or recovery != 5 * expected:
        raise ValueError(f"Expected {10 * expected} home + {5 * expected} recovery, got {home}+{recovery}")
    cfg = Path(__file__).parent / "sim_expert/config.yaml"
    return {"schema": "act_sim_mix_eval_scenes_v2", "count": len(scenes), "home": home,
            "recovery": recovery, "fixed": True, "split": split,
            "scene_sha256": digest(SCENE), "config_sha256": digest(cfg),
            "runtime_sha256": runtime_binding(),
            "note": "Reuse fixed scenes and order; final test is excluded from model selection. Home starts at deployment zero; simulation dynamics remain approximate.",
            "scenes": sorted(scenes, key=lambda s: s["id"])}


def verify_scene_file(report: dict) -> None:
    cfg = Path(__file__).parent / "sim_expert/config.yaml"
    if report.get("schema") != "act_sim_mix_eval_scenes_v2":
        raise ValueError("Legacy/unverified evaluation scenes; create a new v2 scene file")
    if report["scene_sha256"] != digest(SCENE) or report["config_sha256"] != digest(cfg):
        raise ValueError("Scene/config changed since evaluation freeze")
    if report.get("runtime_sha256") != runtime_binding():
        raise ValueError("Evaluation implementation changed since scene freeze")
    for scene in report["scenes"]:
        directory = recorded_path(scene["collection_root"]) / scene["source_episode"]
        for name, field in (("metadata.json", "metadata_sha256"), ("numeric.npz", "numeric_sha256")):
            if digest(directory / name) != scene[field]:
                raise ValueError(f"Evaluation source changed: {directory / name}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--converted", type=Path, required=True)
    parser.add_argument("--collection", type=Path)
    parser.add_argument("--split", choices=("validation", "test", "all"), default="validation")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("scene file exists; preserve it and reuse it")
    report = build_scene_file(args.converted.resolve(), split=args.split,
                              collection=args.collection.resolve() if args.collection else None)
    verify_scene_file(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive publication: another process cannot overwrite an existing freeze.
    with args.output.open("x") as f:
        f.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: report[k] for k in ("count", "home", "recovery", "split")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
