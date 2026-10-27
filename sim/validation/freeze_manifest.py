#!/usr/bin/env python3
"""Create reproducible hashes and commands for the frozen G4/G5 artifacts."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess

import mujoco


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw/openarm_cube_pick/expert_pilot_g3_frozen_20260903"
LEROBOT = ROOT / "data/lerobot/openarm_cube_pick_expert_pilot_v3_g4_frozen"
CHECKPOINT = ROOT / "outputs/act_openarm_cube_pick_smoke_g5_500/checkpoints/001000/pretrained_model"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_sha256(paths: list[Path], *, source_only: bool = False) -> tuple[str, int, int]:
    files = []
    for path in paths:
        if path.is_file():
            files.append(path)
        else:
            files.extend(item for item in path.rglob("*") if item.is_file())
    filtered = []
    for path in files:
        relative = path.relative_to(ROOT)
        if any(part in {"__pycache__", ".pytest_cache"} for part in relative.parts):
            continue
        if source_only and path.suffix not in {".py", ".sh", ".yaml", ".xml", ".toml"}:
            continue
        filtered.append(path)
    digest = hashlib.sha256()
    total_bytes = 0
    for path in sorted(set(filtered), key=lambda item: item.relative_to(ROOT).as_posix()):
        relative = path.relative_to(ROOT).as_posix().encode()
        size = path.stat().st_size
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(size.to_bytes(8, "big"))
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        total_bytes += size
    return digest.hexdigest(), len(filtered), total_bytes


def git_state(path: Path) -> dict:
    def run(*args):
        return subprocess.run(
            ["git", "-C", str(path), *args], check=True, text=True,
            stdout=subprocess.PIPE,
        ).stdout.strip()

    status = run("status", "--short")
    return {
        "path": str(path.relative_to(ROOT)),
        "head": run("rev-parse", "HEAD"),
        "dirty": bool(status),
        "status": status.splitlines(),
    }


def lerobot_version() -> str:
    result = subprocess.run(
        [
            "/home/lushilin/miniconda3/envs/vla/bin/python", "-c",
            "import lerobot; print(lerobot.__version__)",
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    return result.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    code_paths = [
        ROOT / "openarm_cube_pick_expert",
        ROOT / "sim/validation",
        ROOT / "scripts",
        ROOT / "tests",
        ROOT / "official/openarm_mujoco/src/openarm_mujoco/v2/joint_resolver.py",
        ROOT / "official/openarm_dataset/src/openarm_dataset",
        ROOT / "official/dora-openarm-evaluation/nodes/dora-openarm-actions-executor/src/dora_openarm_actions_executor/main.py",
    ]
    raw_hash, raw_files, raw_bytes = tree_sha256([RAW])
    lerobot_hash, lerobot_files, lerobot_bytes = tree_sha256([LEROBOT])
    code_hash, code_files, code_bytes = tree_sha256(code_paths, source_only=True)
    checkpoint_hash, checkpoint_files, checkpoint_bytes = tree_sha256([CHECKPOINT])
    g2_hash, g2_files, g2_bytes = tree_sha256(
        [ROOT / "results/openarm_expert/g2_current_frozen"]
    )

    tracked_files = [
        ROOT / "official/openarm_mujoco/v2/cell/demo.xml",
        ROOT / "official/openarm_mujoco/v2/openarm_bimanual.xml",
        ROOT / "configs/collection/feature_contract.yaml",
        ROOT / "configs/collection/openarm_cube_pick_expert_params.yaml",
        ROOT / "results/expert_replay/g4_strict_episode_0.json",
        ROOT / "results/openarm_expert/g1_current_frozen_fixed_seed.json",
        ROOT / "results/openarm_expert/g4_lerobot_full_validation.json",
        ROOT / "results/act_training/act_openarm_cube_pick_g5_500.log",
        ROOT / "results/act_training/g5_1000_loss_summary.json",
        ROOT / "results/act_rollout/g5_1000_seed_20260902.json",
    ]
    if (ROOT / "results/tests/expert_platform_junit.xml").exists():
        tracked_files.append(ROOT / "results/tests/expert_platform_junit.xml")

    repositories = []
    for relative in (
        "official/dora-openarm-data-collection",
        "official/dora-openarm-evaluation",
        "official/dora-openarm-keyboard",
        "official/openarm_dataset",
        "official/openarm_mujoco",
    ):
        repositories.append(git_state(ROOT / relative))

    report = {
        "manifest_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract": {
            "task": "Pick up the orange cube with the right arm.",
            "state_action_order": "right_joint1..7,right_gripper,left_joint1..7,left_gripper",
            "physics_hz": 1000,
            "dataset_hz": 30,
            "cameras": ["ceiling", "wrist_right"],
        },
        "versions": {
            "openarm_dataset": importlib.metadata.version("openarm-dataset"),
            "lerobot": lerobot_version(),
            "mujoco": mujoco.__version__,
        },
        "conversion": {
            "source": str(RAW.relative_to(ROOT)),
            "destination": str(LEROBOT.relative_to(ROOT)),
            "command": ".venv/bin/openarm-dataset-convert data/raw/openarm_cube_pick/expert_pilot_g3_frozen_20260903 data/lerobot/openarm_cube_pick_expert_pilot_v3_g4_frozen --format lerobot_v3.0 --state qpos --fps 30 --success-only",
            "pre_validation_command": ".venv/bin/openarm-dataset-validate --no-update-metadata data/raw/openarm_cube_pick/expert_pilot_g3_frozen_20260903",
            "loader_validation_command": "/home/lushilin/miniconda3/envs/vla/bin/python sim/validation/validate_lerobot.py data/lerobot/openarm_cube_pick_expert_pilot_v3_g4_frozen --output results/openarm_expert/g4_lerobot_full_validation.json",
        },
        "trees": {
            "openarm_raw": {"sha256": raw_hash, "files": raw_files, "bytes": raw_bytes},
            "lerobot_v3": {"sha256": lerobot_hash, "files": lerobot_files, "bytes": lerobot_bytes},
            "source_code": {"sha256": code_hash, "files": code_files, "bytes": code_bytes},
            "act_1000_pretrained_model": {"sha256": checkpoint_hash, "files": checkpoint_files, "bytes": checkpoint_bytes},
            "g2_current_frozen_results": {"sha256": g2_hash, "files": g2_files, "bytes": g2_bytes},
        },
        "files": {
            str(path.relative_to(ROOT)): {
                "sha256": file_sha256(path), "bytes": path.stat().st_size
            }
            for path in tracked_files
        },
        "repositories": repositories,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
