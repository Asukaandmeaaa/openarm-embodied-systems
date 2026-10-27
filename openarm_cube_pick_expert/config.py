"""Frozen paths and configuration loading for the cube-pick data factory."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "official/openarm_mujoco/v2/cell/demo.xml"
PARAMS = ROOT / "configs/collection/openarm_cube_pick_expert_params.yaml"
METADATA = ROOT / "configs/collection/openarm_cube_pick_metadata.yaml"
CONTRACT = ROOT / "configs/collection/feature_contract.yaml"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def frozen_hashes() -> dict[str, str]:
    return {
        "scene_sha256": sha256(SCENE),
        "params_sha256": sha256(PARAMS),
        "contract_sha256": sha256(CONTRACT),
    }
