"""Workspace-relative paths that never depend on the process working directory."""

from __future__ import annotations

from pathlib import Path


def workspace_root() -> Path:
    """Return the installed/source workspace root from this module's location."""
    root = Path(__file__).resolve().parents[2]
    marker = root / "configs" / "task_contract.yaml"
    if not marker.is_file():
        raise RuntimeError(f"OpenArm workspace marker is missing: {marker}")
    return root


def workspace_path(*parts: str) -> Path:
    path = workspace_root().joinpath(*parts).resolve()
    root = workspace_root().resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"path escapes workspace: {path}")
    return path


def config_path(name: str) -> Path:
    return workspace_path("configs", name)


def vendor_path(*parts: str) -> Path:
    return workspace_path("vendor", "openarm_mujoco", *parts)
