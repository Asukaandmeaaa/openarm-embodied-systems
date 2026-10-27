"""Parse and sanity-check ``configs/ready_search.yaml``.

The dataclasses here are a thin typed view over the YAML: the report needs the raw
mapping (so a reader can see exactly what was configured), while the search code
should not be picking numbers out of nested dictionaries by hand.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from ..config import load_yaml, validate_named
from ..initialization.pipeline import reference_pose_from_poses
from ..paths import workspace_path

CONFIG_NAME = "ready_search.yaml"

KIND_REFERENCE = "reference_orientation"
KIND_TILTED = "tilted_approach"

MODE_CONFIGURED = "configured"
MODE_LEGACY_BASELINE = "legacy_baseline"

LEGACY_FAMILY = "legacy_pi2"

# The single family a new candidate has to beat. Named rather than derived from
# position in the list so reordering the YAML cannot silently change the baseline.
TILT_CONVENTION = "angle_below_horizontal_deg"
ROLL_RESOLUTION = "minimal_rotation_from_reference"


@dataclass(frozen=True)
class FamilyConfig:
    name: str
    enabled: bool
    kind: str
    base_heading_deg: float = 0.0
    tilts_below_horizontal_deg: tuple[float, ...] = ()
    heading_offsets_deg: tuple[float, ...] = (0.0,)
    roll_offsets_deg: tuple[float, ...] = (0.0,)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "enabled": self.enabled,
            "kind": self.kind,
            "base_heading_deg": self.base_heading_deg,
            "tilts_below_horizontal_deg": list(self.tilts_below_horizontal_deg),
            "heading_offsets_deg": list(self.heading_offsets_deg),
            "roll_offsets_deg": list(self.roll_offsets_deg),
        }


@dataclass(frozen=True)
class Reference:
    """A scoring reference scale: either a fixed number or a legacy-baseline metric."""

    mode: str
    value: float | None = None
    metric: str | None = None
    fallback_value: float | None = None
    source: str | None = None

    def resolve(self, legacy_metrics: Mapping[str, float] | None) -> tuple[float, str, float | None]:
        """Return ``(value, resolved_from, legacy_measured_value)``."""
        if self.mode == MODE_CONFIGURED:
            if self.value is None:
                raise ValueError("configured reference is missing `value`")
            return float(self.value), "configured", None
        if self.mode == MODE_LEGACY_BASELINE:
            measured: float | None = None
            if legacy_metrics is not None and self.metric is not None:
                raw = legacy_metrics.get(self.metric)
                if raw is not None and np.isfinite(float(raw)) and float(raw) > 0.0:
                    measured = float(raw)
            if measured is not None:
                return measured, "legacy_baseline", measured
            if self.fallback_value is None or not float(self.fallback_value) > 0.0:
                raise ValueError(f"legacy_baseline reference {self.metric!r} needs a positive fallback_value")
            return float(self.fallback_value), "legacy_baseline_fallback", None
        raise ValueError(f"unknown reference mode: {self.mode!r}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "value": self.value,
            "metric": self.metric,
            "fallback_value": self.fallback_value,
            "source": self.source,
        }


@dataclass(frozen=True)
class ReadySearchConfig:
    raw: Mapping[str, Any]
    initialization: Mapping[str, Any]
    poses: Mapping[str, Any]
    scene: str
    families: tuple[FamilyConfig, ...]
    coarse_offsets_m: np.ndarray
    refine_enabled: bool
    refine_seed_cells: int
    refine_window_m: float
    refine_step_m: float
    refine_roll_offsets_deg: tuple[float, ...]
    footprint_margin_m: float
    minimum_height_above_table_m: float
    maximum_height_above_table_m: float
    ik: Mapping[str, Any]
    filters: Mapping[str, Any]
    metrics: Mapping[str, Any]
    static_weights: Mapping[str, float]
    static_references: Mapping[str, Reference]
    trajectory_endpoint_weight: float
    trajectory_weights: Mapping[str, float]
    trajectory_references: Mapping[str, Reference]
    top_k: int
    top_n: int
    acceptance: Mapping[str, Any]
    report: Mapping[str, Any]
    visualization: Mapping[str, Any]
    known_bad_observations: tuple[Mapping[str, Any], ...]
    extra: Mapping[str, Any] = field(default_factory=dict)

    @property
    def start_pose_name(self) -> str:
        return str(self.raw.get("search", {}).get("start_pose", "simulation_initial_pose"))

    @property
    def reference_pose_name(self) -> str:
        return str(self.raw.get("search", {}).get("reference_pose", "right_task_ready"))

    @property
    def ik_seed_source(self) -> str:
        return str(self.ik.get("seed_source", "initialization_pipeline_ik_seeds"))

    def start_q(self) -> np.ndarray:
        """The initialization start state, read from ``initialization.yaml``.

        ``simulation_initial_pose`` means "whatever the simulation chain installs as
        its current state", which on this station is the physical vertical-down motor
        zero. Real hardware always reads ``/joint_states``; this value is a
        simulation-only convention, which is exactly why `scope.mode` is pinned.
        """
        from ..initialization.pipeline import simulation_start_q

        if self.start_pose_name != "simulation_initial_pose":
            raise ValueError(f"unsupported search.start_pose: {self.start_pose_name!r}")
        return simulation_start_q(self.initialization)

    def reference_q(self) -> np.ndarray:
        return reference_pose_from_poses(self.initialization)

    def table_anchor_offsets(self) -> tuple[float, float, float]:
        """The existing Teaching Ready anchor: table-relative XY offsets and height."""
        block = self.initialization.get("teaching_ready", {})
        return (
            float(block["offset_x_m"]["value"]),
            float(block["offset_y_m"]["value"]),
            float(block["height_above_table_m"]["value"]),
        )

    @property
    def position_anchor_source(self) -> str:
        return str(self.raw["search"]["position"]["anchor_source"])

    def position_anchor_table_offsets(self) -> tuple[float, float]:
        block = self.raw["search"]["position"]
        return (
            float(block.get("anchor_table_offset_x_m", 0.0)),
            float(block.get("anchor_table_offset_y_m", 0.0)),
        )

    def enabled_families(self) -> tuple[FamilyConfig, ...]:
        return tuple(family for family in self.families if family.enabled)

    def legacy_family(self) -> FamilyConfig | None:
        for family in self.families:
            if family.name == LEGACY_FAMILY:
                return family
        return None

    def j4_limits_are_never_defined(self) -> bool:
        """Structural statement used by the report, not a computed metric."""
        return True


def _numbers(node: Any, label: str) -> tuple[float, ...]:
    if not isinstance(node, (list, tuple)) or not node:
        raise ValueError(f"{label}: expected a non-empty list of numbers")
    return tuple(float(item) for item in node)


def _reference(node: Any, label: str) -> Reference:
    if not isinstance(node, Mapping):
        raise ValueError(f"{label}: expected a mapping")
    return Reference(
        mode=str(node["mode"]),
        value=float(node["value"]) if node.get("value") is not None else None,
        metric=str(node["metric"]) if node.get("metric") is not None else None,
        fallback_value=float(node["fallback_value"]) if node.get("fallback_value") is not None else None,
        source=str(node["source"]) if node.get("source") is not None else None,
    )


def coarse_offset_grid(block: Mapping[str, Any]) -> np.ndarray:
    """Cartesian product of the coarse XYZ offsets, in a fixed deterministic order."""
    xs = _numbers(block["x_offsets_m"], "position.coarse.x_offsets_m")
    ys = _numbers(block["y_offsets_m"], "position.coarse.y_offsets_m")
    zs = _numbers(block["z_offsets_m"], "position.coarse.z_offsets_m")
    return np.asarray([[x, y, z] for x in xs for y in ys for z in zs], dtype=float)


def refinement_offsets(center_m: np.ndarray, window_m: float, step_m: float) -> np.ndarray:
    """Offsets within a half-open cube around ``center_m``, nearest first.

    ``center_m`` is itself always included, so a refined pass can never lose the
    coarse candidate it was seeded from.
    """
    if step_m <= 0.0 or window_m <= 0.0:
        return np.zeros((1, 3), dtype=float)
    steps = int(np.floor(window_m / step_m + 1e-9))
    values = [round(index * step_m, 9) for index in range(-steps, steps + 1)]
    offsets: list[tuple[float, float, float, float]] = []
    for dx in values:
        for dy in values:
            for dz in values:
                offsets.append((abs(dx) + abs(dy) + abs(dz), dx, dy, dz))
    offsets.sort()
    return np.asarray([[dx, dy, dz] for _, dx, dy, dz in offsets], dtype=float)


def load_ready_search_config(overrides: Mapping[str, Any] | None = None) -> ReadySearchConfig:
    """Load, validate and parse ``ready_search.yaml``.

    ``overrides`` is a shallow recursive merge used by the tests (and by the
    ``--quick`` CLI profile) to shrink the grid without writing a second config file.
    """
    raw: dict[str, Any] = load_yaml(CONFIG_NAME)
    if overrides:
        raw = _merge(raw, overrides)
    validate_named(CONFIG_NAME, raw)

    initialization = load_yaml(str(raw["sources"]["initialization"]))
    poses = load_yaml(str(raw["sources"]["poses"]))
    scene = raw.get("scene") or str(initialization["initialization"]["scene"])
    if not str(scene).startswith("sim/"):
        raise ValueError(f"scene must live under sim/: {scene!r}")

    orientation_block = raw["orientations"]
    families = tuple(
        FamilyConfig(
            name=str(item["name"]),
            enabled=bool(item.get("enabled", True)),
            kind=str(item["kind"]),
            base_heading_deg=float(item.get("base_heading_deg", 0.0)),
            tilts_below_horizontal_deg=_numbers(
                item.get("tilt_below_horizontal_deg", [0.0]), f"{item['name']}.tilt_below_horizontal_deg"
            ),
            heading_offsets_deg=_numbers(
                item.get("heading_offsets_deg", [0.0]), f"{item['name']}.heading_offsets_deg"
            ),
            roll_offsets_deg=_numbers(item.get("roll_offsets_deg", [0.0]), f"{item['name']}.roll_offsets_deg"),
        )
        for item in orientation_block["families"]
    )

    position = raw["search"]["position"]
    refine = position.get("refine", {})
    scoring = raw["scoring"]
    static_weights = {str(key): float(value) for key, value in scoring["static"]["weights"].items()}
    static_references = {str(key): _reference(value, f"scoring.static.references.{key}")
                         for key, value in scoring["static"]["references"].items()}
    trajectory_weights = {str(key): float(value) for key, value in scoring["trajectory"]["weights"].items()}
    trajectory_references = {str(key): _reference(value, f"scoring.trajectory.references.{key}")
                             for key, value in scoring["trajectory"]["references"].items()}

    return ReadySearchConfig(
        raw=raw,
        initialization=initialization,
        poses=poses,
        scene=str(scene),
        families=families,
        coarse_offsets_m=coarse_offset_grid(position["coarse"]),
        refine_enabled=bool(refine.get("enabled", False)),
        refine_seed_cells=int(refine.get("seed_cells", 0)),
        refine_window_m=float(refine.get("window_m", 0.03)),
        refine_step_m=float(refine.get("step_m", 0.02)),
        refine_roll_offsets_deg=_numbers(refine.get("roll_offsets_deg", [0.0]), "position.refine.roll_offsets_deg"),
        footprint_margin_m=float(position["footprint_margin_m"]),
        minimum_height_above_table_m=float(position["minimum_height_above_table_m"]),
        maximum_height_above_table_m=float(position["maximum_height_above_table_m"]),
        ik=raw["ik"],
        filters=raw["filters"],
        metrics=raw["metrics"],
        static_weights=static_weights,
        static_references=static_references,
        trajectory_endpoint_weight=float(scoring["trajectory"]["endpoint_weight"]),
        trajectory_weights=trajectory_weights,
        trajectory_references=trajectory_references,
        top_k=int(raw["planning"]["top_k"]),
        top_n=int(raw["report"]["top_n"]),
        acceptance=raw["acceptance"],
        report=raw["report"],
        visualization=raw["visualization"],
        known_bad_observations=tuple(raw.get("known_bad_observations", ())),
    )


def _merge(base: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    merged = {key: value for key, value in base.items()}
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def quick_overrides() -> dict[str, Any]:
    """A reduced but structurally identical profile, used by the CLI and the tests."""
    return {
        "search": {
            "position": {
                "coarse": {"x_offsets_m": [0.0], "y_offsets_m": [0.0], "z_offsets_m": [0.0]},
                "refine": {"enabled": False, "seed_cells": 0},
            }
        },
        "orientations": {
            "families": [
                {"name": LEGACY_FAMILY, "enabled": True, "kind": KIND_REFERENCE},
                {
                    "name": "side_horizontal",
                    "enabled": True,
                    "kind": KIND_TILTED,
                    "base_heading_deg": 90.0,
                    "tilt_below_horizontal_deg": [0.0],
                    # The fixed operator point has a valid horizontal solution at 45 deg
                    # table heading (base 90 + offset -45).
                    "heading_offsets_deg": [-45.0],
                    "roll_offsets_deg": [0.0],
                },
                {
                    "name": "oblique",
                    "enabled": True,
                    "kind": KIND_TILTED,
                    "base_heading_deg": 0.0,
                    "tilt_below_horizontal_deg": [30.0, 60.0],
                    "heading_offsets_deg": [0.0],
                    "roll_offsets_deg": [0.0],
                },
            ]
        },
        "planning": {"top_k": 2},
        "report": {"top_n": 2},
        "visualization": {"render": False},
    }


def report_output_path(config: ReadySearchConfig, name: str | None = None):
    output = workspace_path("results", "ready_search", name or str(config.report["output_name"]))
    output.parent.mkdir(parents=True, exist_ok=True)
    return output
