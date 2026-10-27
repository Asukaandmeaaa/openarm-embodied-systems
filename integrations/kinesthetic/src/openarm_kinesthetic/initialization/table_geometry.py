"""Table geometry with explicit per-field provenance, shared by simulation and the real station.

The whole point of this module is prompt section 7/42: simulation and hardware must
differ *only* in the geometry provider that is plugged in, never in the IK, planner,
state machine or teaching-ready definition.

    SimTableGeometryProvider   -> XY and Z are simulation ground truth
    RealTableGeometryProvider  -> XY is a configured prior, Z is a live depth estimate

Every geometry value carries a `source` label and those labels are part of the
returned mapping, because a prior that is silently relabelled as a measurement is
worse than a missing measurement.

The table's world X is derived, never hardcoded:

    robot_front_reference_x  = model-derived front of the pedestal structure
    table_near_edge_x        = robot_front_reference_x + near_edge_gap
    table_center_x           = table_near_edge_x + size_x / 2

The table is 0.40 m deep along X (away from the robot) and 0.60 m wide along Y
(lateral), i.e. `size` is `[depth, width]` in the model frame where +X points from
the robot towards the table.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

# Provenance labels. `measured` is reserved for physical measurements; a simulation
# candidate or a reported approximation must never be relabelled as `measured`.
SOURCE_MEASURED = "measured"
SOURCE_USER_REPORTED = "user_reported_measurement"
SOURCE_USER_APPROXIMATE = "user_measured_approximate"
SOURCE_CONFIGURED_PRIOR = "configured_prior"
SOURCE_MODEL_DERIVED = "model_derived"
SOURCE_DERIVED = "derived_from_prior"
SOURCE_SIM_GROUND_TRUTH = "simulation_ground_truth"
SOURCE_LIVE_DEPTH = "live_depth_estimated"
SOURCE_PROVISIONAL_SIM = "provisional_simulation_candidate"
SOURCE_RECOMMENDED = "recommended_initial_value"
SOURCE_SIM_SCAN = "simulation_scan_selected"
SOURCE_IK_DERIVED = "IK_derived"
SOURCE_FK_REFERENCE = "FK_reference"

ALLOWED_SOURCES = frozenset(
    {
        SOURCE_MEASURED,
        SOURCE_USER_REPORTED,
        SOURCE_USER_APPROXIMATE,
        SOURCE_CONFIGURED_PRIOR,
        SOURCE_MODEL_DERIVED,
        SOURCE_DERIVED,
        SOURCE_SIM_GROUND_TRUTH,
        SOURCE_LIVE_DEPTH,
        SOURCE_PROVISIONAL_SIM,
        SOURCE_RECOMMENDED,
        SOURCE_SIM_SCAN,
        SOURCE_IK_DERIVED,
        SOURCE_FK_REFERENCE,
    }
)

# Fields that must always carry provenance, so a partial dictionary cannot pass as
# a fully labelled prior.
REQUIRED_PROVENANCE = (
    "center.x",
    "center.y",
    "center.z",
    "size.x",
    "size.y",
    "yaw",
    "near_edge_gap",
    "robot_front_reference.x",
)


class TableGeometryError(ValueError):
    """Raised when a table geometry prior is incomplete or self-inconsistent."""


@dataclass(frozen=True)
class TableGeometry:
    """A dead-reckoned table top in world coordinates plus its provenance."""

    frame: str
    center: np.ndarray  # (3,) world centre of the top surface
    size: np.ndarray  # (2,) full size along X (depth) and Y (width)
    yaw: float
    thickness_m: float
    provenance: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        center = np.asarray(self.center, dtype=float)
        size = np.asarray(self.size, dtype=float)
        if center.shape != (3,) or not np.all(np.isfinite(center)):
            raise TableGeometryError("table centre must be three finite values")
        if size.shape != (2,) or not np.all(np.isfinite(size)) or np.any(size <= 0):
            raise TableGeometryError("table size must be two positive finite values")
        if not np.isfinite(self.yaw):
            raise TableGeometryError("table yaw must be finite")
        if not isinstance(self.frame, str) or not self.frame.strip():
            raise TableGeometryError("table frame must be a non-empty name")
        missing = [name for name in REQUIRED_PROVENANCE if name not in self.provenance]
        if missing:
            raise TableGeometryError("table geometry is missing provenance for: " + ", ".join(missing))
        unknown = {value for value in self.provenance.values() if value not in ALLOWED_SOURCES}
        if unknown:
            raise TableGeometryError("unknown provenance labels: " + ", ".join(sorted(unknown)))
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "size", size)
        object.__setattr__(self, "thickness_m", float(self.thickness_m))
        object.__setattr__(self, "yaw", float(self.yaw))

    @property
    def top_z(self) -> float:
        return float(self.center[2])

    @property
    def half_extent(self) -> np.ndarray:
        return self.size / 2.0

    def contains_xy(self, xy: np.ndarray, margin_m: float = 0.0) -> bool:
        """True when a world XY point lies inside the table footprint plus margin."""
        local = self.to_table_xy(np.asarray(xy, dtype=float))
        return bool(np.all(np.abs(local) + margin_m <= self.half_extent))

    def to_table_xy(self, world_xy: np.ndarray) -> np.ndarray:
        """World XY -> table-relative XY (yaw about the table centre)."""
        delta = np.asarray(world_xy, dtype=float) - self.center[:2]
        cos, sin = np.cos(-self.yaw), np.sin(-self.yaw)
        return np.asarray([cos * delta[0] - sin * delta[1], sin * delta[0] + cos * delta[1]])

    def to_world_xy(self, table_xy: np.ndarray) -> np.ndarray:
        """Table-relative XY -> world XY."""
        local = np.asarray(table_xy, dtype=float)
        cos, sin = np.cos(self.yaw), np.sin(self.yaw)
        rotated = np.asarray([cos * local[0] - sin * local[1], sin * local[0] + cos * local[1]])
        return rotated + self.center[:2]

    def corner_xy(self) -> np.ndarray:
        """The four top-surface corners in world XY, counter-clockwise."""
        half = self.half_extent
        local = np.asarray([[-half[0], -half[1]], [half[0], -half[1]], [half[0], half[1]], [-half[0], half[1]]])
        return np.asarray([self.to_world_xy(point) for point in local])

    def as_dict(self) -> dict[str, Any]:
        return {
            "frame": self.frame,
            "center_xyz_m": self.center.tolist(),
            "size_xy_m": self.size.tolist(),
            "yaw_rad": float(self.yaw),
            "thickness_m": float(self.thickness_m),
            "top_z_m": self.top_z,
            "provenance": dict(sorted(self.provenance.items())),
        }


def _value(node: Any, label: str) -> Any:
    if not isinstance(node, Mapping) or "value" not in node:
        raise TableGeometryError(f"{label}: expected a mapping with a `value` field")
    return node["value"]


def _source(node: Any, label: str) -> str:
    if not isinstance(node, Mapping) or "source" not in node:
        raise TableGeometryError(f"{label}: expected a mapping with a `source` field")
    source = node["source"]
    if source not in ALLOWED_SOURCES:
        raise TableGeometryError(f"{label}: unknown source label {source!r}")
    return str(source)


# --------------------------------------------------------------------------------------
# Model introspection: derive the robot's front structure instead of hardcoding it.
# --------------------------------------------------------------------------------------


def world_geom_aabb(model, data, geom_id: int) -> tuple[np.ndarray, np.ndarray]:
    """World-frame axis-aligned bounding box of one geom.

    Mesh vertices are stored by MuJoCo in the mesh's inertial frame, so the world
    position of a vertex is ``geom_xpos + geom_xmat @ vertex``. Rotating all eight
    corners of the local box is required because the inertial frame is generally
    not axis-aligned with the body.
    """
    import mujoco

    kind = int(model.geom_type[geom_id])
    size = np.asarray(model.geom_size[geom_id], dtype=float)
    if kind == int(mujoco.mjtGeom.mjGEOM_MESH):
        mesh_id = int(model.geom_dataid[geom_id])
        start = int(model.mesh_vertadr[mesh_id])
        count = int(model.mesh_vertnum[mesh_id])
        vertices = np.asarray(model.mesh_vert[start : start + count], dtype=float)
    elif kind == int(mujoco.mjtGeom.mjGEOM_BOX):
        vertices = np.asarray(
            [[sx * size[0], sy * size[1], sz * size[2]] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
        )
    elif kind == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        vertices = np.asarray(
            [[sx * size[0], sy * size[0], sz * size[0]] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
        )
    else:  # cylinder, capsule and ellipsoid all embed their half-length in size[1]
        half_length = size[1] if kind != int(mujoco.mjtGeom.mjGEOM_ELLIPSOID) else size[2]
        radius = np.asarray([size[0], size[0], half_length])
        vertices = np.asarray(
            [[sx * radius[0], sy * radius[1], sz * radius[2]] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
        )
    rotation = np.asarray(data.geom_xmat[geom_id], dtype=float).reshape(3, 3)
    world = vertices @ rotation.T + np.asarray(data.geom_xpos[geom_id], dtype=float)
    return world.min(axis=0), world.max(axis=0)


def robot_front_reference_x(model, data, geom_name: str) -> float:
    """Model-derived X of the robot's front-most static structure.

    Uses the named pedestal collision geom rather than a literal, so a vendor model
    update moves the table with it instead of leaving a stale constant behind.
    """
    import mujoco

    geom_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name))
    if geom_id < 0:
        raise TableGeometryError(f"pedestal geom is missing from the model: {geom_name}")
    _, upper = world_geom_aabb(model, data, geom_id)
    return float(upper[0])


def derive_table_center_x(front_reference_x: float, near_edge_gap_m: float, size_x_m: float) -> float:
    return float(front_reference_x + near_edge_gap_m + size_x_m / 2.0)


# --------------------------------------------------------------------------------------
# Providers. These are the only things that differ between simulation and hardware.
# --------------------------------------------------------------------------------------


def _table_blocks(config: Mapping[str, Any]) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    table = config.get("table")
    if not isinstance(table, Mapping):
        raise TableGeometryError("table: required mapping is missing")
    correction = table.get("correction", {}) or {}
    if not isinstance(correction, Mapping):
        raise TableGeometryError("table.correction must be a mapping")
    return table, correction


def sim_table_geometry(model, data, config: Mapping[str, Any]) -> TableGeometry:
    """Simulation provider: XY and Z are ground truth, X is still derived.

    X is derived rather than read back from the scene on purpose. The derived value
    is then asserted against the compiled scene by the geometry tests, so a drifted
    scene file or a vendor pedestal change fails loudly instead of silently
    pretending to be the prior.
    """
    table, correction = _table_blocks(config)
    size_x = float(_value(table["size_x_m"], "table.size_x_m"))
    size_y = float(_value(table["size_y_m"], "table.size_y_m"))
    gap = float(_value(table["near_edge_gap_from_robot_front_m"], "table.near_edge_gap_from_robot_front_m"))
    geom_name = str(table["robot_front_reference"]["geom"])
    front_x = robot_front_reference_x(model, data, geom_name)
    center = np.asarray(
        [
            derive_table_center_x(front_x, gap, size_x) + float(correction.get("x_m", 0.0)),
            float(_value(table["center_y_m"], "table.center_y_m")) + float(correction.get("y_m", 0.0)),
            float(_value(table["top_z_m"], "table.top_z_m")),
        ],
        dtype=float,
    )
    provenance = {
        "center.x": SOURCE_DERIVED,
        "center.y": _source(table["center_y_m"], "table.center_y_m"),
        "center.z": _source(table["top_z_m"], "table.top_z_m"),
        "size.x": _source(table["size_x_m"], "table.size_x_m"),
        "size.y": _source(table["size_y_m"], "table.size_y_m"),
        "yaw": _source(table["yaw_rad"], "table.yaw_rad"),
        "thickness": _source(table["thickness_m"], "table.thickness_m"),
        "near_edge_gap": _source(table["near_edge_gap_from_robot_front_m"], "table.near_edge_gap_from_robot_front_m"),
        "robot_front_reference.x": str(table["robot_front_reference"]["source"]),
    }
    return TableGeometry(
        frame=str(table.get("frame", "world")),
        center=center,
        size=np.asarray([size_x, size_y], dtype=float),
        yaw=float(_value(table["yaw_rad"], "table.yaw_rad")) + float(correction.get("yaw_rad", 0.0)),
        thickness_m=float(_value(table["thickness_m"], "table.thickness_m")),
        provenance=provenance,
    )


def real_table_geometry(
    config: Mapping[str, Any],
    live_top_z_m: float | None,
    *,
    robot_front_reference_x_m: float | None = None,
) -> TableGeometry:
    """Hardware provider: XY from the configured prior, Z from live depth only.

    Nothing about the real table's X is invented here. The front reference must come
    from a model-derived value supplied by the caller (the URDF/MoveIt model on the
    hardware side), and Z must come from an actual depth estimate. Missing inputs
    raise rather than falling back to a simulation number.
    """
    table, correction = _table_blocks(config)
    if live_top_z_m is None:
        raise TableGeometryError("real table Z requires a live depth estimate")
    if robot_front_reference_x_m is None:
        raise TableGeometryError("real table X requires a model-derived robot front reference")
    size_x = float(_value(table["size_x_m"], "table.size_x_m"))
    size_y = float(_value(table["size_y_m"], "table.size_y_m"))
    gap = float(_value(table["near_edge_gap_from_robot_front_m"], "table.near_edge_gap_from_robot_front_m"))
    center = np.asarray(
        [
            derive_table_center_x(float(robot_front_reference_x_m), gap, size_x) + float(correction.get("x_m", 0.0)),
            float(_value(table["center_y_m"], "table.center_y_m")) + float(correction.get("y_m", 0.0)),
            float(live_top_z_m),
        ],
        dtype=float,
    )
    provenance = {
        "center.x": SOURCE_DERIVED,
        "center.y": _source(table["center_y_m"], "table.center_y_m"),
        "center.z": SOURCE_LIVE_DEPTH,
        "size.x": _source(table["size_x_m"], "table.size_x_m"),
        "size.y": _source(table["size_y_m"], "table.size_y_m"),
        "yaw": _source(table["yaw_rad"], "table.yaw_rad"),
        "thickness": _source(table["thickness_m"], "table.thickness_m"),
        "near_edge_gap": _source(table["near_edge_gap_from_robot_front_m"], "table.near_edge_gap_from_robot_front_m"),
        "robot_front_reference.x": SOURCE_MODEL_DERIVED,
    }
    return TableGeometry(
        frame=str(table.get("frame", "world")),
        center=center,
        size=np.asarray([size_x, size_y], dtype=float),
        yaw=float(_value(table["yaw_rad"], "table.yaw_rad")) + float(correction.get("yaw_rad", 0.0)),
        thickness_m=float(_value(table["thickness_m"], "table.thickness_m")),
        provenance=provenance,
    )


def table_geometry_provider(
    config: Mapping[str, Any],
    *,
    model=None,
    data=None,
    live_top_z_m: float | None = None,
    robot_front_reference_x_m: float | None = None,
) -> TableGeometry:
    """Select the provider named by ``initialization.simulation.geometry_source``."""
    source = str(config.get("simulation", {}).get("geometry_source", "ground_truth"))
    if source == "ground_truth":
        if model is None or data is None:
            raise TableGeometryError("the ground-truth provider needs a compiled model and data")
        return sim_table_geometry(model, data, config)
    if source in {"configured_prior", "live_depth"}:
        return real_table_geometry(
            config, live_top_z_m, robot_front_reference_x_m=robot_front_reference_x_m
        )
    raise TableGeometryError(f"unknown table geometry source: {source!r}")


def table_geometry_summary(geometry: TableGeometry) -> dict[str, Any]:
    """Flat report block for the initialization artifacts."""
    return {
        "table_center_x": float(geometry.center[0]),
        "table_center_y": float(geometry.center[1]),
        "table_top_z": geometry.top_z,
        "table_size_x": float(geometry.size[0]),
        "table_size_y": float(geometry.size[1]),
        "table_yaw": float(geometry.yaw),
        "provenance": dict(sorted(geometry.provenance.items())),
    }
