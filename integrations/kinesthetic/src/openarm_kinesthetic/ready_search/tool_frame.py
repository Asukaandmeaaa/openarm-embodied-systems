"""Orientation families built from the compiled model's tool frame and the table frame.

The prompt for this work is explicit that "horizontal" and "30 to 60 degrees" must be
generated from the real TCP/tool frame, and that assuming a world RPY triple equals a
meaningful tool direction is forbidden. This module is where that rule is enforced.

Two frames are derived, never assumed
-------------------------------------
Tool frame. ``right_grasp_frame`` is offset from its parent wrist body
``openarm_right_ee_base_link``. That offset, expressed in the TCP frame, *is* the
gripper's approach axis, and it is read straight out of the compiled model
(``model.site_pos`` / ``model.site_quat``), not written down as a constant. On this
robot it comes out as ``+X`` of the site frame, which is a result, not an input.

Table frame. The normal is the compiled ``table_top_collision`` geom's own local +Z,
and the forward axis is checked against the registered table yaw. If the scene table
were tilted or rotated relative to the registered prior, this module raises instead of
silently generating "horizontal" poses about the wrong plane.

Family construction
-------------------
``tilt_below_horizontal_deg`` is the angle between the approach axis and the
horizontal plane, measured downwards. 0 deg is horizontal, 90 deg points straight
down. The previously fixed orientation turns out to be the 90 deg end of that same
scale (its approach axis is within a degree of world -Z), so the legacy baseline and
the new families are directly comparable members of one parameterization rather than
two unrelated conventions.

The approach axis fixes two rotational degrees of freedom. The third - roll about the
approach axis - is resolved as the smallest rotation away from the reference
orientation, so a family differs from the legacy pose in *where the tool points*
rather than in an arbitrary twist. Roll offsets can be swept explicitly on top.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..initialization.table_geometry import TableGeometry
from ..scene.model import name_id

TOOL_SITE = "right_grasp_frame"
TOOL_PARENT_BODY = "openarm_right_ee_base_link"
TABLE_TOP_GEOM = "table_top_collision"

KIND_REFERENCE = "reference_orientation"
KIND_TILTED = "tilted_approach"

# A site sitting exactly on its parent body origin gives no approach direction at all.
_MINIMUM_TOOL_LENGTH_M = 1e-4
# Tolerance for the scene table agreeing with the registered prior.
TABLE_AXIS_TOLERANCE_RAD = math.radians(2.0)
TABLE_NORMAL_TOLERANCE_RAD = math.radians(2.0)


class ToolFrameError(ValueError):
    """Raised when the tool or table frame cannot be derived from the compiled model."""


def rotation_about(axis: Sequence[float], angle_rad: float) -> np.ndarray:
    """Rodrigues rotation about a (not necessarily unit) axis."""
    unit = np.asarray(axis, dtype=float)
    norm = float(np.linalg.norm(unit))
    if norm <= 1e-12:
        raise ValueError("rotation axis must be non-zero")
    unit = unit / norm
    cross = np.asarray(
        [[0.0, -unit[2], unit[1]], [unit[2], 0.0, -unit[0]], [-unit[1], unit[0], 0.0]], dtype=float
    )
    return np.eye(3) + math.sin(angle_rad) * cross + (1.0 - math.cos(angle_rad)) * (cross @ cross)


def minimal_rotation(from_axis: Sequence[float], to_axis: Sequence[float]) -> np.ndarray:
    """Smallest rotation taking ``from_axis`` onto ``to_axis``."""
    first = np.asarray(from_axis, dtype=float)
    second = np.asarray(to_axis, dtype=float)
    first = first / np.linalg.norm(first)
    second = second / np.linalg.norm(second)
    cross = np.cross(first, second)
    cosine = float(np.clip(first @ second, -1.0, 1.0))
    if cosine > 1.0 - 1e-12:
        return np.eye(3)
    if cosine < -1.0 + 1e-12:
        # Antiparallel: any perpendicular axis gives a valid 180 degree rotation, so
        # pick it deterministically from the smallest component.
        seed = np.eye(3)[int(np.argmin(np.abs(first)))]
        perpendicular = seed - first * float(first @ seed)
        return rotation_about(perpendicular, math.pi)
    return rotation_about(cross, math.atan2(float(np.linalg.norm(cross)), cosine))


@dataclass(frozen=True)
class ToolFrame:
    """The derived tool frame and the derived table frame, both in world coordinates."""

    approach_axis_tool: np.ndarray
    lateral_axis_tool: np.ndarray
    third_axis_tool: np.ndarray
    tool_length_m: float
    approach_axis_is_model_axis: bool
    approach_axis_alignment_deg: float
    table_normal_world: np.ndarray
    table_forward_world: np.ndarray
    table_lateral_world: np.ndarray
    table_scene_forward_error_deg: float
    table_scene_normal_error_deg: float
    table_geom_name: str
    table_yaw_rad: float

    def approach_axis_world(self, orientation: np.ndarray) -> np.ndarray:
        """World approach direction of a TCP orientation: ``R @ approach_axis_tool``."""
        return np.asarray(orientation, dtype=float).reshape(3, 3) @ self.approach_axis_tool

    def tilt_below_horizontal_deg(self, orientation: np.ndarray) -> float:
        """Angle of the approach axis below the table plane, in degrees."""
        axis = self.approach_axis_world(orientation)
        return float(math.degrees(math.asin(float(np.clip(-axis @ self.table_normal_world, -1.0, 1.0)))))

    def heading_deg(self, orientation: np.ndarray) -> float:
        """In-plane heading of the approach axis, measured from the table forward axis."""
        axis = self.approach_axis_world(orientation)
        planar = axis - self.table_normal_world * float(axis @ self.table_normal_world)
        norm = float(np.linalg.norm(planar))
        if norm <= 1e-9:
            return 0.0
        planar = planar / norm
        return float(
            math.degrees(math.atan2(float(planar @ self.table_lateral_world), float(planar @ self.table_forward_world)))
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "tool_site": TOOL_SITE,
            "tool_parent_body": TOOL_PARENT_BODY,
            "tool_length_m": float(self.tool_length_m),
            "approach_axis_tool_frame": self.approach_axis_tool.tolist(),
            "lateral_axis_tool_frame": self.lateral_axis_tool.tolist(),
            "third_axis_tool_frame": self.third_axis_tool.tolist(),
            "approach_axis_alignment_deg": float(self.approach_axis_alignment_deg),
            "approach_axis_is_a_model_axis": bool(self.approach_axis_is_model_axis),
            "table_normal_world": self.table_normal_world.tolist(),
            "table_forward_world": self.table_forward_world.tolist(),
            "table_lateral_world": self.table_lateral_world.tolist(),
            "table_yaw_rad": float(self.table_yaw_rad),
            "table_geom": self.table_geom_name,
            "table_scene_forward_error_deg": float(self.table_scene_forward_error_deg),
            "table_scene_normal_error_deg": float(self.table_scene_normal_error_deg),
            "derivation": (
                "approach axis = model.site_pos[right_grasp_frame] expressed in the "
                "right_grasp_frame frame; table normal = compiled table_top_collision geom local +Z"
            ),
        }


def derive_approach_axis(model, site_id: int) -> tuple[np.ndarray, float, float]:
    """Model-derived tool approach axis in the site frame, plus its length and alignment.

    The vector from the site's parent body origin to the site origin *is* the gripper's
    long axis, and expressing it in the site's own frame gives the approach direction.
    Reading it out of ``model.site_pos`` / ``model.site_quat`` means the axis cannot
    drift from the geometry, unlike a hardcoded unit vector.

    Returns ``(axis_in_site_frame, tool_length_m, alignment_to_nearest_frame_axis_deg)``.
    """
    import mujoco

    offset_in_parent = np.asarray(model.site_pos[site_id], dtype=float)
    length = float(np.linalg.norm(offset_in_parent))
    if length <= _MINIMUM_TOOL_LENGTH_M:
        raise ToolFrameError(
            f"site {site_id} sits on its parent body origin, so the model does not define an approach axis"
        )
    wxyz = np.asarray(model.site_quat[site_id], dtype=float)
    site_rotation = np.empty(9, dtype=float)
    mujoco.mju_quat2Mat(site_rotation, wxyz / np.linalg.norm(wxyz))
    axis = site_rotation.reshape(3, 3).T @ (offset_in_parent / length)
    axis = axis / np.linalg.norm(axis)
    alignment_deg = float(math.degrees(math.acos(float(np.clip(np.max(np.abs(axis)), -1.0, 1.0)))))
    return axis, length, alignment_deg


def tool_frame(model, data, geometry: TableGeometry) -> ToolFrame:
    """Derive the tool and table frames from the compiled model."""
    import mujoco

    mujoco.mj_forward(model, data)
    site = name_id(model, mujoco.mjtObj.mjOBJ_SITE, TOOL_SITE)
    approach, length, alignment_deg = derive_approach_axis(model, site)

    # Build an orthonormal tool frame around the derived approach axis.
    seed = np.eye(3)[int(np.argmin(np.abs(approach)))]
    lateral = seed - approach * float(approach @ seed)
    lateral = lateral / np.linalg.norm(lateral)
    third = np.cross(approach, lateral)
    # How far the derived axis is from the nearest TCP frame axis. Reported because a
    # large value would mean the tool frame is not axis-aligned and "approach" needs
    # re-examining, not because any particular value is required.
    is_model_axis = bool(alignment_deg < 1.0)

    table_geom = name_id(model, mujoco.mjtObj.mjOBJ_GEOM, TABLE_TOP_GEOM)
    geom_rotation = np.asarray(data.geom_xmat[table_geom], dtype=float).reshape(3, 3)
    normal = geom_rotation[:, 2]
    normal = normal / np.linalg.norm(normal)
    scene_forward = geom_rotation[:, 0]
    scene_forward = scene_forward - normal * float(scene_forward @ normal)
    scene_forward = scene_forward / np.linalg.norm(scene_forward)

    yaw = float(geometry.yaw)
    forward = np.asarray([math.cos(yaw), math.sin(yaw), 0.0], dtype=float)
    lateral_world = np.asarray([-math.sin(yaw), math.cos(yaw), 0.0], dtype=float)
    up = np.asarray([0.0, 0.0, 1.0], dtype=float)

    normal_error = math.acos(float(np.clip(abs(normal @ up), -1.0, 1.0)))
    if normal_error > TABLE_NORMAL_TOLERANCE_RAD:
        raise ToolFrameError(
            f"compiled {TABLE_TOP_GEOM} is not horizontal: normal {normal.tolist()} is "
            f"{math.degrees(normal_error):.3f} deg from world +Z"
        )
    forward_error = math.acos(float(np.clip(scene_forward @ forward, -1.0, 1.0)))
    if forward_error > TABLE_AXIS_TOLERANCE_RAD:
        raise ToolFrameError(
            f"compiled {TABLE_TOP_GEOM} yaw disagrees with the registered table prior by "
            f"{math.degrees(forward_error):.3f} deg"
        )
    normal = up * float(np.sign(normal @ up))

    return ToolFrame(
        approach_axis_tool=approach,
        lateral_axis_tool=lateral,
        third_axis_tool=third,
        tool_length_m=length,
        approach_axis_is_model_axis=is_model_axis,
        approach_axis_alignment_deg=alignment_deg,
        table_normal_world=normal,
        table_forward_world=forward,
        table_lateral_world=lateral_world,
        table_scene_forward_error_deg=float(math.degrees(forward_error)),
        table_scene_normal_error_deg=float(math.degrees(normal_error)),
        table_geom_name=TABLE_TOP_GEOM,
        table_yaw_rad=yaw,
    )


@dataclass(frozen=True)
class OrientationCandidate:
    """One TCP orientation, with the approach direction it was generated from."""

    family: str
    label: str
    kind: str
    tilt_below_horizontal_deg: float
    heading_deg: float
    roll_deg: float
    approach_axis_world: np.ndarray
    orientation_matrix: np.ndarray
    reference_distance_rad: float
    source: str

    def as_dict(self) -> dict[str, Any]:
        from ..initialization.teaching_ready import rotation_to_quaternion_xyzw

        return {
            "family": self.family,
            "label": self.label,
            "kind": self.kind,
            "source": self.source,
            "tilt_below_horizontal_deg": float(self.tilt_below_horizontal_deg),
            "heading_deg": float(self.heading_deg),
            "roll_deg": float(self.roll_deg),
            "approach_axis_world": self.approach_axis_world.tolist(),
            "orientation_matrix": self.orientation_matrix.tolist(),
            "orientation_xyzw": rotation_to_quaternion_xyzw(self.orientation_matrix),
            "reference_orientation_distance_rad": float(self.reference_distance_rad),
        }


def orientation_for_approach(
    frame: ToolFrame,
    reference_matrix: np.ndarray,
    *,
    tilt_below_horizontal_deg: float,
    heading_deg: float,
    roll_deg: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Orientation whose tool approach axis points ``tilt`` degrees below horizontal."""
    tilt = math.radians(float(tilt_below_horizontal_deg))
    heading = math.radians(float(heading_deg))
    horizontal = math.cos(heading) * frame.table_forward_world + math.sin(heading) * frame.table_lateral_world
    approach = math.cos(tilt) * horizontal - math.sin(tilt) * frame.table_normal_world
    approach = approach / np.linalg.norm(approach)

    base = np.asarray(reference_matrix, dtype=float).reshape(3, 3)
    aligned = minimal_rotation(base @ frame.approach_axis_tool, approach) @ base
    orientation = rotation_about(approach, math.radians(float(roll_deg))) @ aligned
    achieved = orientation @ frame.approach_axis_tool
    if float(np.max(np.abs(achieved - approach))) > 1e-9:  # pragma: no cover - numerical guard
        raise ToolFrameError("generated orientation does not reproduce the requested approach axis")
    return orientation, approach


def _measure_candidate(
    frame: ToolFrame,
    reference_matrix: np.ndarray,
    *,
    family: str,
    label: str,
    kind: str,
    orientation: np.ndarray,
    roll_deg: float,
    source: str,
) -> OrientationCandidate:
    from ..initialization.teaching_ready import rotation_error_rad

    return OrientationCandidate(
        family=family,
        label=label,
        kind=kind,
        tilt_below_horizontal_deg=frame.tilt_below_horizontal_deg(orientation),
        heading_deg=frame.heading_deg(orientation),
        roll_deg=float(roll_deg),
        approach_axis_world=frame.approach_axis_world(orientation),
        orientation_matrix=np.asarray(orientation, dtype=float).reshape(3, 3),
        reference_distance_rad=rotation_error_rad(orientation, np.asarray(reference_matrix, dtype=float)),
        source=source,
    )


def _number(value: float) -> str:
    return f"{float(value):g}"


def _field(family: Mapping[str, Any] | Any, name: str, default: Any = None) -> Any:
    """Read a family attribute whether it is a dataclass or a plain mapping."""
    if isinstance(family, Mapping):
        return family.get(name, default)
    return getattr(family, name, default)


def orientation_candidates(
    frame: ToolFrame,
    reference_matrix: np.ndarray,
    families: Iterable[Mapping[str, Any] | Any],
) -> list[OrientationCandidate]:
    """All orientation candidates, in the configured family order.

    Deterministic ordering matters: the report is compared between runs and the
    acceptance criterion is that the Top N is reproducible.
    """
    candidates: list[OrientationCandidate] = []
    for family in families:
        name = str(_field(family, "name"))
        kind = str(_field(family, "kind"))
        if not bool(_field(family, "enabled", True)):
            continue
        if kind == KIND_REFERENCE:
            reference = np.asarray(reference_matrix, dtype=float).reshape(3, 3)
            candidates.append(
                _measure_candidate(
                    frame,
                    reference,
                    family=name,
                    label=name,
                    kind=kind,
                    orientation=reference,
                    roll_deg=0.0,
                    source="forward_kinematics_of_reference_pose",
                )
            )
            continue
        if kind != KIND_TILTED:
            raise ToolFrameError(f"unknown orientation family kind: {kind!r}")
        tilts = tuple(float(value) for value in _field(family, "tilts_below_horizontal_deg", ()))
        headings = tuple(float(value) for value in _field(family, "heading_offsets_deg", (0.0,)))
        rolls = tuple(float(value) for value in _field(family, "roll_offsets_deg", (0.0,)))
        base_heading = float(_field(family, "base_heading_deg", 0.0))
        for tilt in tilts:
            for heading_offset in headings:
                for roll in rolls:
                    label = f"{name}_{_number(tilt)}"
                    if heading_offset != 0.0:
                        label += f"_h{heading_offset:+g}"
                    if roll != 0.0:
                        label += f"_r{roll:+g}"
                    orientation, _ = orientation_for_approach(
                        frame,
                        reference_matrix,
                        tilt_below_horizontal_deg=tilt,
                        heading_deg=base_heading + heading_offset,
                        roll_deg=roll,
                    )
                    candidates.append(
                        _measure_candidate(
                            frame,
                            reference_matrix,
                            family=name,
                            label=label,
                            kind=kind,
                            orientation=orientation,
                            roll_deg=roll,
                            source="tool_frame_tilt_and_table_heading",
                        )
                    )
    return candidates


def with_roll_offset(
    frame: ToolFrame,
    reference_matrix: np.ndarray,
    candidate: OrientationCandidate,
    roll_offset_deg: float,
) -> OrientationCandidate:
    """Rotate a candidate about its own approach axis by an extra roll offset."""
    orientation = rotation_about(candidate.approach_axis_world, math.radians(float(roll_offset_deg)))
    orientation = orientation @ candidate.orientation_matrix
    label = candidate.label if roll_offset_deg == 0.0 else f"{candidate.label}_r{roll_offset_deg:+g}"
    return _measure_candidate(
        frame,
        reference_matrix,
        family=candidate.family,
        label=label,
        kind=candidate.kind,
        orientation=orientation,
        roll_deg=candidate.roll_deg + float(roll_offset_deg),
        source=candidate.source + "+roll_offset",
    )
