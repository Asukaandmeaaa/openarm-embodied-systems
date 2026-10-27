"""Real-side Teaching Ready orientation candidates, derived from the registered table.

Simulation's job was to narrow the space: it established that the legacy tool-straight-down
orientation is the most J4-loaded of the families (see `docs/READY_SEARCH.md`). The real
station subsequently showed that changing orientation at the old fixed TCP position can leave
the same J4 geometry locked in place, so the base point itself is now operator-selected in
table coordinates and reconstructed from the live registration. This module
deliberately does *not* hand the real station joint values or a JointTrajectory, because
installation error, joint zero offsets, camera extrinsics, collision geometry and
wrist-camera mass all differ from the simulator.

So the real station gets a prior and re-solves everything itself:

    simulation prior (family, region, priority)
      -> this module: real orientation candidates about the REAL registered table frame
      -> caller: registered-table-relative TCP target (plus optional local corrections)
      -> MoveIt collision-aware IK, multi-seed, on the real model
      -> the project's existing joint-space ranking
      -> real OMPL plan
      -> PLAN_ONLY, operator review, low-speed execution

The arithmetic is the same construction the simulation used - the approach axis is tilted
below the horizontal plane by a configured angle and rotated to a heading in the table
plane - but the frames come from the registration here, and the tool frame is the frame the
real IK actually targets, which is ``openarm_right_ee_base_link`` rather than the MuJoCo
grasp site. That difference is exactly why the real side cannot reuse simulation
quaternions: the two frames are 0.138 m and a quarter turn apart.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from ..ready_search.tool_frame import minimal_rotation, rotation_about

# The approach axis expressed in the frame the real MoveIt IK targets.
#
# It is derived, not assumed: in the compiled model `right_grasp_frame` sits at
# (0, 0, -0.138) in `openarm_right_ee_base_link`, so the gripper's long axis in the
# end-effector link frame is -Z. The 1.4 mm lateral offset of the real grasp site is
# ignored, which is a 0.06 degree difference and far below the orientation tolerance.
APPROACH_AXIS_IN_TARGET_FRAME = (0.0, 0.0, -1.0)

# The real TCP frame the capture contract pins: `end_effector_link: openarm_right_ee_base_link`.
TARGET_FRAME = "openarm_right_ee_base_link"


class ReadyCandidateError(ValueError):
    """Raised when the registration cannot produce usable orientation candidates."""


@dataclass(frozen=True)
class TableFrame:
    """The registered table frame, in world coordinates."""

    normal: np.ndarray
    forward: np.ndarray
    lateral: np.ndarray
    forward_source: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "normal_world": self.normal.tolist(),
            "forward_world": self.forward.tolist(),
            "lateral_world": self.lateral.tolist(),
            "forward_source": self.forward_source,
        }


def table_frame(
    anchor_world_m: Sequence[float],
    normal_world: Sequence[float],
    *,
    forward_hint_world: Sequence[float] | None = None,
) -> TableFrame:
    """Build a table frame from the registration.

    When registration supplies ``forward_hint_world`` it is authoritative: table yaw can
    differ from the radial robot-to-center direction, so deriving yaw from a world XYZ
    would discard part of the measurement.  Derivation from the anchor remains only for
    older generic callers without an explicit registered axis.
    """
    normal = np.asarray(normal_world, dtype=float)
    if normal.shape != (3,) or not np.all(np.isfinite(normal)):
        raise ReadyCandidateError("table normal must be three finite values")
    if float(np.linalg.norm(normal)) < 1.0e-9:
        raise ReadyCandidateError("table normal must be non-zero")
    normal = normal / np.linalg.norm(normal)

    anchor = np.asarray(anchor_world_m, dtype=float)
    if anchor.shape != (3,) or not np.all(np.isfinite(anchor)):
        raise ReadyCandidateError("the operation anchor must be three finite values")
    if forward_hint_world is not None:
        hint = np.asarray(forward_hint_world, dtype=float)
        if hint.shape != (3,) or not np.all(np.isfinite(hint)):
            raise ReadyCandidateError("the forward hint must be three finite values")
        planar = hint - normal * float(hint @ normal)
        source = "explicit registered table forward-axis hint"
        if float(np.linalg.norm(planar)) < 1.0e-6:
            raise ReadyCandidateError("the forward hint lies along the table normal")
    else:
        planar = anchor - normal * float(anchor @ normal)
        source = "horizontal component of the registered operation anchor"
        if float(np.linalg.norm(planar)) < 1.0e-6:
            raise ReadyCandidateError(
                "the anchor has no horizontal component, so the table forward axis is undefined; "
                "supply forward_hint_world explicitly"
            )
    forward = planar / np.linalg.norm(planar)
    lateral = np.cross(normal, forward)
    return TableFrame(normal=normal, forward=forward, lateral=lateral, forward_source=source)


@dataclass(frozen=True)
class OrientationCandidate:
    """One real TCP orientation, with the approach direction it encodes."""

    label: str
    family: str
    priority: int
    tilt_below_horizontal_deg: float
    heading_deg: float
    roll_deg: float
    approach_axis_world: np.ndarray
    orientation_matrix: np.ndarray
    orientation_xyzw: tuple[float, float, float, float]

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "family": self.family,
            "priority": int(self.priority),
            "tilt_below_horizontal_deg": float(self.tilt_below_horizontal_deg),
            "heading_deg": float(self.heading_deg),
            "roll_deg": float(self.roll_deg),
            "approach_axis_world": self.approach_axis_world.tolist(),
            "orientation_xyzw": list(self.orientation_xyzw),
        }


def matrix_to_xyzw(matrix: np.ndarray) -> tuple[float, float, float, float]:
    """Rotation matrix to a quaternion in the contract's xyzw order."""
    value = np.asarray(matrix, dtype=float).reshape(3, 3)
    trace = float(np.trace(value))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (value[2, 1] - value[1, 2]) / scale
        y = (value[0, 2] - value[2, 0]) / scale
        z = (value[1, 0] - value[0, 1]) / scale
    elif value[0, 0] > value[1, 1] and value[0, 0] > value[2, 2]:
        scale = math.sqrt(1.0 + value[0, 0] - value[1, 1] - value[2, 2]) * 2.0
        w = (value[2, 1] - value[1, 2]) / scale
        x = 0.25 * scale
        y = (value[0, 1] + value[1, 0]) / scale
        z = (value[0, 2] + value[2, 0]) / scale
    elif value[1, 1] > value[2, 2]:
        scale = math.sqrt(1.0 + value[1, 1] - value[0, 0] - value[2, 2]) * 2.0
        w = (value[0, 2] - value[2, 0]) / scale
        x = (value[0, 1] + value[1, 0]) / scale
        y = 0.25 * scale
        z = (value[1, 2] + value[2, 1]) / scale
    else:
        scale = math.sqrt(1.0 + value[2, 2] - value[0, 0] - value[1, 1]) * 2.0
        w = (value[1, 0] - value[0, 1]) / scale
        x = (value[0, 2] + value[2, 0]) / scale
        y = (value[1, 2] + value[2, 1]) / scale
        z = 0.25 * scale
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    # Plain floats: these tuples end up in JSON/YAML reports, which numpy scalars break.
    return (float(x / norm), float(y / norm), float(z / norm), float(w / norm))


def quaternion_to_matrix(xyzw: Sequence[float]) -> np.ndarray:
    x, y, z, w = (float(value) for value in xyzw)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1.0e-12:
        raise ReadyCandidateError("reference quaternion has zero length")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def orientation_for_approach(
    frame: TableFrame,
    reference_matrix: np.ndarray,
    *,
    tilt_below_horizontal_deg: float,
    heading_deg: float,
    roll_deg: float = 0.0,
    approach_axis_in_target: Sequence[float] = APPROACH_AXIS_IN_TARGET_FRAME,
) -> tuple[np.ndarray, np.ndarray]:
    """Orientation whose tool approach axis points ``tilt`` degrees below horizontal."""
    axis = np.asarray(approach_axis_in_target, dtype=float)
    axis = axis / np.linalg.norm(axis)
    tilt = math.radians(float(tilt_below_horizontal_deg))
    if not -90.0 <= float(tilt_below_horizontal_deg) <= 90.0:
        raise ReadyCandidateError("tilt must lie in [-90, 90] degrees")
    heading = math.radians(float(heading_deg))
    horizontal = math.cos(heading) * frame.forward + math.sin(heading) * frame.lateral
    approach = math.cos(tilt) * horizontal - math.sin(tilt) * frame.normal
    approach = approach / np.linalg.norm(approach)

    base = np.asarray(reference_matrix, dtype=float).reshape(3, 3)
    aligned = minimal_rotation(base @ axis, approach) @ base
    orientation = rotation_about(approach, math.radians(float(roll_deg))) @ aligned
    achieved = orientation @ axis
    if float(np.max(np.abs(achieved - approach))) > 1.0e-9:  # pragma: no cover - numerical guard
        raise ReadyCandidateError("generated orientation does not reproduce the requested approach axis")
    return orientation, approach


def _tilt_of(frame: TableFrame, approach: np.ndarray) -> float:
    return float(math.degrees(math.asin(float(np.clip(-approach @ frame.normal, -1.0, 1.0)))))


def _heading_of(frame: TableFrame, approach: np.ndarray) -> float:
    planar = approach - frame.normal * float(approach @ frame.normal)
    if float(np.linalg.norm(planar)) < 1.0e-9:
        return 0.0
    planar = planar / np.linalg.norm(planar)
    return float(
        math.degrees(
            math.atan2(float(planar @ frame.lateral), float(planar @ frame.forward))
        )
    )


def _label(family: str, tilt: float, heading: float, roll: float) -> str:
    label = f"{family}_{tilt:g}"
    if heading != 0.0:
        label += f"_h{heading:+g}"
    if roll != 0.0:
        label += f"_r{roll:+g}"
    return label


def candidates_from_prior(
    frame: TableFrame,
    reference_xyzw: Sequence[float],
    *,
    tilts_deg: Mapping[str, Sequence[float]],
    headings_deg: Mapping[str, Sequence[float]] | None = None,
    rolls_deg: Sequence[float] = (0.0,),
    include_reference: bool = True,
) -> list[OrientationCandidate]:
    """Real orientation candidates for the search, in priority order.

    ``tilts_deg`` maps a family name to the tilt angles to try around it, and the mapping's
    order is the search priority: the operator lists the families the simulation liked
    first, and the legacy reference orientation stays available as a control.
    """
    candidates: list[OrientationCandidate] = []
    reference_matrix = quaternion_to_matrix(reference_xyzw)
    if include_reference:
        axis = np.asarray(APPROACH_AXIS_IN_TARGET_FRAME, dtype=float)
        axis = axis / np.linalg.norm(axis)
        approach = reference_matrix @ axis
        candidates.append(
            OrientationCandidate(
                label="legacy_reference",
                family="legacy_reference",
                priority=len(tilts_deg),
                tilt_below_horizontal_deg=_tilt_of(frame, approach),
                heading_deg=_heading_of(frame, approach),
                roll_deg=0.0,
                approach_axis_world=approach,
                orientation_matrix=reference_matrix,
                orientation_xyzw=tuple(float(value) for value in reference_xyzw[:4]),
            )
        )

    for priority, (family, tilts) in enumerate(tilts_deg.items()):
        headings = (headings_deg or {}).get(family, (0.0,))
        for tilt in tilts:
            for heading in headings:
                for roll in rolls_deg:
                    label = _label(family, float(tilt), float(heading), float(roll))
                    orientation, approach = orientation_for_approach(
                        frame,
                        reference_matrix,
                        tilt_below_horizontal_deg=float(tilt),
                        heading_deg=float(heading),
                        roll_deg=float(roll),
                    )
                    candidates.append(
                        OrientationCandidate(
                            label=label,
                            family=family,
                            priority=priority,
                            tilt_below_horizontal_deg=float(tilt),
                            heading_deg=float(heading),
                            roll_deg=float(roll),
                            approach_axis_world=approach,
                            orientation_matrix=orientation,
                            orientation_xyzw=matrix_to_xyzw(orientation),
                        )
                    )
    return candidates


def deduplicate_orientations(
    candidates: Sequence[OrientationCandidate], *, minimum_angle_deg: float = 5.0
) -> tuple[list[OrientationCandidate], list[OrientationCandidate]]:
    """Collapse orientations whose approach axes are effectively the same.

    A local tilt sweep can produce two entries a fraction of a degree apart; running IK for
    both is wasted work and it would let one pose occupy several shortlist slots.
    """
    threshold = math.radians(float(minimum_angle_deg))
    kept: list[OrientationCandidate] = []
    dropped: list[OrientationCandidate] = []
    for candidate in candidates:
        duplicate = any(
            math.acos(
                float(np.clip(candidate.approach_axis_world @ other.approach_axis_world, -1.0, 1.0))
            )
            < threshold
            for other in kept
        )
        (dropped if duplicate else kept).append(candidate)
    return kept, dropped


SIMULATION_PRIOR_FAMILIES = ("oblique_60", "oblique_40", "side_horizontal")


# Nominal tilt of each simulation family, in the same convention this module uses: the
# angle of the tool approach axis below the horizontal plane. The legacy fixed orientation
# is the 90 degree end of that scale and stays available as a control through
# `candidates_from_prior(include_reference=True)`.
SIMULATION_FAMILY_TILT_DEG = {
    "oblique_60": 60.0,
    "oblique_40": 40.0,
    "side_horizontal": 0.0,
}

# The heading offsets the simulation searched per family, mirrored here so the real sweep is
# a local correction around the same prior rather than a new design.
SIMULATION_FAMILY_HEADINGS_DEG = {
    "oblique_60": (0.0,),
    "oblique_40": (0.0,),
    "side_horizontal": (-15.0, 0.0, 15.0),
}


def prior_tilts_deg(
    *,
    window_deg: float,
    step_deg: float,
    families: Sequence[str] = SIMULATION_PRIOR_FAMILIES,
) -> dict[str, list[float]]:
    """A local tilt window around each simulation family, in priority order.

    The simulation reported 60 degrees as the best J4 trade-off, 40 degrees as the second
    choice and the horizontal family as the lowest loaded but box-grazing option, so the
    real search refines around those instead of sweeping 0-90 degrees again.
    """
    if step_deg <= 0.0:
        raise ReadyCandidateError("tilt step must be positive")
    if window_deg < 0.0:
        raise ReadyCandidateError("tilt window must be non-negative")
    steps = int(math.floor(float(window_deg) / float(step_deg) + 1e-9))
    result: dict[str, list[float]] = {}
    for family in families:
        nominal = SIMULATION_FAMILY_TILT_DEG.get(str(family))
        if nominal is None:
            raise ReadyCandidateError(
                f"unknown simulation family {family!r}; known: {sorted(SIMULATION_FAMILY_TILT_DEG)}"
            )
        result[str(family)] = [
            nominal + index * float(step_deg) for index in range(-steps, steps + 1)
        ]
    return result


# --------------------------------------------------------------------------------------
# The real search policy
# --------------------------------------------------------------------------------------

CONFIG_NAME = "ready_search_real.yaml"

# What the J4 improvement gate compares against. It is named here so
# ``configs/ready_search_real.yaml`` can declare it and `validate_real_ready_search` can
# refuse any other value, instead of the rule being implied by the search code.
#
# The gate asks "is this candidate materially different from the pose that kept failing".
# That question only has a referent while the legacy pose measured in this run is itself
# inside the failed region. Once the live baseline is already below the commissioning
# ceiling it is not the pose that failed, and demanding a fixed further reduction from it
# is not a risk gate but an unreachable requirement: at a fixed TCP the elbow angle follows
# the target position, so every orientation reports nearly the same J4 and no branch can
# clear a floor the baseline already sits under. See the 2026-09-18 entry in the header of
# `configs/ready_search_real.yaml` and `docs/NANO_TEACHING_READY_COMMISSIONING.md`.
IMPROVEMENT_GATE_REFERENCE_BASELINE_ONLY_WHEN_ABOVE_CEILING = (
    "baseline_only_when_baseline_above_ceiling"
)

# Metrics of a *planned* path, which is what the second selection stage compares. The
# first stage ranks IK solutions at a pose; these describe what moving there costs.
TRAJECTORY_SELECTION_KEYS = (
    "joint4_improvement_rad",
    "maximum_joint_travel_rad",
    "joint_path_length_rad",
    "joint4_travel_rad",
    "maximum_joint4_deviation_rad",
    "duration_s",
    "maximum_velocity_rad_s",
    "maximum_acceleration_rad_s2",
    "point_count",
)


def _read_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ReadyCandidateError(f"{path}: top level must be a mapping")
    return data


@dataclass(frozen=True)
class RealReadySearchPolicy:
    """Where the real search may look, and how strict its commissioning gates are.

    Every number the search uses lives in ``configs/ready_search_real.yaml``; this class
    only reads and validates it. The point is reviewability: an offset buried in a Python
    default is an offset nobody read before it moved the arm.
    """

    search: Mapping[str, Any]
    guards: Mapping[str, Any]
    selection: Mapping[str, Any]
    raw: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path | None = None) -> "RealReadySearchPolicy":
        """Load ``configs/ready_search_real.yaml`` from the workspace or from a path.

        An empty path means "resolve it": launch files pass an empty string when the
        operator did not override it, and that must not be read as the current directory.
        """
        if path is not None and str(path).strip() in {"", "."}:
            path = None
        if path is not None:
            raw = _read_yaml(Path(path).expanduser())
        else:
            try:
                from ..config import load_yaml
                raw = load_yaml(CONFIG_NAME)
            except (ImportError, FileNotFoundError):
                # The recorder package is also usable without the workspace package on
                # PYTHONPATH: walk up from this file to the workspace's configs/ directory.
                candidate = Path(__file__).resolve().parents[3] / "configs" / CONFIG_NAME
                if not candidate.is_file():
                    raise ReadyCandidateError(
                        f"cannot find {CONFIG_NAME}; pass an explicit path"
                    )
                raw = _read_yaml(candidate)
        # Loading is a runtime safety boundary too. A custom --ready-search-config must
        # receive the same validation as the repository copy; relying on an operator to
        # have run a separate validation command leaves the armed CLI with weaker rules.
        from ..config import validate_real_ready_search

        validate_real_ready_search(raw)
        for block in ("scope", "real_ready_search", "trajectory_selection", "commissioning_guards"):
            if not isinstance(raw.get(block), Mapping):
                raise ReadyCandidateError(f"{CONFIG_NAME}: {block} block is missing")
        return cls(
            search=raw["real_ready_search"],
            guards=raw["commissioning_guards"],
            selection=raw["trajectory_selection"],
            raw=raw,
        )

    # -- search region ------------------------------------------------------------------

    def families(self) -> tuple[str, ...]:
        return tuple(str(item) for item in self.search["families"])

    def simulation_prior(self) -> str:
        return str(self.search["simulation_prior"])

    def tilt_window_deg(self) -> float:
        return float(self.search["tilt_window_deg"])

    def tilt_step_deg(self) -> float:
        return float(self.search["tilt_step_deg"])

    def roll_offsets_deg(self) -> tuple[float, ...]:
        return tuple(float(item) for item in self.search["roll_offsets_deg"])

    def headings_deg(self) -> dict[str, tuple[float, ...]]:
        return {
            str(family): tuple(float(item) for item in values)
            for family, values in self.search["heading_offsets_deg"].items()
        }

    def tilts_deg(self) -> dict[str, list[float]]:
        """Tilt angles per family, in family priority order.

        A family listed under ``tilt_offsets_deg`` uses that explicit list; the others get
        the nominal simulation tilt swept within the configured window.
        """
        derived = prior_tilts_deg(
            window_deg=self.tilt_window_deg(),
            step_deg=self.tilt_step_deg(),
            families=self.families(),
        )
        explicit = self.search.get("tilt_offsets_deg") or {}
        for family, tilts in explicit.items():
            derived[str(family)] = [float(item) for item in tilts]
        return derived

    def _offsets(self, axis: str) -> tuple[float, ...]:
        return tuple(
            float(item) for item in self.search["position_offsets_m"][axis]
        )

    def forward_offsets_m(self) -> tuple[float, ...]:
        return self._offsets("forward")

    def lateral_offsets_m(self) -> tuple[float, ...]:
        return self._offsets("lateral")

    def normal_offsets_m(self) -> tuple[float, ...]:
        return self._offsets("normal")

    def minimum_axis_angle_deg(self) -> float:
        return float(self.search["minimum_axis_angle_deg"])

    def candidate_max(self) -> int:
        return int(self.search["candidate_max"])

    def candidate_index(self) -> int:
        return int(self.search["candidate_index"])

    # -- selection ----------------------------------------------------------------------

    def planning_candidate_limit(self) -> int:
        return int(self.selection["planning_candidate_limit"])

    def selection_order(self) -> tuple[str, ...]:
        return tuple(str(item) for item in self.selection["order"])

    # -- commissioning guards -----------------------------------------------------------

    def min_j4_improvement_rad(self) -> float:
        return float(self.guards["min_j4_improvement_rad"])

    def max_target_j4_rad(self) -> float:
        return float(self.guards["max_target_j4_rad"])

    def max_path_absolute_j4_rad(self) -> float:
        return float(self.guards["max_path_absolute_j4_rad"])

    def improvement_gate_reference(self) -> str:
        return str(self.guards["improvement_gate_reference"])

    def legacy_family(self) -> str:
        return str(self.guards["legacy_family"])

    def on_no_admissible_candidate(self) -> str:
        return str(self.guards["on_no_admissible_candidate"])

    def guard_basis(self) -> str:
        return str(self.guards["basis"])

    def known_bad_observations(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(self.raw.get("known_bad_observations", ()))


# --------------------------------------------------------------------------------------
# Second-stage selection: rank *planned* candidates, not just IK solutions
# --------------------------------------------------------------------------------------


def trajectory_path_metrics(
    trajectory, joint_names: Sequence[str], *, joint4_improvement_rad: float
) -> dict[str, float]:
    """What moving along a planned path costs, for the trajectory gate.

    ``trajectory`` is anything with ``joint_names`` and ``points`` (a JointTrajectory
    message or a test double). Every returned key is usable in the configured selection
    order, so the report and the ranking can never disagree about what was compared.
    """
    names = list(trajectory.joint_names)
    expected = [str(name) for name in joint_names]
    if names != expected:
        raise ReadyCandidateError(
            f"planned trajectory joint order {names} does not match {expected}"
        )
    points = list(trajectory.points)
    if len(points) < 2:
        raise ReadyCandidateError("planned trajectory must contain at least two points")
    if len(expected) != 7:
        raise ReadyCandidateError("the real Teaching Ready search is a seven-joint contract")

    path_length = 0.0
    per_joint_travel = [0.0] * len(expected)
    j4_travel = 0.0
    maximum_velocity = 0.0
    maximum_acceleration = 0.0
    maximum_j4_deviation = 0.0
    j4_index = 3
    previous = [float(value) for value in points[0].positions]
    if len(previous) != len(expected) or not all(math.isfinite(value) for value in previous):
        raise ReadyCandidateError("planned trajectory start point has invalid positions")
    maximum_j4_deviation = abs(previous[j4_index])
    for point in points[1:]:
        current_positions = [float(value) for value in point.positions]
        if len(current_positions) != len(expected) or not all(
            math.isfinite(value) for value in current_positions
        ):
            raise ReadyCandidateError("planned trajectory point has invalid positions")
        deltas = [
            abs(now - before)
            for now, before in zip(current_positions, previous)
        ]
        path_length += sum(deltas)
        per_joint_travel = [
            total + delta for total, delta in zip(per_joint_travel, deltas)
        ]
        j4_travel += deltas[j4_index]
        maximum_j4_deviation = max(
            maximum_j4_deviation, abs(current_positions[j4_index])
        )
        previous = current_positions
    for point in points:
        if point.velocities:
            maximum_velocity = max(
                maximum_velocity, max(abs(float(value)) for value in point.velocities)
            )
        if point.accelerations:
            maximum_acceleration = max(
                maximum_acceleration,
                max(abs(float(value)) for value in point.accelerations),
            )

    final = points[-1].time_from_start
    duration = float(final.sec) + float(final.nanosec) * 1.0e-9
    maximum_travel = max(per_joint_travel)
    if not all(
        math.isfinite(value)
        for value in (
            path_length,
            maximum_travel,
            j4_travel,
            maximum_velocity,
            maximum_acceleration,
            duration,
            maximum_j4_deviation,
        )
    ):
        raise ReadyCandidateError("planned trajectory metrics are not finite")
    return {
        "joint4_improvement_rad": float(joint4_improvement_rad),
        "maximum_joint_travel_rad": float(maximum_travel),
        "joint_path_length_rad": float(path_length),
        "joint4_travel_rad": float(j4_travel),
        "maximum_joint4_deviation_rad": float(maximum_j4_deviation),
        "duration_s": float(duration),
        "maximum_velocity_rad_s": float(maximum_velocity),
        "maximum_acceleration_rad_s2": float(maximum_acceleration),
        "point_count": len(points),
    }


def selection_sort_key(metrics: Mapping[str, float], order: Sequence[str]) -> tuple:
    """Lexicographic key over planned-path metrics; a leading ``-`` means larger wins."""
    key: list[float] = []
    for entry in order:
        name = str(entry)
        descending = name.startswith("-")
        metric = name.lstrip("-")
        if metric not in metrics:
            raise ReadyCandidateError(f"selection metric {metric!r} is not a planned-path metric")
        value = float(metrics[metric])
        key.append(-value if descending else value)
    return tuple(key)


def select_planned_candidate(
    attempts: Sequence[Mapping[str, Any]], order: Sequence[str]
) -> Mapping[str, Any] | None:
    """The winner among candidates that actually produced a plan.

    Attempts are ``{"candidate": ..., "planned": bool, "metrics": ...}``. Planning failure
    disqualifies a candidate outright: an unreachable goal is not a candidate, however good
    its IK looked, and that is the whole reason the real side ranks twice like the
    simulator does.
    """
    planned = [item for item in attempts if item.get("planned")]
    if not planned:
        return None
    return min(
        planned,
        key=lambda item: selection_sort_key(item["metrics"], order),
    )
