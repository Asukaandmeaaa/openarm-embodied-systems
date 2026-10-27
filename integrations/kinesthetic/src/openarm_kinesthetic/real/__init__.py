"""Pure contracts shared by the ROS 2 real-hardware adapter."""

from .operator_protocol import OperatorDecision, OperatorPacket, OperatorWatchdog
from .ready_candidates import (
    APPROACH_AXIS_IN_TARGET_FRAME,
    SIMULATION_FAMILY_HEADINGS_DEG,
    SIMULATION_FAMILY_TILT_DEG,
    OrientationCandidate,
    TableFrame,
    candidates_from_prior,
    deduplicate_orientations,
    matrix_to_xyzw,
    orientation_for_approach,
    prior_tilts_deg,
    quaternion_to_matrix,
    table_frame,
)

__all__ = [
    "APPROACH_AXIS_IN_TARGET_FRAME",
    "OperatorDecision",
    "OperatorPacket",
    "OperatorWatchdog",
    "OrientationCandidate",
    "SIMULATION_FAMILY_HEADINGS_DEG",
    "SIMULATION_FAMILY_TILT_DEG",
    "TableFrame",
    "candidates_from_prior",
    "deduplicate_orientations",
    "matrix_to_xyzw",
    "orientation_for_approach",
    "prior_tilts_deg",
    "quaternion_to_matrix",
    "table_frame",
]
