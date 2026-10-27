"""Expert collection contract for the fixed black-box placement task."""

from .contract import (
    BLACK_BOX_CENTER_XY_M,
    BLACK_BOX_SAFE_HALF_RANGE_M,
    TASK_PROMPT,
    CubeIntoBlackBoxEvaluator,
    TaskContract,
)
from .runtime import ACTIVE_ARM, PARKED_ARM, assert_left_is_parked, park_left_arm, parked_left_command

__all__ = [
    "BLACK_BOX_CENTER_XY_M",
    "BLACK_BOX_SAFE_HALF_RANGE_M",
    "TASK_PROMPT",
    "CubeIntoBlackBoxEvaluator",
    "TaskContract",
    "ACTIVE_ARM",
    "PARKED_ARM",
    "park_left_arm",
    "parked_left_command",
    "assert_left_is_parked",
]
