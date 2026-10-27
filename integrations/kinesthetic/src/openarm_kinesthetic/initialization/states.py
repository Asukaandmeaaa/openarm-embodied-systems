"""Initialization states and error codes, kept distinguishable on purpose.

Prompt section 38/39 requires that perception, IK, planning, execution and manual
teaching never collapse into one vague state, and that an operator seeing a failure
can tell immediately whether it is a simulation-model, vision, table-placement, IK,
planning or hardware problem. That property is enforced structurally here: every
error code belongs to exactly one category, and the state machine refuses to skip
from one stage to the next.
"""

from __future__ import annotations

from dataclasses import dataclass, field

SIM_STATES = (
    "SIM_SCENE_READY",
    "SIM_TABLE_READY",
    "SIM_COMPUTE_TEACH_READY",
    "SIM_IK_READY",
    "SIM_PLAN_READY",
    "SIM_EXECUTING_INIT",
    "SIM_INITIALIZATION_REACHED",
    "SIM_MANUAL_TEACHING",
    "SIM_ACCEPTANCE_PASS",
)

REAL_STATES = (
    "REAL_WAIT_CAMERA",
    "REAL_ESTIMATE_TABLE_HEIGHT",
    "REAL_TABLE_STABLE",
    "REAL_TABLE_PREVIEW",
    "REAL_TABLE_COMMITTED",
    "REAL_CAPTURE_START_STATE",
    "REAL_COMPUTE_TEACH_READY",
    "REAL_IK_READY",
    "REAL_PLAN_READY",
    "REAL_PLAN_ONLY_ACCEPTED",
    "REAL_WAIT_EXECUTION_CONFIRM",
    "REAL_INITIALIZATION_EXECUTING",
    "REAL_INITIALIZATION_REACHED",
    "REAL_TEACH_READY_RIGID",
    "REAL_MANUAL_TEACHING",
)

BOOT_STATE = "BOOT"

# Error code -> category. The category is what the operator needs in order to know
# where to look; the code is what a log or a test asserts on.
ERROR_CATEGORIES: dict[str, str] = {
    "SIM_TABLE_CONFIG_INVALID": "simulation_model",
    "SIM_TEACH_READY_IK_FAILED": "ik",
    "SIM_INITIALIZATION_PLAN_FAILED": "planning",
    "SIM_INITIALIZATION_EXECUTION_FAILED": "simulation_execution",
    "CAMERA_TF_ERROR": "vision",
    "TABLE_HEIGHT_NO_PLANE": "vision",
    "TABLE_HEIGHT_INVALID_NORMAL": "vision",
    "TABLE_HEIGHT_UNSTABLE": "vision",
    "TABLE_PRIOR_MISALIGNED": "table_placement",
    "TEACH_READY_IK_FAILED": "ik",
    "INITIALIZATION_PLAN_FAILED": "planning",
    "START_STATE_CHANGED": "planning",
    "EXECUTION_NOT_ENABLED": "hardware_execution",
    "EXECUTION_CONFIRMATION_REQUIRED": "hardware_execution",
    "EXECUTION_FAILED": "hardware_execution",
    "MANUAL_TEACHING_NOT_READY": "handoff",
}

# Aliases keep the prompt's simulation-side names working while every code still
# resolves to one category.
ERROR_CATEGORIES["SIM_INITIALIZATION_EXECUTION_FAILED"] = "simulation_execution"
ERROR_CATEGORIES["SIM_SCENE_GEOMETRY_MISMATCH"] = "simulation_model"

# Allowed forward transitions. Anything else raises, so a missing stage cannot be
# reported as a later success.
_TRANSITIONS: dict[str, tuple[str, ...]] = {
    BOOT_STATE: ("SIM_SCENE_READY", "REAL_WAIT_CAMERA", "SIM_TABLE_CONFIG_INVALID", "CAMERA_TF_ERROR"),
    "SIM_SCENE_READY": ("SIM_TABLE_READY",),
    "SIM_TABLE_READY": ("SIM_COMPUTE_TEACH_READY",),
    "SIM_COMPUTE_TEACH_READY": ("SIM_IK_READY", "SIM_TEACH_READY_IK_FAILED", "SIM_TABLE_CONFIG_INVALID"),
    "SIM_IK_READY": ("SIM_PLAN_READY", "SIM_INITIALIZATION_PLAN_FAILED"),
    "SIM_PLAN_READY": ("SIM_EXECUTING_INIT", "SIM_INITIALIZATION_EXECUTION_FAILED"),
    "SIM_EXECUTING_INIT": ("SIM_INITIALIZATION_REACHED", "SIM_INITIALIZATION_EXECUTION_FAILED"),
    "SIM_INITIALIZATION_REACHED": ("SIM_MANUAL_TEACHING", "MANUAL_TEACHING_NOT_READY"),
    "SIM_MANUAL_TEACHING": ("SIM_ACCEPTANCE_PASS",),
    "REAL_WAIT_CAMERA": ("REAL_ESTIMATE_TABLE_HEIGHT", "CAMERA_TF_ERROR"),
    "REAL_ESTIMATE_TABLE_HEIGHT": (
        "REAL_TABLE_STABLE",
        "TABLE_HEIGHT_NO_PLANE",
        "TABLE_HEIGHT_INVALID_NORMAL",
        "TABLE_HEIGHT_UNSTABLE",
    ),
    "REAL_TABLE_STABLE": ("REAL_TABLE_PREVIEW", "TABLE_PRIOR_MISALIGNED"),
    "REAL_TABLE_PREVIEW": ("REAL_TABLE_COMMITTED",),
    "REAL_TABLE_COMMITTED": ("REAL_CAPTURE_START_STATE",),
    "REAL_CAPTURE_START_STATE": ("REAL_COMPUTE_TEACH_READY", "START_STATE_CHANGED"),
    "REAL_COMPUTE_TEACH_READY": ("REAL_IK_READY", "TEACH_READY_IK_FAILED"),
    "REAL_IK_READY": ("REAL_PLAN_READY", "INITIALIZATION_PLAN_FAILED"),
    "REAL_PLAN_READY": (
        "REAL_PLAN_ONLY_ACCEPTED",
        "REAL_WAIT_EXECUTION_CONFIRM",
        "INITIALIZATION_PLAN_FAILED",
        "START_STATE_CHANGED",
    ),
    "REAL_PLAN_ONLY_ACCEPTED": ("REAL_WAIT_EXECUTION_CONFIRM",),
    "REAL_WAIT_EXECUTION_CONFIRM": (
        "REAL_INITIALIZATION_EXECUTING",
        "EXECUTION_NOT_ENABLED",
        "EXECUTION_CONFIRMATION_REQUIRED",
        "START_STATE_CHANGED",
    ),
    "REAL_INITIALIZATION_EXECUTING": ("REAL_INITIALIZATION_REACHED", "EXECUTION_FAILED", "EXECUTION_FAILED"),
    "REAL_INITIALIZATION_REACHED": ("REAL_TEACH_READY_RIGID",),
    "REAL_TEACH_READY_RIGID": ("REAL_MANUAL_TEACHING", "MANUAL_TEACHING_NOT_READY"),
}


class StateTransitionError(RuntimeError):
    """Raised when the initialization sequence would skip a required stage."""


def category_of(code: str) -> str:
    try:
        return ERROR_CATEGORIES[code]
    except KeyError as exc:  # pragma: no cover - programming error
        raise KeyError(f"unknown initialization error code: {code}") from exc


def is_error_code(code: str) -> bool:
    return code in ERROR_CATEGORIES


@dataclass
class InitializationStateMachine:
    """Records the ordered stages of one initialization attempt.

    Real hardware execution has a hard interlock here: the machine cannot enter
    ``REAL_INITIALIZATION_EXECUTING`` unless the caller explicitly asserts that
    execution has been enabled and confirmed.
    """

    state: str = BOOT_STATE
    history: list[str] = field(default_factory=list)
    error: str | None = None

    def advance(self, state: str, *, execution_enabled: bool = False, operator_confirmed: bool = False) -> str:
        if state == "REAL_INITIALIZATION_EXECUTING":
            if not execution_enabled:
                raise StateTransitionError("EXECUTION_NOT_ENABLED: real execution is disabled by configuration")
            if not operator_confirmed:
                raise StateTransitionError(
                    "EXECUTION_CONFIRMATION_REQUIRED: the operator must inspect the plan before execution"
                )
        if state in ERROR_CATEGORIES:
            self.error = state
        else:
            allowed = _TRANSITIONS.get(self.state, ())
            if state not in allowed:
                raise StateTransitionError(f"illegal initialization transition: {self.state} -> {state}")
        self.state = state
        self.history.append(state)
        return state

    def fail(self, code: str) -> str:
        if not is_error_code(code):
            raise KeyError(f"unknown initialization error code: {code}")
        return self.advance(code)

    def as_dict(self) -> dict[str, object]:
        return {
            "state": self.state,
            "error": self.error,
            "error_category": category_of(self.error) if self.error else None,
            "history": list(self.history),
        }
