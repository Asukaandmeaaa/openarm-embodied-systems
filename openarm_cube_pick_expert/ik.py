"""IK diagnostics contract used by the OpenArm motion planner."""

from dataclasses import dataclass


@dataclass(frozen=True)
class IKDiagnostic:
    candidate_index: int
    phase: str
    attempt: int
    success: bool
    reason: str | None = None
