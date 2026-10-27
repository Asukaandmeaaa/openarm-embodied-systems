"""Teaching Ready SE(3) pose search: simulation-only decision support.

The initialization chain already answers "can the arm reach the Teaching Ready
pose". It does not answer "which Teaching Ready pose should the arm be in", because
it evaluates exactly one TCP pose with exactly one orientation. On 2026-09-17 the
real arm reported a J4 red light / protection event reaching that pose, and a single
fixed orientation leaves the redundancy resolution no freedom to avoid it.

This package adds the missing search, in simulation only:

    table + tool frame -> orientation families -> XYZ neighbourhood
      -> multi-seed collision-aware IK
      -> static pose scoring (joint margin, J4 margin, manipulability, clearance,
         start distance, MuJoCo gravity torque with J4 called out)
      -> Top-K -> initialization trajectory planning -> trajectory scoring
      -> Top-3 report

Two properties are intentional and load-bearing:

* The orientation families are built from the *compiled model's* grasp-site frame and
  from the registered table frame. No world RPY triple is invented and then called
  "horizontal".
* J4 gravity torque is recorded and compared relatively, against the legacy
  baseline measured in the same run. No hard J4 limit and no absolute torque
  threshold is created from a single real observation.
"""

from __future__ import annotations

from .config import ReadySearchConfig, load_ready_search_config
from .metrics import (
    J4_JOINT_NAME,
    clearance_pairs,
    joint_metrics,
    minimum_clearance,
    static_gravity_torque,
    trajectory_gravity_torque,
    yoshikawa_manipulability,
)
from .scoring import resolve_references, static_score, trajectory_score
from .search import Candidate, TeachingReadySearch, run_ready_search
from .tool_frame import OrientationCandidate, ToolFrame, orientation_candidates, tool_frame
from .visualize import apply_candidate, render_candidates, view_candidates

__all__ = [
    "Candidate",
    "J4_JOINT_NAME",
    "OrientationCandidate",
    "ReadySearchConfig",
    "TeachingReadySearch",
    "ToolFrame",
    "apply_candidate",
    "clearance_pairs",
    "joint_metrics",
    "load_ready_search_config",
    "minimum_clearance",
    "orientation_candidates",
    "render_candidates",
    "resolve_references",
    "run_ready_search",
    "static_gravity_torque",
    "static_score",
    "tool_frame",
    "trajectory_gravity_torque",
    "trajectory_score",
    "view_candidates",
    "yoshikawa_manipulability",
]
