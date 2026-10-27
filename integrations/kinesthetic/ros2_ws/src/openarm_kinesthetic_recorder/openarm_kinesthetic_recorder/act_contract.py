"""Pure validation for the file contract between ACT inference and ROS."""

from __future__ import annotations

import math

from .recorded_motion import JOINT_NAMES


ACTION_SCHEMA = "openarm.act_action_chunk.v1"
EXPECTED_INFERENCE_API = "predict_action_chunk"
ACTION_ORDER = list(JOINT_NAMES) + [
    f"openarm_left_joint{i}" for i in range(1, 8)
] + ["openarm_left_finger_joint1"]


def _finite_number(value, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def validate_action_chunk(
    payload: object,
    now: float,
    previous_sequence: int,
    previous_observation: int,
) -> dict:
    """Validate one complete action document and return a normalized copy."""
    if not isinstance(payload, dict) or payload.get("schema") != ACTION_SCHEMA:
        raise ValueError("action schema must be openarm.act_action_chunk.v1")
    sequence = payload.get("action_sequence")
    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence <= previous_sequence:
        raise ValueError("action_sequence must increase strictly")
    observation = payload.get("source_observation_sequence")
    if isinstance(observation, bool) or not isinstance(observation, int) or observation <= previous_observation:
        raise ValueError("source_observation_sequence must increase strictly")
    published = _finite_number(payload.get("published_monotonic_s"), "published_monotonic_s")
    valid_until = _finite_number(payload.get("valid_until_monotonic_s"), "valid_until_monotonic_s")
    if valid_until <= published or published > now + 0.25 or valid_until <= now:
        raise ValueError("action monotonic clock window is invalid or expired")
    if payload.get("action_order") != ACTION_ORDER:
        raise ValueError("action_order does not match the model contract")
    if payload.get("inference_api") != EXPECTED_INFERENCE_API:
        raise ValueError("inference_api must be predict_action_chunk")
    if _finite_number(payload.get("action_hz"), "action_hz") != 30.0:
        raise ValueError("action_hz must be exactly 30")
    if payload.get("chunk_size") != 10:
        raise ValueError("chunk_size must be exactly 10")

    actions = payload.get("actions")
    right = payload.get("right_actions")
    if not isinstance(actions, list) or len(actions) != 10:
        raise ValueError("actions must contain 10 vectors")
    if not isinstance(right, list) or len(right) != 10:
        raise ValueError("right_actions must contain 10 vectors")
    normalized_actions = []
    normalized_right = []
    for index, vector in enumerate(actions):
        if not isinstance(vector, list) or len(vector) != 16:
            raise ValueError(f"actions[{index}] must contain 16 values")
        normalized_actions.append([_finite_number(v, f"actions[{index}]") for v in vector])
    for index, vector in enumerate(right):
        if not isinstance(vector, list) or len(vector) != 8:
            raise ValueError(f"right_actions[{index}] must contain 8 values")
        normalized_right.append([_finite_number(v, f"right_actions[{index}]") for v in vector])
        if normalized_right[-1] != normalized_actions[index][:8]:
            raise ValueError(f"right_actions[{index}] does not match actions[{index}][:8]")
    result = dict(payload)
    result.update({
        "action_sequence": sequence,
        "source_observation_sequence": observation,
        "published_monotonic_s": published,
        "valid_until_monotonic_s": valid_until,
        "actions": normalized_actions,
        "right_actions": normalized_right,
    })
    return result
