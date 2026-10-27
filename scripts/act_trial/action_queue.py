"""Ten-step ACT inference/execution barrier. No ROS or hardware access."""
from __future__ import annotations

import numpy as np

SCHEDULE = "queued_prefix_v1"
ACTION_STEPS = 10
INTERVAL_NS = 33_333_333


def contract(profile=None):
    if profile == 'visual32':
        from scripts.act_trial.rolling_actions import contract as rolling_contract
        return rolling_contract()
    return {"name": SCHEDULE, "predicted_steps": 50, "executed_steps": ACTION_STEPS,
            "interval_ns": INTERVAL_NS, "old_tail_blend": False,
            "next_inference": "fresh_observation_after_executor_completion"}


def prefix(actions):
    positions = np.asarray(actions["positions"], dtype=np.float32)
    if (positions.shape != (50, 16) or not np.isfinite(positions).all()
            or int(actions["interval"]) != INTERVAL_NS):
        raise ValueError("queued ACT requires finite 50x16 predictions at 30 Hz")
    return positions[:ACTION_STEPS].copy()


class InferenceGate:
    """Only a matching completion unlocks inference on a newer observation."""

    def __init__(self):
        self.inflight = None
        self.observation_after_ns = 0

    def accepts(self, metadata):
        return (self.inflight is None
                and int(metadata.get("observation_monotonic_ns", -1)) >= self.observation_after_ns)

    def dispatch(self, chunk_id):
        if self.inflight is not None:
            raise ValueError("previous ten-step prefix has not completed")
        self.inflight = chunk_id

    def complete(self, metadata, now_ns):
        if self.inflight is None or metadata.get("queue_chunk_id") != self.inflight:
            raise ValueError("unexpected ACT prefix completion")
        completed = int(metadata["completed_monotonic_ns"])
        if (int(metadata["executed_raw_steps"]) != ACTION_STEPS
                or not self.observation_after_ns <= completed <= now_ns):
            raise ValueError("invalid ACT prefix completion")
        self.inflight = None
        self.observation_after_ns = completed


def sample_times(control_hz=100):
    """Ten 30-Hz slots; hold action nine during its final complete slot.

    Only indices 0..9 may affect interpolation. The tenth slot ends at 10/30s,
    rather than at action nine's timestamp (9/30s).
    """
    if not np.isfinite(control_hz) or control_hz <= 0:
        raise ValueError("invalid ACT executor frequency")
    duration_s = ACTION_STEPS * INTERVAL_NS / 1e9
    times = np.arange(0., duration_s, 1. / control_hz)
    evaluation = np.minimum(times, (ACTION_STEPS - 1) * INTERVAL_NS / 1e9)
    return times, evaluation, duration_s
