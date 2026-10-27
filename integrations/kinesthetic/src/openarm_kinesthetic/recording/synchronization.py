"""Timestamp-aware 30 Hz state/action/camera synchronization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SyncReport:
    output_frames: int
    maximum_state_skew_s: float
    maximum_action_skew_s: float
    maximum_camera_skew_s: dict[str, float]
    missing_camera_frames: dict[str, int]
    repeated_camera_frames: dict[str, int]
    policy: dict[str, str]


def _validate_series(times, values, width: int):
    t = np.asarray(times, dtype=float); v = np.asarray(values, dtype=float)
    if t.ndim != 1 or v.shape != (len(t), width) or len(t) < 2: raise ValueError("invalid timestamped vector series")
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(v)) or np.any(np.diff(t) <= 0): raise ValueError("series must be finite and strictly monotonic")
    return t, v


def _nearest_indices(times: np.ndarray, timeline: np.ndarray) -> np.ndarray:
    right = np.searchsorted(times, timeline).clip(max=len(times) - 1)
    left = np.maximum(right - 1, 0)
    return np.where(
        np.abs(times[left] - timeline) <= np.abs(times[right] - timeline),
        left,
        right,
    )


def synchronize_30hz(state_times, states, action_times, actions, camera_times: dict[str, list[float]], *, maximum_camera_skew_s: float = 0.05):
    if not np.isfinite(maximum_camera_skew_s) or maximum_camera_skew_s <= 0.0:
        raise ValueError("maximum camera skew must be finite and positive")
    st, sv = _validate_series(state_times, states, 16); at, av = _validate_series(action_times, actions, 16)
    start = max(st[0], at[0]); end = min(st[-1], at[-1]); timeline = np.arange(start, end + 1e-12, 1.0 / 30.0)
    if start > end or timeline.size == 0:
        raise ValueError("state and action streams do not overlap")
    state_out = np.column_stack([np.interp(timeline, st, sv[:, i]) for i in range(16)]).astype(np.float32)
    action_out = np.column_stack([np.interp(timeline, at, av[:, i]) for i in range(16)]).astype(np.float32)
    nearest_state = _nearest_indices(st, timeline)
    nearest_action = _nearest_indices(at, timeline)
    camera_indices = {}; camera_max = {}; missing = {}; repeated = {}
    for key in ("ceiling", "wrist_right"):
        times = np.asarray(camera_times.get(key, []), dtype=float)
        if times.size == 0:
            camera_indices[key] = np.full(len(timeline), -1, dtype=int); camera_max[key] = float("inf"); missing[key] = len(timeline); repeated[key] = 0; continue
        if times.ndim != 1 or not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0.0):
            raise ValueError(f"camera {key} timestamps must be finite and strictly monotonic")
        indices = _nearest_indices(times, timeline)
        skew = np.abs(times[indices] - timeline); indices = np.where(skew <= maximum_camera_skew_s, indices, -1)
        camera_indices[key] = indices; camera_max[key] = float(np.max(skew)); missing[key] = int(np.sum(indices < 0)); repeated[key] = int(np.sum((indices[1:] == indices[:-1]) & (indices[1:] >= 0)))
    report = SyncReport(
        len(timeline), float(np.max(np.abs(st[nearest_state]-timeline))), float(np.max(np.abs(at[nearest_action]-timeline))),
        camera_max, missing, repeated, {"state_action": "linear", "camera": "nearest_timestamp"},
    )
    return timeline, state_out, action_out, camera_indices, report
