#!/usr/bin/env python3
"""Offline timing check for the ROS ACT FJT chunk scheduler.

This intentionally has no ROS dependency.  It reproduces the observed 10
point/30 Hz chunks, a 0.47 s arrival period, the current 0.32 s FJT horizon,
and the official Hermite + 250 Hz biquad path.  It reports where the old goal
has already reached its final point and how much the first point of the next
chunk jumps.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


class HermiteUpsampler:
    def __init__(self, chunk_hz: float, horizon_sec: float):
        self.chunk_hz = float(chunk_hz)
        self.dt_chunk = 1.0 / self.chunk_hz
        self.t_chunk = np.arange(0.0, horizon_sec + 1e-12, self.dt_chunk)

    @staticmethod
    def _compute_slopes(t, y):
        slopes = np.diff(y, axis=0) / np.diff(t)[:, None]
        result = np.zeros_like(y)
        result[0] = slopes[0]
        result[-1] = slopes[-1]
        for i in range(1, len(y) - 1):
            product = slopes[i - 1] * slopes[i]
            result[i] = 0.5 * (slopes[i - 1] + slopes[i])
            result[i][product <= 0.0] = 0.0
        return result

    def upsample(self, values, times):
        y = np.asarray(values, dtype=np.float32)
        slopes = self._compute_slopes(self.t_chunk, y)
        index = np.clip(np.searchsorted(self.t_chunk, times, side="right") - 1, 0, len(y) - 2)
        t0, t1 = self.t_chunk[index], self.t_chunk[index + 1]
        h = t1 - t0
        u = (times - t0) / h
        h00 = 2 * u**3 - 3 * u**2 + 1
        h10 = (u**3 - 2 * u**2 + u) * h
        h01 = -2 * u**3 + 3 * u**2
        h11 = (u**3 - u**2) * h
        return (
            h00[:, None] * y[index]
            + h10[:, None] * slopes[index]
            + h01[:, None] * y[index + 1]
            + h11[:, None] * slopes[index + 1]
        ).astype(np.float32)


class BiquadLowpass:
    def __init__(self, fs: float, fc: float, q: float = 0.707):
        w0 = 2 * np.pi * float(fc) / float(fs)
        alpha = np.sin(w0) / (2 * float(q))
        a0 = 1 + alpha
        self.b0 = ((1 - np.cos(w0)) / 2) / a0
        self.b1 = (1 - np.cos(w0)) / a0
        self.b2 = self.b0
        self.a1 = (-2 * np.cos(w0)) / a0
        self.a2 = (1 - alpha) / a0
        self.x1 = self.x2 = self.y1 = self.y2 = None

    def step(self, value):
        value = np.asarray(value, dtype=np.float32)
        if self.x1 is None:
            self.x1 = self.x2 = self.y1 = self.y2 = value.copy()
        output = self.b0 * value + self.b1 * self.x1 + self.b2 * self.x2 - self.a1 * self.y1 - self.a2 * self.y2
        self.x2, self.x1 = self.x1, value
        self.y2, self.y1 = self.y1, output
        return output.astype(np.float32)


def blend_remaining(previous, new, consumed):
    remaining = previous[consumed:]
    count = min(len(remaining), len(new))
    if not count:
        return np.asarray(new, dtype=np.float32), count
    weights = np.linspace(1.0, 0.0, count, dtype=np.float32)[:, None]
    return np.concatenate((remaining[:count] * weights + new[:count] * (1 - weights), new[count:])) , count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-interval", type=float, default=0.47)
    parser.add_argument("--start-delay", type=float, default=0.02)
    parser.add_argument("--chunk-hz", type=float, default=30.0)
    parser.add_argument("--control-hz", type=float, default=250.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    t_raw = np.arange(10, dtype=np.float64) / args.chunk_hz
    # A deterministic multi-joint trajectory with a deliberate cross-chunk
    # change.  The metric is independent of the absolute robot pose.
    first = np.stack([0.15 * np.sin(t_raw * (j + 1)) for j in range(8)], axis=1)
    second = first[-1] + np.stack([0.45 + 0.04 * j + 0.1 * np.sin(t_raw * (j + 2)) for j in range(8)], axis=1)
    consumed = min(10, max(0, int((args.chunk_interval - args.start_delay) * args.chunk_hz)))
    blended, blend_count = blend_remaining(first, second, consumed)
    raw_jump = float(np.max(np.abs(second[0] - first[-1])))
    blended_jump = float(np.max(np.abs(blended[0] - first[-1])))
    horizon = args.start_delay + (len(first) - 1) / args.chunk_hz

    upsampler = HermiteUpsampler(args.chunk_hz, (len(first) - 1) / args.chunk_hz)
    eval_times = np.arange(0.0, (len(first) - 1) / args.chunk_hz + 1e-9, 1.0 / args.control_hz)
    lowpass = BiquadLowpass(args.control_hz, 15.0)
    official_first = np.asarray([lowpass.step(value) for value in upsampler.upsample(first, eval_times)])
    lowpass_second = np.asarray([lowpass.step(value) for value in upsampler.upsample(second, eval_times)])
    official_jump = float(np.max(np.abs(lowpass_second[0] - official_first[-1])))
    report = {
        "chunk_interval_s": args.chunk_interval,
        "chunk_hz": args.chunk_hz,
        "chunk_points": 10,
        "start_delay_s": args.start_delay,
        "fjt_first_to_last_s": 9 / args.chunk_hz,
        "fjt_last_point_from_accept_s": horizon,
        "next_chunk_arrival_after_last_point_s": args.chunk_interval - horizon,
        "consumed_raw_steps_at_next_chunk": consumed,
        "blend_count": blend_count,
        "raw_cross_chunk_jump_max_rad": raw_jump,
        "current_blend_cross_chunk_jump_max_rad": blended_jump,
        "official_hermite_lowpass_points": len(official_first),
        "official_cross_chunk_jump_max_rad": official_jump,
        "conclusion": (
            "at 0.47 s the current 0.32 s FJT goal is complete before the next chunk; "
            "the current blend has no overlap and cannot remove the new first-point jump"
            if consumed == 10 else
            "the next chunk overlaps the current trajectory and blend overlap is present"
        ),
    }
    output = json.dumps(report, indent=2, sort_keys=True)
    print(output)
    if args.output:
        args.output.write_text(output + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
