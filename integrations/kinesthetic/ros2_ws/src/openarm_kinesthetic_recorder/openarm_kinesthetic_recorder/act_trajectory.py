"""Pure ACT trajectory shaping helpers used by the ROS executor.

The official Dora executor evaluates a 30 Hz chunk with a cubic Hermite spline,
then applies a persistent 250 Hz biquad low-pass filter.  This module mirrors
those operations without importing ROS so they can be tested on the workstation.
"""

from __future__ import annotations

import math


class HermiteUpsampler:
    def __init__(self, chunk_hz: float = 30.0, control_hz: float = 250.0):
        self.chunk_hz = float(chunk_hz)
        self.control_hz = float(control_hz)
        if not math.isfinite(self.chunk_hz) or self.chunk_hz <= 0.0:
            raise ValueError("chunk_hz must be finite and positive")
        if not math.isfinite(self.control_hz) or self.control_hz <= 0.0:
            raise ValueError("control_hz must be finite and positive")
        self.chunk_dt = 1.0 / self.chunk_hz
        self.control_dt = 1.0 / self.control_hz
        self.horizon = 9.0 / self.chunk_hz
        count = int(round(self.horizon * self.control_hz)) + 1
        self.eval_times = [index * self.control_dt for index in range(count)]
        self.chunk_times = [index * self.chunk_dt for index in range(10)]

    @staticmethod
    def _slopes(values: list[list[float]], times: list[float]) -> list[list[float]]:
        slopes = []
        for index in range(len(values) - 1):
            dt = times[index + 1] - times[index]
            slopes.append([(b - a) / dt for a, b in zip(values[index], values[index + 1])])
        result = [[0.0] * len(values[0]) for _ in values]
        result[0] = list(slopes[0])
        result[-1] = list(slopes[-1])
        for index in range(1, len(values) - 1):
            result[index] = [
                0.0 if a * b <= 0.0 else 0.5 * (a + b)
                for a, b in zip(slopes[index - 1], slopes[index])
            ]
        return result

    def upsample(self, values: list[list[float]]) -> list[list[float]]:
        if len(values) != 10 or any(len(row) != 8 for row in values):
            raise ValueError("ACT trajectory must be 10x8")
        positions = [[float(value) for value in row] for row in values]
        if any(not math.isfinite(value) for row in positions for value in row):
            raise ValueError("ACT trajectory must contain only finite values")
        slopes = self._slopes(positions, self.chunk_times)
        output = []
        for time_s in self.eval_times:
            index = min(len(positions) - 2, max(0, int(math.floor(time_s / self.chunk_dt))))
            t0 = self.chunk_times[index]
            h = self.chunk_dt
            u = (time_s - t0) / h
            h00 = 2.0 * u**3 - 3.0 * u**2 + 1.0
            h10 = (u**3 - 2.0 * u**2 + u) * h
            h01 = -2.0 * u**3 + 3.0 * u**2
            h11 = (u**3 - u**2) * h
            output.append([
                h00 * y0 + h10 * m0 + h01 * y1 + h11 * m1
                for y0, m0, y1, m1 in zip(
                    positions[index], slopes[index], positions[index + 1], slopes[index + 1]
                )
            ])
        return output


class BiquadLowpass:
    def __init__(self, fs: float = 250.0, fc: float = 15.0, q: float = 0.707):
        if not math.isfinite(float(fs)) or float(fs) <= 0.0:
            raise ValueError("filter sample rate must be finite and positive")
        if not math.isfinite(float(fc)) or float(fc) <= 0.0 or float(fc) >= float(fs) / 2.0:
            raise ValueError("filter cutoff must be finite, positive, and below Nyquist")
        if not math.isfinite(float(q)) or float(q) <= 0.0:
            raise ValueError("filter Q must be finite and positive")
        w0 = 2.0 * math.pi * float(fc) / float(fs)
        alpha = math.sin(w0) / (2.0 * float(q))
        a0 = 1.0 + alpha
        cosine = math.cos(w0)
        self.b0 = ((1.0 - cosine) / 2.0) / a0
        self.b1 = (1.0 - cosine) / a0
        self.b2 = self.b0
        self.a1 = (-2.0 * cosine) / a0
        self.a2 = (1.0 - alpha) / a0
        self.x1 = self.x2 = self.y1 = self.y2 = None

    def reset(self, value: list[float]) -> None:
        initial = [float(item) for item in value]
        self.x1 = list(initial)
        self.x2 = list(initial)
        self.y1 = list(initial)
        self.y2 = list(initial)

    def step(self, value: list[float]) -> list[float]:
        current = [float(item) for item in value]
        if self.x1 is None:
            self.reset(current)
        output = [
            self.b0 * x
            + self.b1 * x1
            + self.b2 * x2
            - self.a1 * y1
            - self.a2 * y2
            for x, x1, x2, y1, y2 in zip(current, self.x1, self.x2, self.y1, self.y2)
        ]
        self.x2, self.x1 = self.x1, current
        self.y2, self.y1 = self.y1, output
        return output


class OfficialActShaper:
    def __init__(self, control_hz: float = 250.0, cutoff_hz: float = 15.0):
        self.upsampler = HermiteUpsampler(control_hz=control_hz)
        self.lowpass = BiquadLowpass(fs=control_hz, fc=cutoff_hz)

    def reset(self) -> None:
        self.lowpass.x1 = self.lowpass.x2 = None
        self.lowpass.y1 = self.lowpass.y2 = None

    def shape(self, values: list[list[float]]) -> list[list[float]]:
        return [self.lowpass.step(value) for value in self.upsampler.upsample(values)]


def max_abs_difference(left: list[float], right: list[float]) -> float:
    """Return a finite, dimension-checked maximum absolute joint difference."""
    if len(left) != len(right) or not left:
        raise ValueError("joint vectors must have the same non-zero length")
    values = [abs(float(a) - float(b)) for a, b in zip(left, right)]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("joint vectors must contain only finite values")
    return max(values)
