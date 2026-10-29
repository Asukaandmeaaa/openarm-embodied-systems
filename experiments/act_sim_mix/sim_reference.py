"""Frozen offline reference math. Apache-2.0 upstream classes copied verbatim.
No Dora node, CAN socket, camera device or deployment runtime imports.
See config/reference_provenance.json.
"""
import numpy as np


class HermiteUpsampler:
    """Upsamples coarse trajectory chunks using cubic Hermite spline interpolation."""

    def __init__(self, chunk_hz, horizon_sec):
        """Initialize the upsampler with the chunk frequency and horizon."""
        self.chunk_hz = float(chunk_hz)
        self.horizon_sec = float(horizon_sec)
        self.dt_chunk = 1.0 / self.chunk_hz
        self.t_chunk = np.arange(0.0, self.horizon_sec + 1e-12, self.dt_chunk)

    @staticmethod
    def _compute_slopes(t, y):
        s = np.diff(y, axis=0) / np.diff(t)[:, None]
        m = np.zeros_like(y)
        m[0] = s[0]
        m[-1] = s[-1]
        for i in range(1, len(y) - 1):
            prod = s[i - 1] * s[i]
            mask = prod <= 0.0
            mi = 0.5 * (s[i - 1] + s[i])
            mi[mask] = 0.0
            m[i] = mi
        return m

    def upsample(self, y_chunk, t_eval):
        """Upsample the given chunk of trajectory to the specified evaluation times."""
        y = np.asarray(y_chunk, dtype=np.float32)
        if y.shape[0] != len(self.t_chunk):
            raise ValueError(
                f"Expected chunk length {len(self.t_chunk)}, got {y.shape[0]}"
            )

        slopes = self._compute_slopes(self.t_chunk, y)
        idx = np.searchsorted(self.t_chunk, t_eval, side="right") - 1
        idx = np.clip(idx, 0, len(self.t_chunk) - 2)

        t0 = self.t_chunk[idx]
        t1 = self.t_chunk[idx + 1]
        h = t1 - t0
        u = (t_eval - t0) / h

        h00 = 2 * u**3 - 3 * u**2 + 1
        h10 = (u**3 - 2 * u**2 + u) * h
        h01 = -2 * u**3 + 3 * u**2
        h11 = (u**3 - u**2) * h

        y0 = y[idx]
        y1 = y[idx + 1]
        m0 = slopes[idx]
        m1 = slopes[idx + 1]

        return (
            h00[:, None] * y0
            + h10[:, None] * m0
            + h01[:, None] * y1
            + h11[:, None] * m1
        ).astype(np.float32)


class BiquadLowpass:
    """Biquad low-pass filter for smoothing outputs."""

    def __init__(self, fs, fc, Q=0.707):
        """Initialize the biquad low-pass filter with sampling frequency, cutoff frequency, and Q factor."""
        fs = float(fs)
        fc = float(fc)
        Q = float(Q)
        w0 = 2 * np.pi * fc / fs
        cosw0 = np.cos(w0)
        alpha = np.sin(w0) / (2 * Q)
        a0 = 1 + alpha
        self.b0 = ((1 - cosw0) / 2) / a0
        self.b1 = (1 - cosw0) / a0
        self.b2 = ((1 - cosw0) / 2) / a0
        self.a1 = (-2 * cosw0) / a0
        self.a2 = (1 - alpha) / a0
        self.x1 = None
        self.x2 = None
        self.y1 = None
        self.y2 = None

    def reset_state(self, initial_x):
        """Reset the filter state with the initial input value."""
        initial_x = np.asarray(initial_x, dtype=np.float32)
        self.x1 = initial_x.copy()
        self.x2 = initial_x.copy()
        self.y1 = initial_x.copy()
        self.y2 = initial_x.copy()

    def step(self, x):
        """Apply one step of the biquad low-pass filter to the input x."""
        x = np.asarray(x, dtype=np.float32)
        if self.x1 is None:
            self.reset_state(x)
        y = (
            self.b0 * x
            + self.b1 * self.x1
            + self.b2 * self.x2
            - self.a1 * self.y1
            - self.a2 * self.y2
        )
        self.x2, self.x1 = self.x1, x
        self.y2, self.y1 = self.y1, y
        return y.astype(np.float32)


def prepare_chunk(event, steps, chunk_size=50):
    q = np.asarray(event['value'].to_pylist(), dtype=np.float32)
    if q.shape != (chunk_size, 16) or not np.isfinite(q).all():
        raise ValueError(f'expected finite {chunk_size}x16 ACT chunk')
    interval = event['metadata']['interval']
    if type(interval) is not int or interval != 33_333_333:
        raise ValueError('chunk experiment requires training interval 33333333ns')
    if type(steps) is not int or not 2 <= steps <= chunk_size:
        raise ValueError('invalid committed action count')
    dt = interval/1e9
    # N coarse targets occupy N 30Hz slots. Hold the final target in the
    # last slot; do not interpolate toward any discarded tail action.
    sample_ns = np.arange(0, steps*interval, 4_000_000, dtype=np.int64)
    times = np.minimum(sample_ns/1e9, (steps-1)*dt)
    upsampler = HermiteUpsampler(1/dt, (steps-1)*dt)
    values = upsampler.upsample(q[:steps], times)
    return sample_ns, values, steps*interval
