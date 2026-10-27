# Copyright 2026 Enactic, Inc. (Apache-2.0)
"""Exact numerical algorithms from the official dora action executor.

Source and SHA256 are pinned in configs/replay.yaml. Runtime never imports the
read-only source workspace.
"""

from __future__ import annotations

import numpy as np


class HermiteUpsampler:
    def __init__(self, chunk_hz, horizon_sec):
        self.chunk_hz = float(chunk_hz); self.horizon_sec = float(horizon_sec)
        self.dt_chunk = 1.0 / self.chunk_hz
        self.t_chunk = np.arange(0.0, self.horizon_sec + 1e-12, self.dt_chunk)

    @staticmethod
    def _compute_slopes(t, y):
        s = np.diff(y, axis=0) / np.diff(t)[:, None]; m = np.zeros_like(y)
        m[0] = s[0]; m[-1] = s[-1]
        for i in range(1, len(y) - 1):
            prod = s[i - 1] * s[i]; mask = prod <= 0.0
            mi = 0.5 * (s[i - 1] + s[i]); mi[mask] = 0.0; m[i] = mi
        return m

    def upsample(self, y_chunk, t_eval):
        y = np.asarray(y_chunk, dtype=np.float32)
        if y.shape[0] != len(self.t_chunk): raise ValueError(f"Expected chunk length {len(self.t_chunk)}, got {y.shape[0]}")
        slopes = self._compute_slopes(self.t_chunk, y)
        idx = np.searchsorted(self.t_chunk, t_eval, side="right") - 1; idx = np.clip(idx, 0, len(self.t_chunk) - 2)
        t0 = self.t_chunk[idx]; t1 = self.t_chunk[idx + 1]; h = t1 - t0; u = (t_eval - t0) / h
        h00 = 2*u**3 - 3*u**2 + 1; h10 = (u**3 - 2*u**2 + u)*h
        h01 = -2*u**3 + 3*u**2; h11 = (u**3 - u**2)*h
        y0 = y[idx]; y1 = y[idx+1]; m0 = slopes[idx]; m1 = slopes[idx+1]
        return (h00[:,None]*y0 + h10[:,None]*m0 + h01[:,None]*y1 + h11[:,None]*m1).astype(np.float32)


class BiquadLowpass:
    def __init__(self, fs, fc, Q=0.707):
        fs=float(fs);fc=float(fc);Q=float(Q);w0=2*np.pi*fc/fs;cosw0=np.cos(w0);alpha=np.sin(w0)/(2*Q);a0=1+alpha
        self.b0=((1-cosw0)/2)/a0;self.b1=(1-cosw0)/a0;self.b2=((1-cosw0)/2)/a0
        self.a1=(-2*cosw0)/a0;self.a2=(1-alpha)/a0;self.x1=self.x2=self.y1=self.y2=None

    def reset_state(self, initial_x):
        initial_x=np.asarray(initial_x,dtype=np.float32);self.x1=initial_x.copy();self.x2=initial_x.copy();self.y1=initial_x.copy();self.y2=initial_x.copy()

    def step(self,x):
        x=np.asarray(x,dtype=np.float32)
        if self.x1 is None:self.reset_state(x)
        y=self.b0*x+self.b1*self.x1+self.b2*self.x2-self.a1*self.y1-self.a2*self.y2
        self.x2,self.x1=self.x1,x;self.y2,self.y1=self.y1,y
        return y.astype(np.float32)
