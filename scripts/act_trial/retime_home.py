"""Uniform slowdown of JTC splines; positions and collision path unchanged."""
from copy import deepcopy
import numpy as np
from scipy.interpolate import BPoly


def slow_to_limits(trajectory, velocity_limits, acceleration_limits):
    vmax = np.minimum(np.asarray(velocity_limits, dtype=float)[:7]*.10, .25)
    amax = np.minimum(np.asarray(acceleration_limits, dtype=float), .05)
    if any(x.shape != (7,) or not np.isfinite(x).all() or (x <= 0).any()
           for x in (vmax, amax)):
        raise ValueError('invalid recovery retiming limits')
    result = deepcopy(trajectory)
    points = result.joint_trajectory.points
    if len(points) < 2:
        raise ValueError('retiming requires two or more points')
    times = [p.time_from_start.sec+p.time_from_start.nanosec/1e9 for p in points]
    if times[0] != 0 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError('invalid recovery timestamps')
    derivatives = []
    for p in points:
        if p.accelerations and not p.velocities:
            raise ValueError('acceleration without velocity')
        row = [p.positions]
        if p.velocities:
            row.append(p.velocities)
        if p.accelerations:
            row.append(p.accelerations)
        if any(len(x) != 7 or not np.isfinite(x).all() for x in row):
            raise ValueError('invalid recovery derivatives')
        derivatives.append(row)
    if len({len(row) for row in derivatives}) != 1:
        raise ValueError('inconsistent recovery derivatives')
    if len(derivatives[0]) < 2:
        raise ValueError('retiming requires velocity knots to avoid unbounded corner acceleration')
    vb, ab = np.zeros(7), np.zeros(7)
    for i in range(len(points)-1):
        poly = BPoly.from_derivatives(times[i:i+2], derivatives[i:i+2])
        # Bernstein convex-hull bound includes between-knot extrema, unlike
        # checking only waypoint derivatives. May conservatively overestimate.
        vb = np.maximum(vb, np.max(np.abs(poly.derivative().c), axis=(0, 1)))
        ab = np.maximum(ab, np.max(np.abs(poly.derivative(2).c), axis=(0, 1)))
    ratio = max(float(np.max(vb/vmax)), float(np.sqrt(np.max(ab/amax))))
    scale = max(1., ratio*1.02)
    if not np.isfinite(scale) or times[-1]*scale > 600 or np.max(np.diff(times))*scale > 60:
        raise ValueError('retimed HOME exceeds validation duration budget')
    for p, t in zip(points, times):
        ns = round(t*scale*1e9)
        p.time_from_start.sec, p.time_from_start.nanosec = divmod(ns, 10**9)
        p.velocities = (np.asarray(p.velocities)/scale).tolist()
        if p.accelerations:
            p.accelerations = (np.asarray(p.accelerations)/scale**2).tolist()
    return result, dict(time_scale=scale, original_duration_s=times[-1],
                        duration_s=times[-1]*scale, velocity_bound_before=vb.tolist(),
                        acceleration_bound_before=ab.tolist(), velocity_limit=vmax.tolist(),
                        acceleration_limit=amax.tolist(),
                        method='uniform time dilation; Bernstein derivative bounds; full revalidation required')
