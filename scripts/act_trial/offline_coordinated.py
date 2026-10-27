"""OFFLINE ONLY: bounded polynomial splice experiments, not a robot controller.

No ROS/CAN/transport imports. A collision callback must be supplied explicitly.
This searches a finite family, not all feasible paths. Failure is NOT proof of
infeasibility. Success is only sampled static geometry, not physical safety.
"""
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import BPoly, PPoly


def vector(x):
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError("expected finite vector")
    return x


def splice(q, v, a, target, duration):
    """Quintic with inherited p/v/a and terminal zero velocity/acceleration."""
    q, v, a, target = map(vector, (q, v, a, target))
    if not (q.shape == v.shape == a.shape == target.shape):
        raise ValueError("shape mismatch")
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("invalid duration")
    z = np.zeros_like(q)
    return BPoly.from_derivatives([0., duration], [[q, v, a], [target, z, z]])


def bounds(poly):
    """Conservative Bernstein convex-hull bounds, including inter-knot motion."""
    return {
        "lower": np.min(poly.c, axis=(0, 1)),
        "upper": np.max(poly.c, axis=(0, 1)),
        "speed": np.max(np.abs(poly.derivative().c), axis=(0, 1)),
        "acceleration": np.max(np.abs(poly.derivative(2).c), axis=(0, 1)),
    }


def extrema_bounds(poly):
    """Numerical extrema from derivative roots, offline cross-check only.

    Evaluates boundaries and all real interior critical points for each component
    on every segment. Not interval arithmetic or a certified floating-point proof.
    """
    low = np.full(poly.c.shape[-1], np.inf)
    high = -low.copy()
    magnitudes = [np.zeros_like(low), np.zeros_like(low)]
    for segment, duration in enumerate(np.diff(poly.x)):
        # Translate position and normalize time to reduce cancellation for tiny
        # segments around a large nonzero joint angle.
        base = poly.c[0, segment].copy()
        unit = BPoly((poly.c[:, segment:segment+1]-base), [0., 1.])
        power = PPoly.from_bernstein_basis(unit)
        for order in range(3):
            derivative = power.derivative(order)
            for joint in range(len(low)):
                c = derivative.c[:, 0, joint]
                roots = np.roots(np.polyder(c))
                ts = [0., 1.]
                ts += [float(z.real) for z in roots
                       if abs(z.imag) < 1e-9 and 0 < z.real < 1.]
                values = np.polyval(c, ts)/duration**order
                if order == 0:
                    values += base[joint]
                    low[joint] = min(low[joint], np.min(values))
                    high[joint] = max(high[joint], np.max(values))
                else:
                    magnitudes[order-1][joint] = max(
                        magnitudes[order-1][joint], np.max(np.abs(values)))
    return {"lower": low, "upper": high, "speed": magnitudes[0],
            "acceleration": magnitudes[1]}


def within_limits(poly, lower, upper, speed, acceleration, bound_method="bernstein"):
    if bound_method not in ("bernstein", "extrema"):
        raise ValueError("unknown bound method")
    b = bounds(poly) if bound_method == "bernstein" else extrema_bounds(poly)
    return (np.all(b["lower"] >= np.asarray(lower)-1e-10)
            and np.all(b["upper"] <= np.asarray(upper)+1e-10)
            and np.all(b["speed"] <= np.asarray(speed)+1e-10)
            and np.all(b["acceleration"] <= np.asarray(acceleration)+1e-10))


def sampled_collision(poly, collision, max_dt=.005, max_dq=.002):
    """Derivative bounds ensure samples have <=max_dq coordinate travel.

    This is NOT continuous collision detection or a positive-clearance proof.
    """
    b = bounds(poly)
    dt = min(max_dt, max_dq/max(float(np.max(b["speed"])), 1e-12))
    n = max(1, int(np.ceil((poly.x[-1]-poly.x[0])/dt)))
    if n > 200000:
        raise ValueError("offline sampling budget exceeded")
    for i, t in enumerate(np.linspace(poly.x[0], poly.x[-1], n+1)):
        contacts = collision(poly(t))
        if contacts:
            return {"collision": True, "time_s": float(t), "contacts": contacts,
                    "samples": i+1}
    return {"collision": False, "samples": n+1}


@dataclass
class SearchResult:
    poly: object = None
    fraction: float = 0.
    duration: float = 0.
    kind: str = "none"
    tested: int = 0
    bound_passed: int = 0
    geometry_rejected: int = 0


def search_splice(q, v, a, target, *, lower, upper, speed, acceleration,
                  remaining_age, collision, fractions=(1., .5, .25, .1, .05, .02, .01),
                  bound_method="bernstein"):
    """Finite search of short target prefixes followed by rest, plus braking.

    All candidates end before the ORIGINAL observation expires. Braking endpoint
    integrates a cubic velocity transition to rest; it is still collision checked.
    No zeroing of inherited velocity or acceleration on a replan.
    """
    q, v, a, target = map(vector, (q, v, a, target))
    result = SearchResult()
    if not np.isfinite(remaining_age) or remaining_age < .02:
        return result
    durations = sorted(set([float(remaining_age)] +
                           [t for t in (.05, .10, .15, .20, .25, .30, .35, .40, .45)
                            if t <= remaining_age]))
    for fraction in (*fractions, 0.):
        for duration in durations:
            endpoint = (q+fraction*(target-q) if fraction else
                        q+v*duration/2+a*duration**2/12)
            p = splice(q, v, a, endpoint, duration)
            result.tested += 1
            if not within_limits(p, lower, upper, speed, acceleration, bound_method):
                continue
            result.bound_passed += 1
            if sampled_collision(p, collision)["collision"]:
                result.geometry_rejected += 1
                continue
            result.poly, result.fraction, result.duration = p, fraction, duration
            result.kind = "target_prefix" if fraction else "brake"
            return result
    return result


def fit_and_slow(times, positions, speed, acceleration):
    """Offline reference: C2 quintic interpolation, then uniform time dilation.

    NOT exact preservation of recorded piecewise-linear segments: the fitted
    curve passes all knots and needs its own collision check. Retiming preserves
    that fitted curve exactly; no statistical smoothing/denoising is performed.
    """
    t, q = np.asarray(times, float), np.asarray(positions, float)
    if (t.ndim != 1 or len(t) < 3 or q.ndim != 2 or len(q) != len(t)
            or not np.isfinite(t).all() or not np.isfinite(q).all()
            or np.any(np.diff(t) <= 0)):
        raise ValueError("invalid reference data")
    speed, acceleration = vector(speed), vector(acceleration)
    if (speed.shape != q.shape[1:] or acceleration.shape != speed.shape
            or np.any(speed <= 0) or np.any(acceleration <= 0)):
        raise ValueError("invalid reference limits")
    t = t-t[0]
    v = np.gradient(q, t, axis=0)
    a = np.gradient(v, t, axis=0)
    v[[0, -1]] = 0
    a[[0, -1]] = 0
    original = BPoly.from_derivatives(t, [[p, vel, acc] for p, vel, acc in zip(q, v, a)])
    b = bounds(original)
    scale = max(1., 1.02*float(np.max(b["speed"]/speed)),
                1.02*float(np.sqrt(np.max(b["acceleration"]/acceleration))))
    slowed = BPoly(original.c.copy(), original.x*scale)
    return original, slowed, scale
