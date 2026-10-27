import numpy as np
import pytest

from scripts.act_trial.offline_coordinated import (
    bounds, extrema_bounds, fit_and_slow, sampled_collision, search_splice, splice, within_limits,
)


def test_splice_preserves_moving_start_and_terminal_rest():
    p = splice([0., .1], [.04, -.02], [.03, -.01], [.02, .09], .4)
    np.testing.assert_allclose(p(0), [0, .1], atol=1e-12)
    np.testing.assert_allclose(p(0, 1), [.04, -.02], atol=1e-12)
    np.testing.assert_allclose(p(0, 2), [.03, -.01], atol=1e-12)
    np.testing.assert_allclose(p(.4), [.02, .09], atol=1e-12)
    np.testing.assert_allclose(p(.4, 1), 0, atol=1e-12)
    np.testing.assert_allclose(p(.4, 2), 0, atol=1e-12)


def test_bounds_cover_interior_and_reject_overspeed():
    p = splice([0.], [0.], [0.], [1.], 1.)
    b = bounds(p)
    t = np.linspace(0, 1, 1001)
    assert np.all(np.abs(p(t, 1)) <= b["speed"]+1e-12)
    assert np.all(np.abs(p(t, 2)) <= b["acceleration"]+1e-12)
    assert not within_limits(p, [-2], [2], [.25], [.25])


def test_extrema_bounds_against_known_minimum_jerk_polynomial():
    p = splice([0.], [0.], [0.], [1.], 1.)
    b = extrema_bounds(p)
    np.testing.assert_allclose(b["lower"], [0.], atol=1e-12)
    np.testing.assert_allclose(b["upper"], [1.], atol=1e-12)
    np.testing.assert_allclose(b["speed"], [1.875], atol=1e-12)
    np.testing.assert_allclose(b["acceleration"], [10/np.sqrt(3)], atol=1e-12)


def test_extrema_constant_and_multisegment():
    from scipy.interpolate import BPoly
    p = BPoly.from_derivatives([0., .1, .3], [
        [[1., 0.], [0., .01], [0., 0.]],
        [[1., .001], [0., .01], [0., 0.]],
        [[1., .003], [0., .01], [0., 0.]],
    ])
    b = extrema_bounds(p)
    np.testing.assert_allclose(b["speed"], [0., .01], atol=1e-12)
    np.testing.assert_allclose(b["acceleration"], [0., 0.], atol=1e-12)


def test_random_splice_extrema_enclose_dense_samples():
    rng = np.random.default_rng(42)
    for _ in range(20):
        duration = rng.uniform(.05, .5)
        p = splice(rng.uniform(-1, 1, 3), rng.uniform(-.2, .2, 3),
                   rng.uniform(-.25, .25, 3), rng.uniform(-1, 1, 3), duration)
        b = extrema_bounds(p)
        times = np.linspace(0, duration, 1001)
        assert np.all(p(times) >= b["lower"]-1e-9)
        assert np.all(p(times) <= b["upper"]+1e-9)
        assert np.all(np.abs(p(times, 1)) <= b["speed"]+1e-9)
        assert np.all(np.abs(p(times, 2)) <= b["acceleration"]+1e-8)


def test_replan_is_continuous_not_velocity_reset():
    first = splice([1.], [.03], [.01], [1.01], .4)
    t = .1
    next_poly = splice(first(t), first(t, 1), first(t, 2), [1.02], .3)
    for order in range(3):
        np.testing.assert_allclose(next_poly(0, order), first(t, order), atol=1e-12)
    assert abs(next_poly(0, 1)[0]) > .001


def test_collision_checked_between_endpoints():
    p = splice([0.], [0.], [0.], [.01], 1.)
    def contact(q):
        return ["obstacle"] if .004 <= q[0] <= .006 else []
    assert not contact(p(0)) and not contact(p(1))
    assert sampled_collision(p, contact)["collision"]


def test_retime_preserves_fitted_path_and_derivative_limits():
    t = np.linspace(0, 2, 31)
    q = np.c_[np.sin(t), np.cos(t)]
    original, slowed, scale = fit_and_slow(t, q, [.25]*2, [.25]*2)
    assert scale > 1
    probes = np.linspace(0, 2, 351)
    np.testing.assert_allclose(original(probes), slowed(probes*scale), atol=1e-12)
    np.testing.assert_allclose(original(t), q, atol=1e-12)
    assert np.all(bounds(slowed)["speed"] <= .25)
    assert np.all(bounds(slowed)["acceleration"] <= .25)


def test_search_never_exceeds_observation_deadline_or_skips_collision():
    args = dict(lower=[-1], upper=[1], speed=[.25], acceleration=[.25],
                remaining_age=.31)
    r = search_splice([0], [0], [0], [.1], collision=lambda q: [], **args)
    assert r.poly is not None and r.duration <= .31
    assert r.fraction < 1  # Cannot go .1 rad rest-to-rest within .31 s.
    blocked = search_splice([0], [.01], [0], [.1],
                            collision=lambda q: ["blocked"], **args)
    assert blocked.poly is None
    assert blocked.geometry_rejected > 0
    stale = search_splice([0], [0], [0], [.1], collision=lambda q: [],
                         **{**args, "remaining_age": -.01})
    assert stale.poly is None


def test_bad_inputs_rejected():
    with pytest.raises(ValueError):
        splice([0], [0], [0], [float("nan")], 1)
    with pytest.raises(ValueError):
        fit_and_slow([0, 0, 1], [[0], [1], [2]], [.25], [.25])
