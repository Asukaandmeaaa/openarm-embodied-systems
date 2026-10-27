from types import SimpleNamespace as NS
import numpy as np
import pytest
from scripts.act_trial.keyboard import dispatch
from scripts.act_trial.retime_home import slow_to_limits


def test_keyboard_ready_failure_never_starts():
    calls=[]
    dispatch('s', run=lambda s: calls.append(s) or 1)
    assert calls == ['ready']


def test_home_requires_separate_confirmation():
    calls=[]
    run=lambda s: calls.append(s) or 0
    dispatch('h',run=run)
    dispatch('r',run=run,confirm=lambda _: 'no')
    assert calls == ['plan-home']
    dispatch('r',run=run,confirm=lambda _: 'HOME')
    assert calls == ['plan-home','return-home']


def test_pause_never_resumes_or_opens_gripper():
    calls=[]
    dispatch('p',run=lambda s:calls.append(s) or 0)
    assert calls == ['stop','status']


def test_uniform_slowdown_preserves_geometry_and_derivatives():
    from scipy.interpolate import BPoly
    def p(q,t):
        return NS(positions=[q]*7,velocities=[0.]*7,accelerations=[0.]*7,
                  time_from_start=NS(sec=t,nanosec=0))
    trajectory=NS(joint_trajectory=NS(points=[p(0,0),p(.2,1)]))
    changed,audit=slow_to_limits(trajectory,np.ones(16)*5,np.ones(7)*.05)
    scale=audit['time_scale']
    assert scale>1 and trajectory.joint_trajectory.points[-1].time_from_start.sec==1
    pt=changed.joint_trajectory.points[-1]
    end=pt.time_from_start.sec+pt.time_from_start.nanosec/1e9
    poly=BPoly.from_derivatives([0,end],[[[0]*7,[0]*7,[0]*7],[[.2]*7,[0]*7,[0]*7]])
    assert np.max(abs(poly.derivative(2)(np.linspace(0,end,1001))))<.05
    original=BPoly.from_derivatives([0,1],[[[0]*7,[0]*7,[0]*7],[[.2]*7,[0]*7,[0]*7]])
    np.testing.assert_allclose(poly(np.linspace(0,end,101)),original(np.linspace(0,1,101)),atol=1e-10)


def test_invalid_times_refused():
    p=NS(positions=[0]*7,velocities=[0]*7,accelerations=[],time_from_start=NS(sec=0,nanosec=0))
    with pytest.raises(ValueError,match='timestamps'):
        slow_to_limits(NS(joint_trajectory=NS(points=[p,p])),np.ones(7),np.ones(7))
