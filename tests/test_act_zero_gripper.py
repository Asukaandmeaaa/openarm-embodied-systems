"""Gripper setup tests using fake authority, clock and CAN feedback only."""
import json

import numpy as np
import pytest

from scripts.act_trial import zero_gripper as opening
from scripts.act_trial import manual_experiment as manual


def test_opening_path_keeps_arm_fixed_and_bounds_velocity_acceleration():
    initial = np.r_[np.arange(7)*.001, 0.]
    times = np.arange(0., 10.021, .02)
    path = np.array([opening.opening_reference(initial, t) for t in times])
    np.testing.assert_array_equal(path[:, :7], np.tile(initial[:7], (len(times), 1)))
    assert path[0, 7] == 0.
    assert path[-1, 7] == pytest.approx(-.4)
    assert np.all(np.diff(path[:, 7]) <= 1e-10)
    assert np.max(np.abs(np.diff(path[:, 7])/.02)) < .10
    assert np.max(np.abs(np.diff(path[:, 7], n=2)/.02**2)) < .03


def test_failed_receive_only_zero_acceptance_is_archived_for_recheck(monkeypatch, tmp_path):
    (tmp_path/'zero_actual_pose.json').write_text(json.dumps({
        'passed': False, 'receive_only': True}))
    (tmp_path/'zero_actual_pose.can.jsonl').write_text('original CAN evidence\n')

    class FakeClient:
        def __enter__(self): return self
        def __exit__(self, *_): pass
        def request(self, _):
            return {'status': {'state': 'idle'}, 'synchronized': True,
                    'supports_dora_arm_velocity': True,
                    'controller_command_interfaces': {'right_joint_trajectory_controller': ['position', 'velocity']},
                    'controllers': {c: 'active' for c in (
                        'right_joint_trajectory_controller', 'left_joint_trajectory_controller',
                        'right_gripper_controller')}}

    phases = []
    monkeypatch.setattr(manual, 'Client', FakeClient)
    monkeypatch.setattr(manual, 'verify_pose', lambda phase, *_: phases.append(phase))
    monkeypatch.setattr(manual, 'wait_idle', lambda *a, **k: None)
    monkeypatch.setattr(manual.subprocess, 'run', lambda *a, **k: None)
    manual.zero_handoff(tmp_path)
    assert phases == ['visual32_zero', 'visual32_zero']
    archive, = tmp_path.glob('zero_rejected_*')
    assert (archive/'zero_actual_pose.can.jsonl').read_text() == 'original CAN evidence\n'


def test_opening_uses_lease_scope_and_never_changes_arm_targets(monkeypatch, tmp_path):
    from scripts.act_trial import telemetry
    now = [100.]
    reference = np.zeros(8)
    events = []
    monkeypatch.setattr(opening.time, 'monotonic', lambda: now[0])
    monkeypatch.setattr(opening.time, 'sleep', lambda seconds: now.__setitem__(0, now[0]+seconds))
    monkeypatch.setattr(manual, 'wait_idle', lambda *a, **k: None)

    class FakeTelemetry:
        def __init__(self, _): pass
        def snapshot(self):
            q = np.zeros(16); q[:8] = reference
            return {'positions_rad': q.tolist(), 'command_positions_rad': q.tolist(),
                    'velocity_rad_s': [0.]*16, 'motor_status': [1]*16,
                    'max_age_s': .01, 'command_max_age_s': .01}
        def close(self): events.append('observer_closed')

    class FakeClient:
        token = None
        def __enter__(self): return self
        def __exit__(self, *_): pass
        def request(self, op, **kwargs):
            assert op == 'status', 'successful setup must not freeze/reset'
            return {'status': {'state': 'idle'}, 'synchronized': True,
                    'supported_dora_scopes': ['right_arm_gripper']}
        def acquire(self, owner, scope):
            assert owner == 'dora' and scope == 'right_arm_gripper'
            self.token = 'fake'
            events.append('acquire')
            return {'right_reference': [0.]*7}
        def renew(self): assert self.token
        def command(self, q):
            assert self.token and q[:7] == [0.]*7
            reference[:] = q
        def release(self):
            self.token = None
            events.append('release')

    monkeypatch.setattr(opening, 'Client', FakeClient)
    monkeypatch.setattr(telemetry, 'Telemetry', FakeTelemetry)
    opening.open_at_zero(tmp_path, tmp_path)
    report = json.loads((tmp_path/'opening.json').read_text())
    assert report['passed'] and report['motion_requested']
    assert events == ['acquire', 'release', 'observer_closed']
    assert reference[7] == pytest.approx(-.4)
