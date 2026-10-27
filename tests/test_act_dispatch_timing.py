"""Pure deadline checks; never initialize ROS or connect to hardware."""
import ast
from pathlib import Path

import pytest
from scripts.act_trial.ros_bridge import check_dispatch_freshness
from scripts.act_trial.ros_bridge import start_request_ready
from scripts.act_trial.ros_bridge import action_ready_for_start


def test_rolling_ready_survives_normal_inference_gap():
    # Observed chunk arrivals ~132 ms apart; source age remains fresh.
    for age_ms in range(133):
        assert action_ready_for_start(10.+age_ms/1000, 10.,
                                      .04+age_ms/1000, rolling=True)
    assert not action_ready_for_start(10.132, 10., .172, rolling=False)


@pytest.mark.parametrize('received,age', [
    (None, .1), (10., None), (10., float('nan')), (float('nan'), .1),
    (10., float('inf')), (10.2, .1), (10., -.01),
    (10.13, .48), (10.13, .51), (9.5, .1),
])
def test_rolling_start_rejects_missing_future_or_expired_evidence(received, age):
    assert not action_ready_for_start(10.132, received, age, rolling=True)


def test_legacy_start_keeps_receipt_and_observation_deadlines():
    assert action_ready_for_start(10.05, 10., .4)
    assert not action_ready_for_start(10.05, 10., .51)
    assert not action_ready_for_start(10.11, 10., .2)


def test_fresh_start_waits_for_next_eligible_window_without_weakening_admission():
    request = {'created_monotonic_ns': 1_000_000_000, 'profile': 'task', 'duration_s': None}
    assert not start_request_ready(request, 1_010_000_000, False, 'task', None)
    assert start_request_ready(request, 1_150_000_000, True, 'task', None)
    with pytest.raises(ValueError, match='freshness'):
        start_request_ready(request, 3_000_000_001, True, 'task', None)
    with pytest.raises(ValueError, match='profile'):
        start_request_ready(request, 1_150_000_000, True, 'commissioning', 3.)


def test_fresh_proposal():
    check_dispatch_freshness(10., 10.005, .01, 9_900_000_000, 10.02)


@pytest.mark.parametrize('now,age,obs,match', [
    (10.106, .01, 9_900_000_000, 'deadline'),
    (10.09, .02, 9_900_000_000, 'feedback stale'),
    (10.02, .01, 9_400_000_000, 'observation stale'),
    (10.02, .01, None, 'timestamp missing'),
    (9.99, .01, 9_900_000_000, 'deadline'),
    (float('nan'), .01, 9_900_000_000, 'deadline'),
])
def test_no_dispatch_with_expired_original_evidence(now, age, obs, match):
    with pytest.raises(ValueError, match=match):
        check_dispatch_freshness(10., 10.005, age, obs, now)


def test_check_immediately_precedes_authority_command():
    tree = ast.parse(Path('scripts/act_trial/ros_bridge.py').read_text())
    found = False
    for node in ast.walk(tree):
        body = getattr(node, 'body', None)
        if not isinstance(body, list):
            continue
        for i, statement in enumerate(body):
            if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
                if ast.unparse(statement.value.func) == 'authority.command':
                    assert ast.unparse(body[i-1].value.func) == 'check_dispatch_freshness'
                    found = True
    assert found
