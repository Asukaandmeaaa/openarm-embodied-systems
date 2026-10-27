"""Offline tests; no motor/CAN/ROS access."""
import numpy as np
import pytest

from scripts.act_trial.speed_contract import measured_speed_contract, check_measured_speed


def source(tmp_path, value):
    path = tmp_path/'real_kinesthetic.yaml'
    path.write_text('right_kinesthetic_bridge:\n  ros__parameters:\n'
                    f'    cartesian_max_joint_velocity_rad_s: {value}\n')
    return path


@pytest.fixture(autouse=True)
def planning_source(monkeypatch, tmp_path):
    from scripts.act_trial import speed_contract
    path = tmp_path/'joint_limits.yaml'
    path.write_text('default_velocity_scaling_factor: 0.1\n')
    monkeypatch.setattr(speed_contract, 'ROS_PLANNING_CONFIG', path)
    return path


def test_recorded_transient_passes_but_overspeed_still_trips(tmp_path):
    contract = measured_speed_contract([5.]*7, source(tmp_path, .25))
    assert check_measured_speed([-.2307692307692335]+[0.]*6, contract)['passed']
    assert check_measured_speed([.50]*7, contract)['passed']
    report = check_measured_speed([0.]*6+[-.50001], contract)
    assert not report['passed'] and report['exceeded_joints'] == [7]
    assert len(contract['source_sha256']) == 64


def test_source_and_canonical_limits_can_tighten_not_silently_raise(tmp_path):
    cap = measured_speed_contract([.12]+[5.]*6, source(tmp_path, 10.))
    assert cap['limits_rad_s'] == [.012]+[.50]*6
    assert cap['command_limits_rad_s'] == [.012]+[.25]*6
    cap = measured_speed_contract([5.]*7, source(tmp_path, .08))
    assert cap['limits_rad_s'] == [.50]*7
    assert cap['command_limits_rad_s'] == [.08]*7


def test_planning_scale_tightens_both_bounds_but_never_auto_widens(tmp_path, planning_source):
    teaching = source(tmp_path, .25)
    planning_source.write_text('default_velocity_scaling_factor: 1.0\n')
    cap = measured_speed_contract([5.]*7, teaching)
    assert cap['limits_rad_s'] == [.50]*7
    planning_source.write_text('default_velocity_scaling_factor: 0.01\n')
    cap = measured_speed_contract([5.]*7, teaching)
    assert cap['limits_rad_s'] == [.05]*7
    assert cap['command_limits_rad_s'] == [.05]*7


@pytest.mark.parametrize('scale', ['.nan', '0', '-1', '1.1'])
def test_invalid_planning_scale_refused(tmp_path, planning_source, scale):
    planning_source.write_text('default_velocity_scaling_factor: '+scale+'\n')
    with pytest.raises(ValueError): measured_speed_contract([5.]*7, source(tmp_path, .25))


@pytest.mark.parametrize('bad', [[0.]*6, [float('nan')]*7, [float('inf')]*7])
def test_invalid_feedback_refused(tmp_path, bad):
    cap = measured_speed_contract([5.]*7, source(tmp_path, .25))
    with pytest.raises(ValueError): check_measured_speed(bad, cap)


@pytest.mark.parametrize('bad', ['.nan', '0', '-1'])
def test_invalid_config_refused(tmp_path, bad):
    with pytest.raises(ValueError): measured_speed_contract([5.]*7, source(tmp_path, bad))


def test_nonfinite_or_wrong_canonical_limits_refused(tmp_path):
    for value in ([5.]*6, [np.nan]*7, [0.]*7):
        with pytest.raises(ValueError): measured_speed_contract(value, source(tmp_path, .25))
