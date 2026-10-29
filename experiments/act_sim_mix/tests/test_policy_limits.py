"""Teacher limits and hardware endpoints have different meanings; never hide clipping."""
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.act_sim_mix.policy_limits import DeploymentEnvelope, DeploymentPolicyWorld, range_diagnostics
from experiments.act_sim_mix.sim_expert.world import SimWorld, load_config
from experiments.act_sim_mix.closed_loop import execute_block
from experiments.act_sim_mix.sim_reference import BiquadLowpass


def command():
    q = np.zeros(16)
    q[[7, 15]] = -.02
    return q


def test_small_teacher_range_excess_is_not_a_hardware_violation():
    profile = DeploymentEnvelope()
    q = command()
    q[7] = -.45340436697
    sent, axes = profile.apply(q)
    np.testing.assert_array_equal(sent, q)
    assert len(axes) == 0
    world = object.__new__(SimWorld)
    world.limits = np.asarray(load_config()['position_limits_rad'])
    world.cmd_limits = np.array([-.45, -.02])
    with pytest.raises(RuntimeError, match='gripper command outside'):
        world.validate_command(q)
    policy_world = object.__new__(DeploymentPolicyWorld)
    policy_world.deployment_envelope = profile
    np.testing.assert_array_equal(policy_world.validate_command(sent), q)


def test_hardware_saturation_matches_numpy_clip_and_does_not_change_input():
    profile = DeploymentEnvelope()
    q = command()
    q[:8] = [9, -9, 9, -9, 9, -9, 9, -9]
    previous = q.copy()
    sent, axes = profile.apply(q)
    np.testing.assert_array_equal(sent[:8], np.clip(q[:8], profile.right[:, 0], profile.right[:, 1]))
    np.testing.assert_array_equal(q, previous)
    np.testing.assert_array_equal(sent[8:], q[8:])
    assert axes.tolist() == list(range(8))


def test_nonfinite_targets_are_rejected_instead_of_clipped():
    q = command()
    q[7] = np.nan
    with pytest.raises(ValueError):
        DeploymentEnvelope().apply(q)


def test_raw_range_diagnostics_separate_teacher_and_hardware_and_discard_tail():
    pred = np.tile(command(), (100, 1))
    pred[:50, 7] = -.4534
    pred[50:] = 999
    r = range_diagnostics(pred, 50, load_config(), DeploymentEnvelope())
    assert r['expert_outside_counts_by_axis'][7] == 50
    assert r['hardware_outside_counts_by_axis'] == [0] * 8


class FakeWorld:
    def __init__(self, fail=False):
        self.data = SimpleNamespace(time=0.)
        self.commands = []
        self.fail = fail
    def state16(self): return command()
    def validate_command(self, q): self.commands.append(q.copy())
    def step(self, q, phase):
        if self.fail: raise RuntimeError('forbidden_contact:table')
        self.data.time += .001


def test_executor_logs_downstream_saturation_without_resetting_or_clipping_filter():
    profile = DeploymentEnvelope()
    pred = np.tile(command(), (100, 1)).astype(np.float32)
    pred[:50, 7] = -.9
    pred[50:, 7] = 999
    world = FakeWorld()
    lowpass = BiquadLowpass(250, 15)
    r = execute_block(world, pred, 50, lowpass, reset=True, left_hold=command()[8:], envelope=profile)
    assert r['clipped_actions'] == r['reference_count'] == 417
    assert r['clipped_counts_by_right_axis'][7] == 417
    assert r['first_filtered_right'][7] == pytest.approx(-.9)
    assert r['first_commanded_right'][7] == pytest.approx(-.7854)
    assert r['last_commanded_right'][7] == pytest.approx(-.7854)
    assert float(lowpass.y1[7]) == pytest.approx(-.9)  # Filter memory is not clipped.


def test_collision_is_still_a_failure_and_retains_failed_block_targets():
    pred = np.tile(command(), (100, 1)).astype(np.float32)
    world = FakeWorld(fail=True)
    with pytest.raises(RuntimeError, match='forbidden_contact') as caught:
        execute_block(world, pred, 50, BiquadLowpass(250, 15), reset=True,
                      left_hold=command()[8:], envelope=DeploymentEnvelope())
    details = caught.value.block_diagnostics
    assert details['completed_reference_slots'] == 0
    assert details['clipped_actions'] == 0
    assert len(details['raw_prediction_committed_right']) == 50
    assert details['commanded_right'][7] == pytest.approx(-.02)
