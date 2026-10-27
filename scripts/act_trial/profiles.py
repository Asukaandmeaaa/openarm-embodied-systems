"""Explicit reviewed experiment profiles; no hardware operations."""
import hashlib

import yaml

from scripts.act_trial.speed_contract import ROS_TEACHING_CONFIG


def collision_checks_enabled(profile='commissioning'):
    """Task tests omit MuJoCo collision admission at the operator's request."""
    if profile not in ('commissioning', 'task', 'visual32'):
        raise ValueError('unknown ACT experiment profile')
    return profile == 'commissioning'


def execution_profile(name='commissioning'):
    if name not in ('commissioning', 'task', 'visual32'):
        raise ValueError('unknown ACT experiment profile')
    # Operator requested no elapsed-time stop for task runs on 2026-10-15
    # and visual32 runs on 2026-10-23. Fault/manual/completion stops remain.
    # None is explicit JSON null, not a large or infinite numeric timeout.
    result = {'name':name, 'duration_s':3. if name == 'commissioning' else None,
              'arm_envelope_rad':.04 if name == 'commissioning' else None,
              'command_gripper':name in ('task', 'visual32'), 'automatic_home':False,
              'tracking_policy_version': 3}
    if name == 'task':
        payload = ROS_TEACHING_CONFIG.read_bytes()
        params = yaml.safe_load(payload)['right_kinesthetic_bridge']['ros__parameters']
        opened, closed = float(params['gripper_open_rad']), float(params['gripper_close_rad'])
        # These are the reviewed ROS teaching endpoints, not new model offsets.
        if opened != -.35 or closed != -.10:
            raise ValueError('ROS gripper endpoints differ from reviewed task contract')
        result.update(gripper_limits_rad=[opened,closed], gripper_source=str(ROS_TEACHING_CONFIG),
                      gripper_source_sha256=hashlib.sha256(payload).hexdigest(),
                      contact_policy='gripper tracking diagnostic only; no tracking-error stop or force/contact protection')
    elif name == 'visual32':
        # Keep the existing operator profile name; invalidate old saved trials.
        from scripts.act_trial.closed_contract import execution_contract
        result.update(
            start_contract='mechanical_zero',
            deployment_contract=execution_contract(),
            gripper_limits_rad=[-.45, -.02],
            gripper_source='closed_transit_v2 collection command targets',
            contact_policy=('gripper tracking diagnostic only; no force/contact protection; '
                            'returned zero is episode completion, not task-success evidence'),
            mechanical_zero_completion={
                'arm_tolerance_rad': .06,
                'closed_gripper_range_rad': [-.055, 0.],
                'closed_gripper_command_rad': -.02,
                'departure_arm_rad': .12,
                'departure_gripper_rad': -.20,
                'return_velocity_rad_s': .05,
                'return_settle_s': .50,
                'minimum_active_s': 3.0,
            },
        )
    return result


def profile_from_config(config):
    expected = execution_profile(config.get('execution_profile', {}).get('name', 'commissioning'))
    if config.get('execution_profile') != expected:
        raise ValueError('experiment profile missing/changed; prepare a new trial')
    return expected
