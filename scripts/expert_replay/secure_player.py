"""Reuse the recorded-motion player through the existing secured ROS gateway.

No new DDS permissions. Recorded samples/markers/timing and validation remain
in the original player. Dense arm FJT is carried by MoveIt ExecuteTrajectory,
whose controller endpoint is already remapped to the authorized gateway.
Collection-only publications are written locally, never published as ACT data.
"""
import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import sys
import time


def wait_for_collision_service(node, log, *, timeout_s=10., clock=time.monotonic, spin=None):
    """Bounded startup discovery, before validating any path or sending goals.

Keep the original per-state RPC deadline and collision rejection unchanged.
This only gives a new authenticated DDS participant time to discover MoveIt.
"""
    if spin is None:
        import rclpy
        spin = lambda: rclpy.spin_once(node, timeout_sec=.05)
    started = clock()
    log('collision_service_discovery_started', timeout_s=timeout_s)
    while True:
        reason = node.stop_reason()
        if reason: raise RuntimeError(reason)
        if node.validity.service_is_ready():
            log('collision_service_discovery_ready', elapsed_s=clock()-started)
            return
        if clock()-started >= timeout_s:
            raise RuntimeError('/check_state_validity unavailable after startup discovery deadline; no trajectory dispatched')
        spin()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audit-file', type=Path, required=True)
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    from scripts.control_ownership.h2 import validated_token, inject_action_capabilities
    from openarm_kinesthetic_recorder import recorded_motion_player as player
    from moveit_msgs.msg import RobotTrajectory, MoveItErrorCodes
    from action_msgs.msg import GoalStatus
    from rosidl_runtime_py.convert import message_to_ordereddict
    options = player.parse_arguments()
    if options.return_home or options.start_marker is not None or options.stop_after_marker is not None:
        raise ValueError('expert demo entry requires one complete marked trajectory')
    payload = player.load_recorded_motion(options.trajectory)
    player.validate_collection_trajectory(payload)
    token = validated_token(require_token=not options.plan_only)
    audit = args.audit_file.open('x', buffering=1)

    def log(kind, **fields):
        audit.write(json.dumps({'kind': kind, 'source': 'expert_recording_not_ACT',
                                'wall_ns': time.time_ns(), 'monotonic_ns': time.monotonic_ns(),
                                **fields}, allow_nan=False)+'\n')

    class LocalPublisher:
        def __init__(self, topic): self.topic = topic
        def publish(self, message):
            log('player_event', topic=self.topic, message=message_to_ordereddict(message))

    class NoRecorder:
        def service_is_ready(self): return False
        def wait_for_service(self, **kwargs): return False

    class SecuredPlayer(player.RecordedMotionPlayer):
        def __init__(self):
            # Matches the existing H2 enclave's service permissions.
            super().__init__('home_guarded_executor')

        def create_publisher(self, msg_type, topic, *a, **kw):
            if topic in ('/openarm_kinesthetic/expert_action',
                         '/openarm_kinesthetic/expert_gripper_event',
                         '/openarm_kinesthetic/expert_replay_status'):
                return LocalPublisher(topic)
            return super().create_publisher(msg_type, topic, *a, **kw)

        def create_subscription(self, msg_type, topic, *a, **kw):
            if topic == '/openarm_kinesthetic/gamepad_status':
                # Parent reads the real joystick and freezes the gateway on
                # RT/RB/disconnection. No teleoperation publisher is launched.
                return None
            return super().create_subscription(msg_type, topic, *a, **kw)

        def create_client(self, srv_type, name, *a, **kw):
            if name == '/real_episode_recorder/finish_failure': return NoRecorder()
            return super().create_client(srv_type, name, *a, **kw)

        def validate_marked_path(self, *args, **kwargs):
            wait_for_collision_service(self, log)
            return super().validate_marked_path(*args, **kwargs)

        def on_all_joint_state(self, message):
            super().on_all_joint_state(message)
            log('measured_joint_state', message=message_to_ordereddict(message))

        def on_controller_state(self, message):
            super().on_controller_state(message)
            log('arm_controller_state', message=message_to_ordereddict(message))

        def on_gripper_controller_state(self, message):
            super().on_gripper_controller_state(message)
            log('gripper_controller_state', message=message_to_ordereddict(message))

        def execute_dense_arm_segment(self, payload, start_sample, end_sample, time_scale, segment_index):
            expected = payload['samples'][start_sample]['positions_rad'][:7]
            current = [self.all_positions[n] for n in player.ARM_NAMES]
            if max(abs(a-b) for a,b in zip(current, expected)) > .06:
                raise RuntimeError(f'MARKED_SEGMENT_{segment_index}_START_CHANGED')
            if start_sample == end_sample: return
            trajectory = RobotTrajectory()
            trajectory.joint_trajectory = player.build_dense_arm_trajectory(
                payload['samples'], start_sample, end_sample, time_scale=time_scale)
            log('recorded_arm_segment', index=segment_index,
                trajectory=message_to_ordereddict(trajectory))
            result = self.execute_same_trajectory(trajectory,
                player.duration_seconds(trajectory.joint_trajectory.points[-1].time_from_start),
                timeout_margin_s=5.)
            log('arm_segment_result', index=segment_index, result=result)
            if (result['action_status'] != GoalStatus.STATUS_SUCCEEDED
                    or result['moveit_error_code'] != MoveItErrorCodes.SUCCESS):
                raise RuntimeError(f'MARKED_SEGMENT_{segment_index}_EXECUTION_FAILED')
            self.wait_for_arm_target(payload['samples'][end_sample]['positions_rad'][:7],
                                     tolerance_rad=self.keyframe_settle_tolerance_rad)

    # Force the reviewed gripper route; do not accept an arbitrary ROS remap.
    expected_remap = '/right_gripper_controller/follow_joint_trajectory:=/authorized/right_gripper_controller/follow_joint_trajectory'
    if any(':=' in value and value != expected_remap for value in remaining):
        raise ValueError('unreviewed ROS remapping')
    if expected_remap not in remaining: raise ValueError('authorized gripper remapping required')
    original = player.RecordedMotionPlayer
    player.RecordedMotionPlayer = SecuredPlayer
    try:
        with inject_action_capabilities(token) if token else nullcontext():
            return player.main()
    finally:
        player.RecordedMotionPlayer = original
        audit.close()


if __name__ == '__main__':
    raise SystemExit(main())
