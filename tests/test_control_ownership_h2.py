"""Pure send-boundary tests: no ROS initialization, CAN or hardware calls."""
import base64
from copy import deepcopy
from types import SimpleNamespace as NS

import pytest

from scripts.control_ownership.h2 import capability_frame, decorated_goal, inject_action_capabilities

TOKEN = "a" * 64


class ExecuteGoal(NS):
    pass


class FollowGoal(NS):
    pass


def trajectory(frame="world"):
    return NS(header=NS(frame_id=frame, stamp=NS(sec=123, nanosec=987654321)),
              joint_names=["j1", "j2"], points=[
                  NS(positions=[.0123456789, -.3], velocities=[.02, -.01], accelerations=[0., .4],
                     effort=[], time_from_start=NS(sec=0, nanosec=123456789)),
                  NS(positions=[.2, .1], velocities=[0., 0.], accelerations=[0., 0.],
                     effort=[], time_from_start=NS(sec=2, nanosec=12345))])


@pytest.mark.parametrize("frame", ["", "world", "arm:base/日本語"])
def test_capability_roundtrip_preserves_original_frame(frame):
    parts = capability_frame(frame, TOKEN).split(":")
    assert len(parts) == 4 and parts[:2] == ["openarm-lease", TOKEN]
    assert len(parts[2]) == 32
    assert base64.b64decode(parts[3], altchars=b"-_", validate=True).decode() == frame


@pytest.mark.parametrize("bad_token", [None, "", "a"*63, "G"*64, "A"*64])
def test_bad_capability_rejected_without_echoing_secret(bad_token):
    with pytest.raises(ValueError) as error:
        capability_frame("world", bad_token)
    if bad_token:
        assert bad_token not in str(error.value)


@pytest.mark.parametrize("frame", ["x"*1025, "openarm-lease:unexpected", None])
def test_invalid_or_nested_frame_rejected(frame):
    with pytest.raises(ValueError):
        capability_frame(frame, TOKEN)


@pytest.mark.parametrize("kind", ["moveit", "gripper"])
def test_only_copy_frame_changes_all_numeric_fields_and_original_remain_exact(kind):
    if kind == "moveit":
        original = ExecuteGoal(trajectory=NS(joint_trajectory=trajectory(),
                                            multi_dof_joint_trajectory=trajectory("world")),
                               controller_names=["right_joint_trajectory_controller"])
        select = lambda goal: goal.trajectory.joint_trajectory
    else:
        original = FollowGoal(trajectory=trajectory(""), goal_time_tolerance=NS(sec=2, nanosec=9),
                              path_tolerance=[NS(name="j1", position=.3, velocity=.1)])
        select = lambda goal: goal.trajectory
    saved = deepcopy(original)
    outbound = decorated_goal(original, TOKEN, ExecuteGoal, FollowGoal)
    assert original == saved and outbound is not original
    metadata = select(outbound).header.frame_id
    assert metadata.startswith("openarm-lease:")
    select(outbound).header.frame_id = base64.urlsafe_b64decode(metadata.split(":")[3]).decode()
    assert outbound == original


def test_each_submission_gets_a_unique_nonce():
    goal = FollowGoal(trajectory=trajectory(""))
    frames = {decorated_goal(goal, TOKEN, ExecuteGoal, FollowGoal).trajectory.header.frame_id
              for _ in range(20)}
    assert len(frames) == 20


def test_injection_preserves_send_arguments_and_restores_method_even_on_error():
    calls = []
    result = object()
    class Client:
        def send_goal_async(self, goal, *args, **kwargs):
            calls.append((goal, args, kwargs))
            return result
    method = Client.send_goal_async
    goal = ExecuteGoal(trajectory=NS(joint_trajectory=trajectory()))
    callback = object()
    with pytest.raises(RuntimeError, match="original H2 error"):
        with inject_action_capabilities(TOKEN, action_client_type=Client,
                                        execute_goal_type=ExecuteGoal, follow_goal_type=FollowGoal):
            assert Client().send_goal_async(goal, callback, goal_uuid="same-uuid") is result
            with pytest.raises(ValueError, match="unsupported"):
                Client().send_goal_async(NS(other="unexpected goal"))
            assert len(calls) == 1
            assert calls[0][1:] == ((callback,), {"goal_uuid": "same-uuid"})
            assert calls[0][0].trajectory.joint_trajectory.header.frame_id.startswith("openarm-lease:")
            assert goal.trajectory.joint_trajectory.header.frame_id == "world"
            raise RuntimeError("original H2 error")
    assert Client.send_goal_async is method


@pytest.mark.parametrize("kind", ["moveit", "gripper"])
def test_real_ros_message_serialization_is_identical_after_metadata_restore(kind):
    pytest.importorskip("rclpy.serialization")
    from rclpy.serialization import serialize_message
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
    from control_msgs.action import FollowJointTrajectory
    from moveit_msgs.action import ExecuteTrajectory
    traj = JointTrajectory()
    traj.header.frame_id = "world" if kind == "moveit" else ""
    traj.header.stamp.sec, traj.header.stamp.nanosec = 123, 987654321
    traj.joint_names = ["j1", "j2"]
    point = JointTrajectoryPoint(positions=[.0123456789, -.3], velocities=[.02, -.01],
                                 accelerations=[.0, .4])
    point.time_from_start.sec, point.time_from_start.nanosec = 2, 123456789
    traj.points = [point]
    if kind == "moveit":
        goal = ExecuteTrajectory.Goal()
        goal.trajectory.joint_trajectory = traj
        select = lambda msg: msg.trajectory.joint_trajectory
    else:
        goal = FollowJointTrajectory.Goal(trajectory=traj)
        select = lambda msg: msg.trajectory
    before = serialize_message(goal)
    outbound = decorated_goal(goal, TOKEN, ExecuteTrajectory.Goal, FollowJointTrajectory.Goal)
    assert serialize_message(goal) == before
    metadata = select(outbound).header.frame_id
    select(outbound).header.frame_id = base64.urlsafe_b64decode(metadata.split(":")[3]).decode()
    assert serialize_message(outbound) == before


def test_plan_only_security_still_enforced_and_execute_needs_token(monkeypatch):
    from scripts.control_ownership import h2,security
    monkeypatch.setattr(security,'environment',lambda *a:{'ROS_SECURITY_ENABLE':'true','ROS_SECURITY_STRATEGY':'Enforce'})
    monkeypatch.setenv('ROS_SECURITY_ENABLE','true');monkeypatch.setenv('ROS_SECURITY_STRATEGY','Enforce')
    monkeypatch.delenv('OPENARM_AUTHORITY_TOKEN',raising=False)
    monkeypatch.delenv('ROS_SECURITY_ENCLAVE_OVERRIDE',raising=False)
    monkeypatch.setattr(h2.sys,'argv',['program','--ros-args','--enclave','/openarm/h2'])
    assert h2.validated_token(require_token=False) is None
    with pytest.raises(ValueError,match='connection-bound'):
        h2.validated_token()
    monkeypatch.setenv('ROS_SECURITY_STRATEGY','Permissive')
    with pytest.raises(ValueError,match='enforced DDS'):
        h2.validated_token(require_token=False)
