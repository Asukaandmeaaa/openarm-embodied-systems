"""Exercise the real ROS converter used immediately before action dispatch."""
from copy import deepcopy
import json

from moveit_msgs.msg import RobotTrajectory
from rosidl_runtime_py.convert import message_to_ordereddict
from trajectory_msgs.msg import JointTrajectoryPoint

from vision_grasp_baseline.waypoint_place_test import prepare_trajectory


def test_goal_conversion_leaves_plan_json_serializable_and_repeatable(tmp_path):
    trajectory=RobotTrajectory()
    trajectory.joint_trajectory.joint_names=['openarm_right_joint1']
    for q,sec in [(0.,0),(.1,1)]:
        point=JointTrajectoryPoint()
        point.positions=[q];point.velocities=[0.];point.accelerations=[0.]
        point.time_from_start.sec=sec
        trajectory.joint_trajectory.points.append(point)
    planned={'name':'vertical_clearance','trajectory':message_to_ordereddict(trajectory),'audit':{'duration_s':1.}}
    original=deepcopy(planned)
    first=prepare_trajectory(planned,tmp_path/'first.json')
    assert planned==original
    assert json.loads(json.dumps(planned))==json.loads((tmp_path/'first.json').read_text())
    assert message_to_ordereddict(first)==original['trajectory']
    first.joint_trajectory.points[1].positions[0]=.9
    second=prepare_trajectory(planned,tmp_path/'second.json')
    assert second.joint_trajectory.points[1].positions[0]==.1
    assert planned==original
