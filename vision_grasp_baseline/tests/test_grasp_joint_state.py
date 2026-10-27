"""Real GraspNode state logic with ROS construction/transport stubbed out."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from sensor_msgs.msg import JointState

from vision_grasp_baseline import continuous_plan
from vision_grasp_baseline.grasp import GraspNode,FINGER
from vision_grasp_baseline.hover import HoverNode,NAMES
from vision_grasp_baseline.gripper_settings import TRANSIT_POSITION
from openarm_kinesthetic_recorder.home_runtime import HomeRuntimeNode


def message(finger=-.085):
    return JointState(name=[*NAMES,FINGER],position=[0.]*7+[finger])


@pytest.fixture
def node(monkeypatch):
    def init(self):
        self.expected_names=list(NAMES)
        self.current_positions=None
        self.current_state_monotonic_s=None
    monkeypatch.setattr(HoverNode,'__init__',init)
    return GraspNode()


def test_planner_node_owns_full_state_without_executor_subclass(node):
    assert node.full_joints is None
    msg=message();msg.name.reverse();msg.position.reverse()
    node._joint_state(msg)
    assert node.full_joints is msg
    assert node.current_positions==[0.]*7
    assert node.full_joint_state_monotonic_s==node.current_state_monotonic_s


@pytest.mark.parametrize('kind',['missing_finger','nan_finger','nan_arm','short_positions','duplicate_names'])
def test_incomplete_or_invalid_state_cannot_refresh_arm_only(node,kind):
    node._joint_state(message())
    previous=node.current_state_monotonic_s
    msg=message()
    if kind=='missing_finger':msg.name.pop();msg.position.pop()
    elif kind=='nan_finger':msg.position[-1]=float('nan')
    elif kind=='nan_arm':msg.position[0]=float('nan')
    elif kind=='short_positions':msg.position.pop()
    else:msg.name.append(FINGER);msg.position.append(-.085)
    node._joint_state(msg)
    assert node.full_joints is None
    assert node.full_joint_state_monotonic_s is None
    assert node.current_state_monotonic_s==previous


def test_new_refresh_cannot_reuse_old_full_state(node,monkeypatch):
    node._joint_state(message())
    monkeypatch.setattr(HomeRuntimeNode,'refresh_rigid_state',lambda *a:[0.]*7)
    with pytest.raises(RuntimeError,match='Fresh complete arm/gripper'):
        node.refresh_rigid_state(1.)


@pytest.mark.parametrize('finger,accepted',[(TRANSIT_POSITION,True),(-.085,False),(-.1204,False),(-.45,False)])
def test_actual_geometry_entry_checks_fresh_measured_gripper(node,monkeypatch,finger,accepted):
    def refresh(self,timeout):
        self._joint_state(message(finger))
        return list(self.current_positions)
    monkeypatch.setattr(HomeRuntimeNode,'refresh_rigid_state',refresh)
    pickup=Mock(side_effect=RuntimeError('geometry gate passed'))
    monkeypatch.setattr(continuous_plan,'plan_pickup',pickup)
    monkeypatch.setattr(continuous_plan,'with_edge_clearance',lambda scene,pad:scene)
    monkeypatch.setattr(continuous_plan,'with_top_clearance',lambda scene,pad:scene)
    if accepted:
        with pytest.raises(RuntimeError,match='geometry gate passed'):
            continuous_plan.plan_geometry(node,{},[0.,0.,0.])
        pickup.assert_called_once()
    else:
        with pytest.raises(ValueError,match='closed right gripper'):
            continuous_plan.plan_geometry(node,{},[0.,0.,0.])
        pickup.assert_not_called()


def test_executor_no_longer_overwrites_parent_full_state_subscription(monkeypatch):
    from vision_grasp_baseline import grasp_test
    node=grasp_test.TestNode.__new__(grasp_test.TestNode)
    marker=message()
    def parent(self):self.full_joints=marker
    monkeypatch.setattr(GraspNode,'__init__',parent)
    monkeypatch.setattr(grasp_test,'ActionClient',lambda *a:SimpleNamespace())
    subscribe=Mock(side_effect=AssertionError('Do not duplicate unvalidated full-state subscription'))
    monkeypatch.setattr(grasp_test.TestNode,'create_subscription',subscribe)
    node.__init__()
    assert node.full_joints is marker
    subscribe.assert_not_called()


def test_real_planner_constructor_wires_shared_joint_callback_without_ros_transport(monkeypatch):
    from rclpy.node import Node
    from openarm_kinesthetic_recorder import home_runtime
    subscriptions={}
    monkeypatch.setattr(Node,'__init__',lambda *a,**kw:None)
    monkeypatch.setattr(Node,'create_client',lambda *a,**kw:SimpleNamespace())
    monkeypatch.setattr(Node,'create_publisher',lambda *a,**kw:SimpleNamespace())
    def subscribe(self,msg_type,topic,callback,qos):
        subscriptions[topic]=callback
        return SimpleNamespace()
    monkeypatch.setattr(Node,'create_subscription',subscribe)
    monkeypatch.setattr(home_runtime,'ActionClient',lambda *a,**kw:SimpleNamespace())
    planner=GraspNode()
    assert planner.full_joints is None
    callback=subscriptions['/joint_states']
    assert callback.__func__ is GraspNode._joint_state
    msg=message();callback(msg)
    assert planner.full_joints is msg and planner.current_positions==[0.]*7
