from types import SimpleNamespace as NS
from unittest.mock import Mock

import numpy as np

from vision_grasp_baseline.continuous_plan import plan_pickup,check_gripper_sweep
from vision_grasp_baseline.gripper_settings import TRANSIT_POSITION,OPEN_POSITION
import pytest


def test_rotated_cube_tries_equivalent_face_and_keeps_start_and_contact_checks(monkeypatch):
    measured=dict(cube_yaw=-.7,cube_center=[.3,-.1,.28],support_z=.26)
    from vision_grasp_baseline.hover import box
    cube=box('vision_baseline_cube',[.3,-.1,.28],[.04]*3)
    node=NS(precise_scene=Mock(return_value=(measured,cube)),finger_contacts=Mock(),update_object=Mock())
    sweep=Mock()
    monkeypatch.setattr('vision_grasp_baseline.continuous_plan.check_gripper_sweep',sweep)
    calls=[]
    def pose(label,target,q,gripper,**kwargs):
        calls.append((label,target.copy(),list(q),gripper))
        if len(calls)==1:raise RuntimeError('no goal IK')
        return None,dict(trajectory=dict(joint_trajectory=dict(points=[dict(positions=[.1]*7)])))
    node.plan_pose=pose
    scene=dict(table_collision_envelope=dict(center=[0,0,0],rotation=np.eye(3).tolist(),size=[1,1,1]))
    _,_,target,pre,approach=plan_pickup(node,scene,[0,0,0],[0.]*7,[.002,0,0])
    assert pre['grasp_candidate']['square_turn_rad']==np.pi/2
    assert len(pre['grasp_candidate']['rejected'])==1
    assert calls[0][2]==calls[1][2]==[0.]*7
    assert [c[3] for c in calls]==[TRANSIT_POSITION,TRANSIT_POSITION,OPEN_POSITION]
    sweep.assert_called_once_with(node,[.1]*7,TRANSIT_POSITION,OPEN_POSITION,'pregrasp_open')
    np.testing.assert_allclose(target[:3,3],[.302,-.1,.315])
    assert all(call.args==(False,) for call in node.finger_contacts.call_args_list)
    assert node.update_object.call_count==2
    assert list(node.update_object.call_args.args[0].primitives[0].dimensions)==pytest.approx([.1]*3)
    assert list(cube.primitives[0].dimensions)==pytest.approx([.04]*3)


def test_gripper_sweep_rejects_collision_without_contact_permission():
    from moveit_msgs.msg import RobotState
    node=NS(valid_client=object(),state=Mock(return_value=RobotState()),
            call=Mock(side_effect=[NS(valid=True),NS(valid=False)]))
    with pytest.raises(RuntimeError,match='gripper sweep collision'):
        check_gripper_sweep(node,[0.]*7,OPEN_POSITION,TRANSIT_POSITION,'transit_close')
    assert node.call.call_count==2
