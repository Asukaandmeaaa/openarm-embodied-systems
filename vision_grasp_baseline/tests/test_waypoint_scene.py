"""Regression for MoveIt's attach-then-world-update scene diff ordering."""
from copy import deepcopy
import json
from types import SimpleNamespace
from unittest.mock import patch

from moveit_msgs.msg import AttachedCollisionObject,CollisionObject
import pytest

from vision_grasp_baseline.waypoint_place_test import preflight_waypoints,scene_attachment


class SceneNode:
    apply_client=object()

    def __init__(self):
        self.attached={'cube'};self.world=set();self.calls=0

    def call(self,client,request):
        self.calls+=1;ok=True
        for body in request.scene.robot_state.attached_collision_objects:
            key=body.object.id
            if body.object.operation==CollisionObject.ADD:
                self.world.discard(key);self.attached.add(key)
            else:
                self.attached.discard(key);self.world.add(key)
        for body in request.scene.world.collision_objects:
            if body.operation==CollisionObject.REMOVE:
                ok=body.id in self.world
                self.world.discard(body.id)
            else:self.world.add(body.id)
        return SimpleNamespace(success=ok)


def objects():
    body=AttachedCollisionObject();body.object.id='cube'
    cube=CollisionObject();cube.id='cube'
    return body,cube


def test_restore_consumes_world_copy_without_redundant_remove(tmp_path):
    node=SceneNode();attached,cube=objects()
    steps=[('vertical_release',None,False),('vertical_retreat',None,False),('horizontal_return_1',None,False)]
    with patch('vision_grasp_baseline.waypoint_place_test.plan_linear',side_effect=lambda n,label,q,*a,**k:(q,{'name':label})):
        plans=preflight_waypoints(node,attached,cube,[0.]*7,steps,tmp_path/'preflight.json')
    assert len(plans)==3
    assert node.attached=={'cube'} and not node.world
    assert json.loads((tmp_path/'preflight.json').read_text())['success']


def test_failed_empty_arm_plan_restores_original_attachment_and_saves_progress(tmp_path):
    node=SceneNode();attached,cube=objects()
    steps=[('vertical_release',None,False),('vertical_retreat',None,False)]
    with patch('vision_grasp_baseline.waypoint_place_test.plan_linear',side_effect=[([0.]*7,{'name':'vertical_release'}),RuntimeError('unreachable')]):
        with pytest.raises(RuntimeError,match='unreachable'):
            preflight_waypoints(node,attached,cube,[0.]*7,steps,tmp_path/'preflight.json')
    assert node.attached=={'cube'} and not node.world
    report=json.loads((tmp_path/'preflight.json').read_text())
    assert not report['success'] and report['error']=='unreachable' and len(report['stages'])==1


def test_conflicting_same_id_attach_remove_is_rejected_before_scene_change():
    node=SceneNode();attached,cube=objects();cube.operation=CollisionObject.REMOVE
    with pytest.raises(ValueError,match='Attach consumes'):
        scene_attachment(node,attached,world=deepcopy(cube))
    assert node.calls==0
