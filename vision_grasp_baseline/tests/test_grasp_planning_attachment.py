"""Virtual lift must survive MoveIt's delayed world-diff ACM cleanup."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
from moveit_msgs.msg import AttachedCollisionObject, CollisionObject
from moveit_msgs.srv import GetPlanningScene

from vision_grasp_baseline.grasp import GraspNode


class Monitor:
    apply_client = object()
    scene_client = object()
    planning_attachment = GraspNode.planning_attachment

    def __init__(self):
        self.cube = CollisionObject(); self.cube.id = 'cube'
        self.attached = AttachedCollisionObject(); self.attached.object = deepcopy(self.cube)
        self.held = {}
        self.world = {'cube': self.cube}
        self.allowed_support_contact = False
        self.destroy_pending = False
        self.calls = []

    def publish_diff(self):
        # PlanningScene::pushDiffs removes an ACM entry on world DESTROY unless
        # the monitored robot state contains an attached body with that ID.
        if self.destroy_pending and 'cube' not in self.held:
            self.allowed_support_contact = False
        self.destroy_pending = False

    def call(self, client, request):
        if isinstance(request, GetPlanningScene.Request):
            return SimpleNamespace(scene=SimpleNamespace(
                robot_state=SimpleNamespace(attached_collision_objects=list(self.held.values())),
                world=SimpleNamespace(collision_objects=list(self.world.values()))))
        self.calls.append(deepcopy(request))
        for body in request.scene.robot_state.attached_collision_objects:
            if body.object.operation == CollisionObject.REMOVE:
                self.held.pop(body.object.id, None)
            else:
                self.held[body.object.id] = body
                if body.object.id in self.world:
                    del self.world[body.object.id]
                    self.destroy_pending = True
        for body in request.scene.world.collision_objects:
            if body.operation == CollisionObject.REMOVE:
                self.world.pop(body.id, None)
                self.destroy_pending = True
            else:
                self.world[body.id] = body
        return SimpleNamespace(success=True)


def test_old_request_local_attachment_loses_contact_permission_on_publish():
    node = Monitor()
    node.world.clear(); node.destroy_pending = True
    node.allowed_support_contact = True
    node.publish_diff()
    assert not node.allowed_support_contact


@pytest.mark.parametrize('planning_fails', [False, True])
def test_monitored_attachment_survives_publish_and_restores_on_all_exits(planning_fails):
    node = Monitor()

    def plan():
        with node.planning_attachment(node.cube, node.attached):
            assert 'cube' in node.held and 'cube' not in node.world
            node.allowed_support_contact = True
            node.publish_diff()
            assert node.allowed_support_contact
            if planning_fails:
                raise RuntimeError('planner failed')

    if planning_fails:
        with pytest.raises(RuntimeError, match='planner failed'):
            plan()
    else:
        plan()
    assert not node.held and 'cube' in node.world
    assert node.attached.object.operation == CollisionObject.ADD
    # Attach consumes the world body. No redundant world REMOVE is sent.
    assert not node.calls[0].scene.world.collision_objects
    assert node.calls[-1].scene.world.collision_objects[0].id == 'cube'
