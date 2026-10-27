from types import SimpleNamespace

import numpy as np
from moveit_msgs.srv import GetPlanningScene

from vision_grasp_baseline.grasp import GraspNode


def test_extra_edge_margin_preserves_measured_table_top_and_object_target():
    def call(client,request):
        if isinstance(request,GetPlanningScene.Request):
            return SimpleNamespace(scene=SimpleNamespace(robot_state=SimpleNamespace(attached_collision_objects=[])))
        return SimpleNamespace(success=True)
    node=SimpleNamespace(call=call,scene_client=object(),apply_client=object())
    scene={'measured':{
        'table':{'normal':[0.,0.,1.],'center_xy':[.4,0.],'top_z':.27,'yaw':0.,'size_xy':[.4,.6]},
        'cube':{'center':[.33,-.17,.29],'size':[.04]*3,'yaw':.1,'support_z':.27},
        'bin':{'center_xy':[.4,.03],'support_z':.27,'rim_height':.04,'outer_size_xy':[.14,.14],'yaw':0.}}}
    original,_=GraspNode.precise_scene(node,scene,[0.]*3,table_edge_padding_m=.02)
    padded,_=GraspNode.precise_scene(node,scene,[0.]*3)
    a,b=[x['objects'][0] for x in [original,padded]]
    assert np.allclose(np.array(b['primitives'][0]['dimensions'])-a['primitives'][0]['dimensions'],[.02,.02,0.])
    assert padded['table_edge_padding_m']==.03
    assert a['primitive_poses']==b['primitive_poses']
    assert original['cube_center']==padded['cube_center']
