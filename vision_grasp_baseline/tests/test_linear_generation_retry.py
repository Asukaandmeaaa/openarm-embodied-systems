from copy import deepcopy
from types import SimpleNamespace as NS

import pytest

from vision_grasp_baseline.linear_place_plan import request_linear_trajectory


def fixture_node(codes):
    motion=NS(max_velocity_scaling_factor=.03,max_acceleration_scaling_factor=.01,
              start_state=[.1,.2],goal_constraints=['same pose and tolerances'])
    request=NS(motion_plan_request=motion)
    seen=[]
    codes=iter(codes)
    def call(client,request,timeout):
        seen.append(deepcopy(request.motion_plan_request))
        return NS(motion_plan_response=NS(error_code=NS(val=next(codes),message='')))
    return NS(plan_client=object(),call=call),request,seen


def test_conversion_retry_keeps_geometry_and_reduces_generation_speed():
    node,request,seen=fixture_node([99999,1])
    result,attempts=request_linear_trajectory(node,request,'vertical_release')
    assert result.error_code.val==1 and len(attempts)==2
    assert [r.max_velocity_scaling_factor for r in seen]==[.03,.015]
    assert [r.max_acceleration_scaling_factor for r in seen]==[.01,.005]
    assert all(r.start_state==[.1,.2] and r.goal_constraints==['same pose and tolerances'] for r in seen)


@pytest.mark.parametrize('codes,count',[([99999]*4,4),([-10],1)])
def test_failed_or_invalid_path_never_becomes_success(codes,count):
    node,request,seen=fixture_node(codes)
    with pytest.raises(RuntimeError,match='LIN generation failed'):
        request_linear_trajectory(node,request,'vertical_release')
    assert len(seen)==count
