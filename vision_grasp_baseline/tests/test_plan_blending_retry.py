import json
from pathlib import Path
from unittest.mock import Mock
import pytest
from vision_grasp_baseline import continuous_plan


@pytest.mark.parametrize('all_collide',[False,True])
def test_collision_reduces_blending_before_any_plan_is_published(tmp_path,monkeypatch,all_collide):
    source=Path(__file__).resolve().parent/'fixtures/cube_yaw_jitter/zero_scene/scene.json'
    registration=tmp_path/'registration.json';registration.write_text('{"translation_world_m":[0,0,0]}')
    model=tmp_path/'robot.urdf';model.write_text('test model')
    output=tmp_path/'plan'
    monkeypatch.setattr('sys.argv',['planner','--scene',str(source),'--registration',str(registration),
                                  '--robot-model',str(model),'--output',str(output)])
    monkeypatch.setattr(continuous_plan,'validated_token',lambda *args:None)
    monkeypatch.setattr(continuous_plan.rclpy,'init',lambda:None)
    monkeypatch.setattr(continuous_plan.rclpy,'shutdown',lambda:None)
    monkeypatch.setattr(continuous_plan,'GraspNode',Mock)
    monkeypatch.setattr(continuous_plan,'plan_geometry',lambda *args:([[],[],[]],None,None,None))
    monkeypatch.setattr(continuous_plan,'read_limits',lambda *args:({}, {}, {}))
    sigmas=[]
    def retime(*args,**kwargs):
        sigmas.append(json.loads((output/'geometry.json').read_text())['smoothing_sigma_s'])
        (output/'timing.json').write_text('{"duration_s":33,"events":[]}')
    monkeypatch.setattr(continuous_plan.subprocess,'run',retime)
    def validate(*args):
        assert not (output/'trajectory.json').exists()
        if all_collide or len(sigmas)==1:raise RuntimeError('Interpolated path collision at 32.270s: table')
        return {'passed':True}
    monkeypatch.setattr(continuous_plan,'validate_timeline',validate)
    if all_collide:
        with pytest.raises(RuntimeError,match='collision'):continuous_plan.main()
        assert sigmas==[.12,.06,.03] and not (output/'trajectory.json').exists()
    else:
        continuous_plan.main()
        assert sigmas==[.12,.06]
        assert json.loads((output/'trajectory.json').read_text())['validation']['passed']
