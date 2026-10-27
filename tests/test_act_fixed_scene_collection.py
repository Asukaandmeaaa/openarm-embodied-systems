"""Dataset and admission checks; no robot interfaces or long physics runs."""
from pathlib import Path
import json
import sys

import numpy as np
import pytest
from PIL import Image
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'official/openarm_mujoco/src')]
from openarm_cube_pick_expert.recording import ExpertRecorder


def test_fixed_scene_recorder_keeps_640x360_and_distinct_actions_states(tmp_path):
    writer=ExpertRecorder(tmp_path,{'tasks':[{'prompt':'place cube'}],'source_domain':'simulation'},frame_shape=(360,640,3))
    writer.start(0)
    frames={key:np.full((360,640,3),25,dtype=np.uint8) for key in ['ceiling','wrist_right']}
    action=np.linspace(-.1,.1,16,dtype=np.float32);state=action+.01
    for t in [1_000_000,34_000_000]:writer.sample(t,action,state,frames)
    writer.finish(True,{'source_trajectory_trial_id':'sim_seed_1'})
    image=next((tmp_path/'episodes/0/cameras/ceiling').glob('*.jpeg'))
    assert Image.open(image).size==(640,360)
    recorded_action=pq.read_table(tmp_path/'episodes/0/action/arms/right/state.parquet')['qpos'].to_pylist()
    recorded_state=pq.read_table(tmp_path/'episodes/0/obs/arms/right/state.parquet')['qpos'].to_pylist()
    np.testing.assert_allclose(recorded_action[0],action[:8])
    np.testing.assert_allclose(recorded_state[0],state[:8])


def test_fixed_scene_rejects_wrong_image_shape_without_committing(tmp_path):
    writer=ExpertRecorder(tmp_path,{},frame_shape=(360,640,3));writer.start(0)
    with pytest.raises(ValueError,match='shape'):
        writer.sample(0,np.zeros(16),np.zeros(16),{'ceiling':np.zeros((600,960,3),dtype=np.uint8)})
    writer.cancel()
    assert not (tmp_path/'episodes').exists()
    assert not list((tmp_path/'.transactions').iterdir())


def test_unapproved_scene_refuses_batch_before_creating_output(tmp_path,monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'scripts'))
    import collect_act_fixed_scene as collector
    scene=tmp_path/'scene';scene.mkdir()
    (scene/'measurements.json').write_text(json.dumps({'approved_for_collection':False}))
    monkeypatch.setattr(collector,'SCENE',scene)
    output=tmp_path/'data'
    monkeypatch.setattr(sys,'argv',['collect_act_fixed_scene.py','--output',str(output)])
    with pytest.raises(SystemExit) as exc:collector.main()
    assert exc.value.code==2
    assert not output.exists()


def test_table_transform_moves_objects_but_not_robot_or_camera(tmp_path,monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'scripts'))
    from act_fixed_scene_pilot import Pilot
    a=Pilot(42,tmp_path/'nominal')
    b=Pilot(42,tmp_path/'random',.01,30.,.02,5.)
    c=Pilot(42,tmp_path/'repeat',.01,30.,.02,5.)
    import mujoco
    np.testing.assert_allclose(b.initial_qpos,c.initial_qpos)
    assert b.randomization==c.randomization
    for name in ('table','black_frame'):
        assert not np.allclose(a.data.xpos[a.model.body(name).id],b.data.xpos[b.model.body(name).id])
    for name in ('camera_ceiling','camera_wrist_right'):
        np.testing.assert_allclose(a.data.cam_xpos[a.model.camera(name).id],b.data.cam_xpos[b.model.camera(name).id])
    np.testing.assert_allclose(a.state(),b.state())
    table=b.model.body('table').id
    R=b.data.xmat[table].reshape(3,3)
    box=b.model.body('black_frame').id
    np.testing.assert_allclose(R.T@(b.data.xpos[box]-b.data.xpos[table]),[0,.01,.009],atol=1e-9)
    local=R.T@(b.data.xpos[b.cube]-b.data.xpos[table])
    np.testing.assert_allclose(local[:2],np.array([-.053,-.196])+b.randomization['cube_translation_in_table_xy_m'])
    # A cube resting centrally in a rotated/moved box passes. A cube at the
    # old world-coordinate box center must not pass once the box is elsewhere.
    b.data.qpos[b.cube_qadr:b.cube_qadr+3]=b.data.xpos[box]+[0,0,.03]
    mujoco.mj_forward(b.model,b.data)
    assert b.box_outcome()==(True,True)
    b.data.qpos[b.cube_qadr:b.cube_qadr+3]=b.data.xpos[box]+R@np.array([.055,0,.03])
    mujoco.mj_forward(b.model,b.data)
    assert b.box_outcome()==(False,True)
