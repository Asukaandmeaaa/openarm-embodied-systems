"""Capture/replay admission and collision regressions; no motor interfaces."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts/home_replay'))
from core import NAMES, check_driver_contract, check_start, decode, dense_samples, digest, preserve_nonfinite, sample, validate_bundle, write_json
from export import export_capture
from simulate import Scene, validate
import validation_contract
from validation_contract import aligned_pose, replay_arrays, simulation_output

EVAL = ROOT/'official/dora-openarm-evaluation'
KIN = ROOT.parent/'openarm_kinesthetic_ws'


def bundle():
    home = json.loads((EVAL/'guarded-home-right.json').read_text())['home']
    q = home + [0]*8
    return {'schema':'openarm_recorded_home_v1', 'joint_names':NAMES,
            'phase':'H2', 'source':'command', 'synthetic_test_fixture':True,
            'interpolation':'piecewise_linear_recorded_samples',
            'time_s':[0., .1], 'positions_rad':[q,q],
            'measured_start_rad':q, 'measured_end_rad':q,
            'home_config':yaml.safe_load((KIN/'ros2_ws/src/openarm_kinesthetic_recorder/config/home_initialization.yaml').read_text()),
            'driver_config':yaml.safe_load((EVAL/'openarm_real_ros_coordinates.yaml').read_text()),
            'h2_execution':{'success':True}, 'physical_replay_authorized':False}


def test_decode_mit_bits_and_feedback():
    # Midscale known raw frame; values are quantized, never round to false zero.
    d = decode(1, bytes.fromhex('8000800800800800'))
    assert d['kind']=='command' and d['axis']==0
    assert d['q']==pytest.approx(12.5/65535)
    assert d['kp']==pytest.approx(2048*500/4095)
    assert d['kd']==pytest.approx(2048*5/4095)
    fb = decode(0x11, bytes.fromhex('1180008008002021'))
    assert fb['status']==1 and fb['q']==d['q']
    assert fb['temperature_rotor']==33
    assert decode(1, b'\xff'*7+b'\xfc')['operation']=='enable'
    assert decode(0x20000001, b'\0'*8) is None
    assert decode(0x7ff, b'\0'*8) is None


def test_ros_parameter_infinity_preserved_without_stopping_capture():
    encoded=preserve_nonfinite({'limits':[float('inf'),-float('inf'),float('nan')], 'position':[.1]})
    restored=json.loads(json.dumps(encoded,allow_nan=False))
    assert restored['limits']==[{'__ros_nonfinite_float__':v} for v in ('inf','-inf','nan')]
    assert restored['position']==[.1]


@pytest.mark.parametrize('bad', ['nan','order','time','start'])
def test_refuse_bad_bundle_or_wrong_start(bad):
    b = bundle()
    if bad=='nan':
        b['positions_rad'][0] = [float('nan')]*16
    elif bad=='order':
        b['joint_names'] = NAMES[::-1]
    elif bad=='time':
        b['time_s'] = [0.,0.]
    else:
        with pytest.raises(ValueError, match='start mismatch'):
            check_start(b, np.array(b['measured_start_rad'])+.1)
        return
    with pytest.raises(ValueError):
        validate_bundle(b)


def test_recorded_interior_waypoints_are_never_shortcut():
    b=bundle()
    middle=list(b['positions_rad'][0]); middle[0]+=.2
    b['time_s']=[0.,.05,.1]
    b['positions_rad']=[b['positions_rad'][0],middle,b['positions_rad'][-1]]
    np.testing.assert_allclose(sample(b,.05),middle)
    dense=list(dense_samples(b))
    assert max(q[0] for _,q in dense)==pytest.approx(middle[0])
    assert max(np.max(np.abs(dense[i][1]-dense[i-1][1])) for i in range(1,len(dense)))<=.00200001


def make_capture(tmp_path):
    root=tmp_path/'capture'; root.mkdir(); (root/'snapshots').mkdir()
    b=bundle()
    (root/'snapshots/home_config.yaml').write_text(yaml.safe_dump(b['home_config']))
    (root/'snapshots/driver_config.yaml').write_text(yaml.safe_dump(b['driver_config']))
    base=10**18
    for name, offset in [('H2_start',.1),('H2_end',1.)]:
        write_json(root/f'mark_{name}.json', {'wall_time_ns':base+int(offset*1e9)})
    trajectory={'joint_trajectory':{'joint_names':NAMES[:7], 'points':[
        {'positions':b['positions_rad'][0][:7], 'time_from_start':{'sec':i,'nanosec':0}}
        for i in (0,1)]}}
    plan_hash=hashlib.sha256(json.dumps(trajectory,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    h2={'success':True,'state':'HOME_READY','home_gripper':{'accepted':True},
        'config_sha256':digest(root/'snapshots/home_config.yaml'),
        'started_wall_time_s':(base+200000000)/1e9,
        'completed_wall_time_s':(base+900000000)/1e9,
        'stages':[{'state':'ACCEPTED','robot_trajectory':trajectory,'trajectory_sha256':plan_hash}]*2}
    write_json(root/'h2_execution.json',h2)
    rows=[]
    for tick in range(56):
        for axis in range(16):
            for kind in ('command','feedback'):
                stamp=base+tick*20000000+axis*1000+(500 if kind=='feedback' else 0)
                d={'kind':kind,'axis':axis%8,'q':b['positions_rad'][0][axis],
                   'dq':0.,'torque':.1,'kp':70.,'kd':2.,'status':1}
                rows.append({'kernel_wall_ns':stamp,'receive_wall_ns':stamp+1000,
                             'receive_monotonic_ns':stamp-base+1000,
                             'side':'right' if axis<8 else 'left','decoded':d})
    (root/'can.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    ros_rows=[{'topic':'/joint_states','receive_wall_ns':base+800000000,
               'message':{'name':NAMES,'position':b['positions_rad'][0],
                          'header':{'stamp':{'sec':base//10**9,'nanosec':800000000}}}},
              {'topic':'/right_joint_trajectory_controller/controller_state',
               'receive_wall_ns':base+800000000, 'message':{}}]
    (root/'ros.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in ros_rows))
    reseal(root)
    return root


def reseal(root):
    write_json(root/'manifest.json',{'complete':True,'counts':{'socket_drops':0,'can_errors':0},
        'files':{str(p.relative_to(root)):digest(p) for p in root.rglob('*') if p.is_file() and p.name!='manifest.json'}})


def test_export_preserves_synchronized_all_axis_data_and_home(tmp_path):
    root=make_capture(tmp_path)
    out=export_capture(root)
    assert len(out['positions_rad'][0])==16
    assert len(out['recorded_mit_parameters'][0])==16
    assert out['time_s'][0]==0 and out['time_s'][-1]==pytest.approx(.9)
    assert out['post_H2_comparison']['max_error_rad']<1e-9
    assert not out['physical_replay_authorized']


@pytest.mark.parametrize('failure', ['tamper','gap','prefix','stale_report','lifecycle','clock','ros_mismatch'])
def test_export_rejects_incomplete_or_wrong_capture(tmp_path,failure):
    root=make_capture(tmp_path)
    if failure=='tamper':
        with (root/'can.jsonl').open('a') as f: f.write('\n')
    elif failure=='stale_report':
        p=root/'h2_execution.json'; data=json.loads(p.read_text()); data['completed_wall_time_s']=1.; write_json(p,data); reseal(root)
    elif failure=='ros_mismatch':
        p=root/'ros.jsonl'; data=[json.loads(x) for x in p.read_text().splitlines()]
        data[0]['message']['position'][0]+=.1
        p.write_text(''.join(json.dumps(r)+'\n' for r in data)); reseal(root)
    else:
        p=root/'can.jsonl'; rows=[json.loads(x) for x in p.read_text().splitlines()]
        start=json.loads((root/'mark_H2_start.json').read_text())['wall_time_ns']
        if failure=='prefix': rows=[r for r in rows if r['kernel_wall_ns']>=start]
        elif failure=='gap': rows=[r for r in rows if not start+200000000<r['kernel_wall_ns']<start+500000000]
        elif failure=='clock': rows[500]['receive_monotonic_ns']+=50000000
        else: rows[500]['decoded']={'kind':'lifecycle','axis':0,'operation':'disable'}
        p.write_text(''.join(json.dumps(r)+'\n' for r in rows)); reseal(root)
    with pytest.raises(ValueError):
        export_capture(root)


def test_decoder_contract_rejects_other_motor_order():
    b=bundle(); b['driver_config']['motor_config']['send_ids'][0]=8
    with pytest.raises(ValueError):
        check_driver_contract(b['driver_config'])


def test_current_dora_side_shortcut_is_rejected_by_collision_model():
    b=bundle(); scene=Scene(b)
    cfg=json.loads((EVAL/'guarded-home-right.json').read_text())
    q=np.zeros(16); q[:7]=np.array(cfg['side_prehome'][:7])*.5; q[7]=-.0528343633
    hits=scene.put(q)
    assert any('robot_table_collision' in x['failure_labels'] for x in hits)


def test_safe_static_simulation_does_not_authorize_hardware(tmp_path):
    path=tmp_path/'synthetic.json'; write_json(path,bundle())
    report=validate(path,tmp_path/'validation.json')
    assert report['geometric_checks_passed'], report['failures']
    assert report['physical_replay_authorized'] is False


def ros_fixture(tmp_path, monkeypatch):
    """Synthetic captured ROS bounds reproduce the actual scope/limit failures."""
    b = bundle()
    scene = Scene(b)
    limits = scene.model.jnt_range[scene.joint_ids].copy()
    limits[[3, 11], 0] = -.02  # Existing ROS encoder-zero allowance.
    root = tmp_path/'ros_capture'; (root/'snapshots').mkdir(parents=True)
    joints = ''.join(f'<joint name="{name}" type="revolute"><limit lower="{lo}" upper="{hi}" velocity="5"/></joint>'
                     for name, (lo, hi) in zip(NAMES, limits))
    (root/'snapshots/robot_description.urdf').write_text('<robot>'+joints+'</robot>')
    reseal(root)
    planning = tmp_path/'moveit_limits.yaml'
    planning.write_text(yaml.safe_dump({'joint_limits': {
        NAMES[5]: {'has_position_limits': True, 'min_position': -.7654, 'max_position': .7654}}}))
    monkeypatch.setattr(validation_contract, 'MOVEIT_LIMITS', planning)
    b['capture_directory'] = str(root)
    b['capture_manifest_sha256'] = digest(root/'manifest.json')
    b['positions_rad'] = [list(q) for q in b['positions_rad']]
    return b


def save_fixture(tmp_path, b):
    b['measured_positions_rad'] = copy.deepcopy(b['positions_rad'])
    b['measured_start_rad'] = list(b['positions_rad'][0])
    b['measured_end_rad'] = list(b['positions_rad'][-1])
    path = tmp_path/'trajectory.json'; write_json(path, b)
    return path


def test_parked_left_limits_do_not_block_right_only_but_still_block_all(tmp_path, monkeypatch):
    b = ros_fixture(tmp_path, monkeypatch)
    for q in b['positions_rad']:
        q[11] = -.002; q[15] = -.0193
    path = save_fixture(tmp_path, b)
    right = validate(path, tmp_path/'right.json', scope='right')
    assert right['geometric_checks_passed'], right['failures']
    assert not right['physical_replay_authorized']
    assert [v['joint_name'] for v in right['joint_limit_violations']] == [NAMES[15]]
    assert right['joint_limit_violations'][0]['blocks_simulation'] is False
    assert right['effective_position_limits_rad'][11][0] == -.02
    all_arms = validate(path, tmp_path/'all.json')
    assert 'joint_limits' in all_arms['failures']


@pytest.mark.parametrize('axis,value', [(3, -.025), (5, .775), (7, -.80)])
def test_right_ros_hard_and_moveit_limits_still_reject(tmp_path, monkeypatch, axis, value):
    b = ros_fixture(tmp_path, monkeypatch)
    for q in b['positions_rad']: q[axis] = value
    path = save_fixture(tmp_path, b)
    report = validate(path, tmp_path/'report.json', scope='right')
    assert 'joint_limits' in report['failures']
    assert any(v['joint_name'] == NAMES[axis] and v['blocks_simulation'] for v in report['joint_limit_violations'])


def test_right_scope_retains_left_measured_collision_context_and_only_emits_right():
    b = bundle(); b['measured_positions_rad'] = copy.deepcopy(b['positions_rad'])
    b['measured_positions_rad'][0][11] = -.002
    _, q, _ = replay_arrays(b, 'right')
    assert q[0,11] == -.002 and b['positions_rad'][0][11] == 0
    name, values = simulation_output(q[0], 'right')
    assert name == 'simulated_position_right' and values.shape == (8,)
    del b['measured_positions_rad']
    with pytest.raises(ValueError, match='collision context'): replay_arrays(b, 'right')


def test_right_scope_does_not_exclude_left_collisions(tmp_path, monkeypatch):
    b = ros_fixture(tmp_path, monkeypatch)
    for q in b['positions_rad']: q[11] = -.002
    path = save_fixture(tmp_path, b)
    def collision_with_parked_left(self, q):
        assert q[11] == -.002
        return [{'distance_m': -.001, 'failure_labels': ['robot_robot_collision']}]
    monkeypatch.setattr(Scene, 'put', collision_with_parked_left)
    assert 'collision' in validate(path, tmp_path/'report.json', scope='right')['failures']


def test_captured_ros_limits_cannot_be_changed_after_sealing(tmp_path, monkeypatch):
    b = ros_fixture(tmp_path, monkeypatch); path = save_fixture(tmp_path, b)
    with (Path(b['capture_directory'])/'snapshots/robot_description.urdf').open('a') as f: f.write(' ')
    with pytest.raises(ValueError, match='URDF hash mismatch'):
        validate(path, tmp_path/'report.json', scope='right')


def test_command_archive_preserves_start_offset_without_replay_admission(tmp_path):
    root = make_capture(tmp_path)
    path = root/'can.jsonl'; rows = [json.loads(x) for x in path.read_text().splitlines()]
    for r in rows:
        if r['side'] == 'right' and r['decoded']['axis'] == 7 and r['decoded']['kind'] == 'command':
            r['decoded']['q'] += .044
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows)); reseal(root)
    b = export_capture(root)
    assert not b['start_alignment']['passed'] and not b['physical_replay_authorized']
    assert b['positions_rad'][0][7]-b['measured_start_rad'][7] == pytest.approx(.044)
    with pytest.raises(ValueError, match='start mismatch'): check_start(b, b['measured_start_rad'])


def test_fast_playback_interpolation_preserves_every_axis_and_boundary():
    rng = np.random.default_rng(9)
    t = np.r_[0., np.cumsum(rng.uniform(.001, .03, 500))]
    q = rng.normal(size=(len(t), 16))
    for stamp in [-1., *t[::7], *rng.uniform(0, t[-1], 30), t[-1], t[-1]+1]:
        expected = [np.interp(stamp, t, q[:, i]) for i in range(16)]
        np.testing.assert_allclose(aligned_pose(t, q, stamp), expected, atol=1e-14)
