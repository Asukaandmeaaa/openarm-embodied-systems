#!/usr/bin/env python3
"""Verify a generated collection using pure math/message construction, no nodes."""
import argparse
import hashlib
import json
from pathlib import Path
from openarm_human_retarget.kinematics import OpenArmRightKinematics
from openarm_kinesthetic_recorder.recorded_motion import validate_collection_trajectory,marker_segment_ranges
from openarm_kinesthetic_recorder.recorded_motion_player import replay_provenance,build_dense_arm_trajectory


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    args=parser.parse_args();root=args.root.resolve()
    report=json.loads((root/'notes/processing_report.json').read_text())
    kin=OpenArmRightKinematics.from_current_xacro();rows=[]
    for row in report['rows']:
        path=Path(row['file']);v=json.loads(path.read_text());info=v['processing']
        assert path.parent==root/'trajectories'
        assert digest(path)==row['sha256']
        source=Path(info['source_path']);old=json.loads(source.read_text())
        assert digest(source)==info['source_sha256']
        assert digest(Path(info['limits_path']))==info['limits_sha256']
        validate_collection_trajectory(v)
        old_ids=[0]+[m['sample_index'] for m in old['markers']]+[len(old['samples'])-1]
        new_ids=[0]+[m['sample_index'] for m in v['markers']]+[len(v['samples'])-1]
        assert [old['samples'][i]['positions_rad'] for i in old_ids]==[v['samples'][i]['positions_rad'] for i in new_ids]
        for a,b in zip(old['markers'],v['markers']):
            assert (a['type'],a['source_button'],a['target_position_rad'])==(b['type'],b['source_button'],b['target_position_rad'])
        for s in info['segments']:
            assert all(lo>=lim for lo,lim in zip(s['position_min_rad'],kin.lower_limits)),path
            assert all(hi<=lim for hi,lim in zip(s['position_max_rad'],kin.upper_limits)),path
            assert max(s['max_acceleration_rad_s2'])<=.5 and max(s['max_jerk_rad_s3'])<=5
            assert s['max_reference_difference_rad']<=.01
            assert s['endpoint_velocity_max']<1e-8 and s['endpoint_acceleration_max']<1e-7
        ranges=marker_segment_ranges(v)
        assert replay_provenance(path,v,1.,ranges,.04)['trajectory_processing']==info
        for segment in ranges:
            msg=build_dense_arm_trajectory(v['samples'],segment['start_sample_index'],segment['end_sample_index'],time_scale=1.)
            assert len(msg.points)==segment['end_sample_index']-segment['start_sample_index']+1
        rows.append(dict(trial_id=v['trial_id'],original_s=old['duration_s'],processed_s=v['duration_s'],
                         tail_original_s=info['segments'][-1]['original_duration_s'],
                         tail_processed_s=info['segments'][-1]['processed_duration_s'],
                         max_joint_path_difference_rad=max(s['max_reference_difference_rad'] for s in info['segments'])))
    expected=[Path(s).resolve() for s in (root/'notes/replay14_files.txt').read_text().splitlines()]
    assert expected==sorted((root/'trajectories').glob('*.json'))
    assert set(expected)=={Path(r['file']) for r in report['rows']}
    outcome=dict(hardware_operations=False,all_passed=True,checks=['source_and_candidate_hash','current_limits_hash','trajectory_contract','URDF_position_bounds','configured_acceleration_jerk','keyframe_identity','controller_message_build','provenance','directory_selection_match'],rows=rows,blocked=report['blocked'])
    (root/'notes/offline_integration.json').write_text(json.dumps(outcome,indent=2)+'\n')
    print('PASS',len(rows),'candidates; blocked',len(report['blocked']))
    print('Reference total seconds',sum(r['original_s'] for r in rows),'->',sum(r['processed_s'] for r in rows))

if __name__=='__main__':main()
