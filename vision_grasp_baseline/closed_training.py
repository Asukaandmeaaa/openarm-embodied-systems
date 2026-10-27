"""Frozen closed-transit training selection and full-episode phase coverage.

Offline files only; no imports from robot-control or camera runtimes.
"""
from bisect import bisect_right
import hashlib
import json
from pathlib import Path
import random

from .collection_io import ORDER
from .gripper_settings import COLLECTION_POLICY,COLLECTION_EVENTS

SOURCE_FILES=('metadata.json','joint_observation.jsonl','demonstration_action.jsonl',
              'camera.jsonl','controller_raw.jsonl','target_right_arm.jsonl',
              'target_right_gripper.jsonl','target_left_arm.jsonl','events.jsonl')


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def eligible(meta):
    return (meta.get('gripper_policy')==COLLECTION_POLICY and not meta.get('diagnostic_only')
            and meta.get('review_status')=='approved' and meta.get('terminal_label')=='returned_zero'
            and all(meta.get(k) is True for k in (
                'operator_marked_success','automatic_task_contract_passed','data_quality_passed')))


def select(source,seed=20261023):
    splits=dict(train=[],validation=[]);counts={};excluded=[];seen_groups=set();seen_plans=set()
    rng=random.Random(seed)
    for tag in ('nearby','spread','rotation'):
        paths=[]
        for file in sorted((Path(source)/tag/'episodes').glob('episode_*/metadata.json')):
            meta=json.loads(file.read_text());path=str(file.parent.resolve())
            if not eligible(meta):excluded.append(path);continue
            group=meta.get('source_trajectory_trial_id');plan=meta.get('complete_trajectory_sha256')
            if not group or not plan:raise ValueError('Missing physical trajectory provenance: '+path)
            # This first recipe refuses repeated/replayed source geometry rather
            # than accidentally leaking identical plans between the splits.
            if group in seen_groups or plan in seen_plans:raise ValueError('Duplicate trajectory group/plan: '+path)
            seen_groups.add(group);seen_plans.add(plan);paths.append(path)
        rng.shuffle(paths)
        nval=max(1,round(len(paths)*.2)) if len(paths)>=5 else 0
        splits['validation'].extend(paths[:nval]);splits['train'].extend(paths[nval:])
        counts[tag]=dict(train=len(paths)-nval,validation=nval)
    if not splits['train'] or not splits['validation']:
        raise ValueError('Need at least five approved independent episodes in one category')
    return dict(schema=1,seed=seed,gripper_policy=COLLECTION_POLICY,counts=counts,splits=splits,
                excluded=excluded,fps=30,image_size_wh=[640,480],joint_order=ORDER,hardware_io=False,
                validation='all frames of every held-out episode; no prefix sampling',
                scope='Only categories and placements included at freeze time; not a physical success guarantee')


def verify_sources(manifest):
    for path,files in manifest['source_sha256'].items():
        for name,expected in files.items():
            if digest(Path(path)/name)!=expected:raise ValueError('Frozen source changed: '+path+'/'+name)


def phase_coverage(path,report):
    """Locate the complete controller timeline on the exported 30Hz grid."""
    path=Path(path);meta=json.loads((path/'metadata.json').read_text())
    events=[json.loads(s) for s in (path/'events.jsonl').read_text().splitlines() if s.strip()]
    dispatch=[r['detail'] for r in events if r['phase']=='continuous_dispatch']
    marks=[r['detail'] for r in events if r['phase']=='timeline_event']
    if len(dispatch)!=1 or tuple(r['name'] for r in marks)!=COLLECTION_EVENTS:
        raise ValueError('Missing/duplicate complete closed-transit events: '+str(path))
    start=(int(dispatch[0]['start_ros_ns'])+int(meta['ros_time_mapping']['offset_ns']))/1e9
    bounds=[start+float(row['time_s']) for row in marks]
    if bounds!=sorted(set(bounds)):raise ValueError('Non-monotonic phase events')
    end=start+float(dispatch[0]['duration_s'])
    counts={name:0 for name in ('hold_start',*COLLECTION_EVENTS,'hold_end')}
    for i in range(report['frame_count']):
        stamp=report['start_timestamp_s']+i/30.
        label='hold_end' if stamp>=end else 'hold_start' if stamp<start else marks[bisect_right(bounds,stamp)-1]['name']
        counts[label]+=1
    if any(counts[name]<1 for name in COLLECTION_EVENTS):
        raise ValueError('Export loses one or more task phases: '+str(counts))
    return dict(frame_counts=counts,all_task_phases_present=True,
                timeline_start_timestamp_s=start,timeline_end_timestamp_s=end)
