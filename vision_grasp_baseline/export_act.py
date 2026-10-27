"""Export approved visual-expert episodes at 30 Hz, keeping native 640x480 images."""
import argparse
import json
from pathlib import Path

import cv2

from .collection_io import ORDER, CAMERAS, audit_episode, write_json
from scripts.convert_real_episodes_act import prepare
from .gripper_settings import COLLECTION_POLICY,require_closed_transit


def save_original_episode(dataset,path,prepared,task):
    """Public pre-built episode API embeds source JPEG bytes without transcoding.

    All values/timestamps match add_frame; source paths are only read. The
    resulting Parquet embeds bytes, so training does not depend on these paths.
    """
    streams,q,indices,entry=prepared
    buffer={key:[] for key in dataset.features}
    count=entry['frame_count']
    buffer.update(size=count,task=[task]*count,episode_index=dataset.meta.total_episodes,
                  frame_index=list(range(count)),timestamp=[i/30 for i in range(count)])
    for index in range(count):
        buffer['observation.state'].append(q['joint_observation'][indices['joint_observation'][index]].copy())
        buffer['action'].append(q['demonstration_action'][indices['demonstration_action'][index]].copy())
        for key in CAMERAS:
            image_path=path/streams[key][indices[key][index]]['path']
            image=cv2.imread(str(image_path))
            if image is None or image.shape!=(480,640,3):
                raise ValueError('Missing/invalid native camera image: '+str(image_path))
            buffer[f'observation.images.{key}'].append(str(image_path.resolve()))
    dataset.save_episode(episode_data=buffer)


def collection_policy(paths,expected=None):
    policies={json.loads((path/'metadata.json').read_text()).get('gripper_policy','legacy_open_transit_v1')
              for path in paths}
    if len(policies)!=1:raise ValueError('Do not mix different gripper-policy versions in one dataset')
    policy=next(iter(policies))
    if expected is not None and policy!=expected:raise ValueError('Unexpected collection gripper policy: '+str(policy))
    return policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='Session episodes directory')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--episode-list',type=Path,help='JSON array of episode directories within --source')
    parser.add_argument('--audit-only', action='store_true')
    parser.add_argument('--image-writer-threads',type=int,default=4)
    parser.add_argument('--original-jpeg',action='store_true',help='Embed original image bytes via public pre-built episode API; no PNG transcoding')
    parser.add_argument('--gripper-policy',choices=[COLLECTION_POLICY,'legacy_open_transit_v1'],
                        help='Reject a mismatched or mixed collection before exporting')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    paths = []
    skipped = []
    if args.episode_list:
        selected=[Path(p).resolve() for p in json.loads(args.episode_list.read_text())]
        if len(set(selected))!=len(selected) or any(not p.is_relative_to(args.source.resolve()) for p in selected):
            raise ValueError('Episode selection must be unique and inside source')
        files=[p/'metadata.json' for p in selected]
    else:files=sorted(args.source.glob('episode_*/metadata.json'))
    for file in files:
        meta = json.loads(file.read_text())
        if (meta.get('review_status') == 'approved' and meta.get('operator_marked_success') is True
                and not meta.get('diagnostic_only')
                and meta.get('automatic_task_contract_passed') is True
                and meta.get('data_quality_passed') is True and meta.get('terminal_label') == 'returned_zero'):
            paths.append(file.parent)
        else:
            skipped.append(file.parent.name)
    if not paths:
        raise ValueError('No approved, successful, quality-passed episodes')
    if args.episode_list and skipped:raise ValueError('Explicit selection contains unapproved episodes: '+str(skipped))
    policy=collection_policy(paths,args.gripper_policy)
    prepared = []
    for path in paths:
        audit = audit_episode(path)
        if not audit['passed']:
            raise ValueError(f'Episode failed fresh data audit: {path}: {audit}')
        streams, q, indices, report = prepare(path, 30)
        if policy==COLLECTION_POLICY:
            for i in (0,-1):
                require_closed_transit(float(q['joint_observation'][indices['joint_observation'][i]][7]),
                                       float(q['demonstration_action'][indices['demonstration_action'][i]][7]))
        for key in CAMERAS:
            if report['alignment'][key]['max_age_ms'] > 150:
                raise ValueError('Camera alignment age exceeds 150 ms: '+str(path))
        report['quality'] = audit
        if policy == COLLECTION_POLICY:
            from .training_contract import audit_action_contract
            report['action_contract_audit'] = audit_action_contract(path)
            from .closed_training import phase_coverage
            report['phase_coverage']=phase_coverage(path,report)
        report['source_episode_path']=str(path.resolve())
        report['collection_tag']=json.loads((path/'metadata.json').read_text()).get('collection_tag')
        prepared.append((streams, q, indices, report))
    report = dict(fps=30, image_size_wh=[640, 480], joint_order=ORDER,gripper_policy=policy,
        image_storage='original JPEG bytes embedded in Parquet' if args.original_jpeg else 'lossless PNG of decoded RGB embedded in Parquet',
        action_source='JTC output.positions; inactive left finger frozen hold label',
        timestamp_alignment='Latest source timestamp at or before each 30 Hz host-monotonic grid timestamp; see source metadata for source/receive time basis',
        skipped_episodes=skipped, episodes=[item[3] for item in prepared],
        train_validation_split='Split by whole physical episode, never by individual frames',
        task='Pick up the blue cube with the right arm, place it into the black box, release it, and return to mechanical zero.')
    report_path = args.output.with_name(args.output.name+'_conversion_report.json')
    if policy == COLLECTION_POLICY:
        report['action_contract'] = prepared[0][3]['action_contract_audit']['contract']
    if args.audit_only:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(report_path, report)
        print(json.dumps(dict(episodes=len(paths), skipped=skipped, report=str(report_path)), indent=2))
        return
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    features = {'observation.state': dict(dtype='float32', shape=(16,), names=ORDER),
                'action': dict(dtype='float32', shape=(16,), names=ORDER)}
    features.update({f'observation.images.{key}': dict(dtype='image', shape=(480, 640, 3),
                     names=['height', 'width', 'channels']) for key in CAMERAS})
    dataset = LeRobotDataset.create(repo_id='local/'+args.output.name, root=args.output, fps=30,
        features=features, robot_type='openarm', use_videos=False, image_writer_threads=0 if args.original_jpeg else args.image_writer_threads)
    for path, (streams, q, indices, entry) in zip(paths, prepared):
        if args.original_jpeg:
            save_original_episode(dataset,path,(streams,q,indices,entry),report['task'])
            print('SAVED '+path.parent.parent.name+'/'+path.name+' (original JPEG embedded)',flush=True)
            continue
        for index in range(entry['frame_count']):
            frame = {'observation.state': q['joint_observation'][indices['joint_observation'][index]].copy(),
                     'action': q['demonstration_action'][indices['demonstration_action'][index]].copy(),
                     'task': report['task']}
            for key in CAMERAS:
                image = cv2.imread(str(path/streams[key][indices[key][index]]['path']))
                if image is None or image.shape != (480, 640, 3):
                    raise ValueError('Missing/invalid native camera image: '+str(path))
                frame[f'observation.images.{key}'] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            dataset.add_frame(frame)
        dataset.save_episode()
        print('SAVED '+path.parent.parent.name+'/'+path.name, flush=True)
    dataset.finalize()
    write_json(args.output/'meta/visual_expert_manifest.json', report)
    write_json(report_path, report)
    print(f'Exported {len(paths)} episodes to {args.output}')


if __name__ == '__main__':
    main()
