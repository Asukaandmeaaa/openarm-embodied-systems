#!/usr/bin/env python3
"""Copy one synchronized close/open crop for OFFLINE review; never control a robot."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def trim_preview(source, output, margin=3.):
    source, output = Path(source).resolve(), Path(output).resolve()
    if not source.is_dir() or output.is_relative_to(source.parent):
        raise ValueError('Preview must be outside the original episodes directory')
    if output.exists():
        raise FileExistsError(f'Preview already exists; nothing overwritten: {output}')
    if not math.isfinite(margin) or margin < 0:
        raise ValueError('Margin must be finite and non-negative')
    meta = json.loads((source/'metadata.json').read_text())
    events = read_rows(source/'events.jsonl')
    anchors = {}
    for name in ('close', 'open'):
        matches = [r for r in events if r.get('phase') == 'timeline_event'
                   and r.get('detail', {}).get('name') == name]
        if len(matches) != 1:
            raise ValueError(f'Expected exactly one {name} event, found {len(matches)}')
        anchors[name] = float(matches[0]['timestamp_s'])
    close, opened = anchors['close'], anchors['open']
    start, end = close-margin, opened+margin
    if not all(math.isfinite(t) for t in (start, end, close, opened)) or not close < opened:
        raise ValueError('Invalid close/open timeline')
    if start < meta['start_timestamp_s'] or end > meta['end_timestamp_s']:
        raise ValueError('The requested full margins extend outside the recording')
    grip = read_rows(source/'target_right_gripper.jsonl')
    before = lambda t: max((r for r in grip if r['timestamp_s'] <= t),
                           key=lambda r: r['timestamp_s'])['positions'][0]
    values = [float(r['positions'][0]) for r in grip]
    if not all(math.isfinite(v) for v in values):
        raise ValueError('Nonfinite gripper command')
    # Pinch gripper: more negative opens. Verify the recorded phase markers
    # against command directions, not against contact or object-success claims.
    for name, t, sign in (('close', close, 1), ('open', opened, -1)):
        later = [r['positions'][0] for r in grip if t <= r['timestamp_s'] <= t+margin]
        if not later or sign*(later[-1]-before(t)) <= .10:
            raise ValueError(f'{name} event is not corroborated by gripper targets')
    files = sorted(source.glob('*.jsonl'))
    hashes = {p.name: digest(p) for p in [source/'metadata.json', source/'quality.json', *files]
              if p.is_file()}
    selected = {}
    for file in files:
        rows = []
        with file.open('rb') as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                t = float(row['timestamp_s'])
                if not math.isfinite(t):
                    raise ValueError(f'Nonfinite timestamp: {file}')
                if start <= t <= end:
                    rows.append((row, line))
        selected[file.name] = rows
    for name in ('joint_observation.jsonl', 'demonstration_action.jsonl', 'camera.jsonl'):
        if not selected.get(name):
            raise ValueError(f'Empty cropped stream: {name}')
    camera_rows = [r for r, _ in selected['camera.jsonl']]
    if {r['key'] for r in camera_rows} != {'ceiling', 'wrist_right'}:
        raise ValueError('Expected both recorded cameras')
    image_paths = {}
    for row in camera_rows:
        relative = Path(row['path'])
        target = (source/relative).resolve()
        if relative.is_absolute() or '..' in relative.parts or not target.is_relative_to(source) or not target.is_file():
            raise ValueError(f'Invalid image path: {relative}')
        image_paths[relative] = target
    output.mkdir(parents=True, exist_ok=False)
    # Stream lines and image bytes are preserved, without resampling, smoothing,
    # timestamp rebasing, shared hard links, or source mutations.
    for name, rows in selected.items():
        with (output/name).open('xb') as stream:
            for _, line in rows:
                stream.write(line if line.endswith(b'\n') else line+b'\n')
    for relative, original in image_paths.items():
        destination = output/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, destination)
        if digest(original) != digest(destination):
            raise ValueError(f'Image copy verification failed: {relative}')
    shutil.copy2(source/'metadata.json', output/'source_metadata.json')
    crop = dict(source_episode=str(source), physical_operations=False,
                scope='synchronized middle segment of BOTH cameras, states and actions; offline preview only',
                rule='recorded close START minus margin through recorded open START plus margin',
                margin_s=margin, close_timestamp_s=close, open_timestamp_s=opened,
                start_timestamp_s=start, end_timestamp_s=end, duration_s=end-start,
                source_start_timestamp_s=meta['start_timestamp_s'],
                source_duration_s=meta['end_timestamp_s']-meta['start_timestamp_s'],
                source_relative_start_s=start-meta['start_timestamp_s'],
                source_relative_end_s=end-meta['start_timestamp_s'],
                source_sha256=hashes,
                stream_counts={name: len(rows) for name, rows in selected.items()},
                camera_counts={key: sum(r['key'] == key for r in camera_rows)
                               for key in ('ceiling', 'wrist_right')},
                timestamps='original host-monotonic and source headers preserved; NOT rebased',
                not_safe_for_physical_replay=True,
                start_and_end_are_not_mechanical_zero=True)
    preview_meta = {key: meta[key] for key in (
        'schema_version', 'timestamp_basis', 'joint_order', 'target_fps', 'image_size_wh',
        'camera_configuration', 'action_source', 'source_trajectory_trial_id', 'collection_tag')
        if key in meta}
    preview_meta.update(trial_id=output.name, start_timestamp_s=start, end_timestamp_s=end,
                        initial_state=selected['joint_observation.jsonl'][0][0]['positions'],
                        review_status='preview_only', diagnostic_only=True,
                        operator_marked_success=False, automatic_task_contract_passed=False,
                        terminal_label='cropped_middle_not_zero_to_zero', recording_error=None,
                        source_episode_path=str(source), crop=crop)
    write_json(output/'metadata.json', preview_meta)
    # The existing offline viewer accepts this small event stream.
    with (output/'gripper_event.jsonl').open('x') as stream:
        for name, t in anchors.items():
            stream.write(json.dumps(dict(timestamp_s=t, event=name,
                source='derived from original events.jsonl timeline_event'))+'\n')
    write_json(output/'crop_report.json', crop)
    if any(digest(source/name) != value for name, value in hashes.items()):
        raise ValueError('Source changed during crop; inspect before using preview')
    return crop


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--margin', type=float, default=3.)
    args = parser.parse_args()
    report = trim_preview(args.source, args.output, args.margin)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from vision_grasp_baseline.collection_io import audit_episode
    quality = audit_episode(args.output)
    write_json(args.output/'quality.json', quality)
    if not quality['passed']:
        raise ValueError('Cropped timing/path audit failed; inspect preview quality.json')
    print(json.dumps(dict(output=str(args.output.resolve()),
                          source_relative_start_s=report['source_relative_start_s'],
                          source_relative_end_s=report['source_relative_end_s'],
                          duration_s=report['duration_s'], camera_counts=report['camera_counts'],
                          timing_audit_passed=True, source_files_unchanged=True,
                          physical_operations=False), indent=2))


if __name__ == '__main__':
    main()
