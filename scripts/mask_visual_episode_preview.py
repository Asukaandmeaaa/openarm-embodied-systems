#!/usr/bin/env python3
"""Full-timeline offline preview with a timestamped wrist-image mask; no motion."""
import argparse
import io
import json
import math
from pathlib import Path
import shutil
import sys

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.trim_visual_episode_preview import digest, read_rows, write_json


def mask_preview(source, output, before_close=3., after_open=5.):
    source, output = Path(source).resolve(), Path(output).resolve()
    if not source.is_dir() or output.is_relative_to(source.parent):
        raise ValueError('Preview must be outside the original episodes directory')
    if output.exists():
        raise FileExistsError(f'Preview already exists; nothing overwritten: {output}')
    if any(not math.isfinite(v) or v < 0 for v in (before_close, after_open)):
        raise ValueError('Margins must be finite and non-negative')
    meta = json.loads((source/'metadata.json').read_text())
    events = read_rows(source/'events.jsonl')
    anchors = {}
    for name in ('close', 'open'):
        selected = [r for r in events if r.get('phase') == 'timeline_event'
                    and r.get('detail', {}).get('name') == name]
        if len(selected) != 1:
            raise ValueError(f'Expected exactly one {name} event')
        anchors[name] = float(selected[0]['timestamp_s'])
    close, opened = anchors['close'], anchors['open']
    start, end = close-before_close, opened+after_open
    if not all(math.isfinite(t) for t in (start, end, close, opened)) or not close < opened:
        raise ValueError('Invalid close/open timeline')
    if start < meta['start_timestamp_s'] or end > meta['end_timestamp_s']:
        raise ValueError('The requested wrist window extends outside the recording')
    grip = read_rows(source/'target_right_gripper.jsonl')
    for name, t, sign in (('close', close, 1), ('open', opened, -1)):
        before = [r for r in grip if r['timestamp_s'] <= t]
        later = [r for r in grip if t <= r['timestamp_s'] <= t+3.]
        if not before or not later:
            raise ValueError(f'Cannot corroborate {name} gripper event')
        delta = float(later[-1]['positions'][0])-float(before[-1]['positions'][0])
        if not math.isfinite(delta) or sign*delta <= .10:
            raise ValueError(f'{name} event is not corroborated by gripper targets')
    camera = read_rows(source/'camera.jsonl')
    if {r['key'] for r in camera} != {'ceiling', 'wrist_right'}:
        raise ValueError('Expected both recorded cameras')
    width, height = meta['image_size_wh']
    if (width, height) != (640, 480):
        raise ValueError('Expected native 640x480 images')
    seen, references = set(), []
    for row in camera:
        t = float(row['timestamp_s'])
        relative = Path(row['path'])
        original = (source/relative).resolve()
        if not math.isfinite(t):
            raise ValueError('Nonfinite camera timestamp')
        if (relative.is_absolute() or '..' in relative.parts or not original.is_relative_to(source)
                or not original.is_file() or relative in seen):
            raise ValueError(f'Invalid or duplicate image path: {relative}')
        seen.add(relative)
        valid = row['key'] != 'wrist_right' or start <= t <= end
        references.append((row, relative, original, valid))
    files = [source/'metadata.json', *sorted(source.glob('*.jsonl'))]
    if (source/'quality.json').is_file():
        files.append(source/'quality.json')
    hashes = {p.name: digest(p) for p in files}
    source_image_hashes = {str(relative): digest(original)
                           for _, relative, original, _ in references}
    buffer = io.BytesIO()
    Image.new('RGB', (width, height), (0, 0, 0)).save(buffer, format='JPEG', quality=95)
    black = buffer.getvalue()
    output.mkdir(parents=True, exist_ok=False)
    # Preserve camera metadata, ALL other JSONL streams, and their bytes/order/
    # timestamps exactly. The separate sidecar does not redefine frame times.
    for file in source.glob('*.jsonl'):
        shutil.copy2(file, output/file.name)
    shutil.copy2(source/'metadata.json', output/'source_metadata.json')
    validity, counts = [], {'ceiling': {'valid': 0, 'masked': 0},
                            'wrist_right': {'valid': 0, 'masked': 0}}
    for row, relative, original, valid in references:
        destination = output/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if valid:
            shutil.copy2(original, destination)
            if digest(destination) != source_image_hashes[str(relative)]:
                raise ValueError(f'Image copy verification failed: {relative}')
        else:
            with destination.open('xb') as stream:
                stream.write(black)
        counts[row['key']]['valid' if valid else 'masked'] += 1
        validity.append(dict(timestamp_s=row['timestamp_s'], key=row['key'],
            frame_number=row['frame_number'], path=row['path'], image_valid=valid,
            image_mask=1 if valid else 0,
            reason='original_image' if valid else 'outside_wrist_close_minus3_open_plus5_window',
            source_sha256=source_image_hashes[str(relative)]))
    with (output/'camera_validity.jsonl').open('x') as stream:
        for row in validity:
            stream.write(json.dumps(row, allow_nan=False)+'\n')
    # Extra sidecars are not training observations until an explicitly adapted
    # loader consumes them. Existing dual-camera models must not use this copy.
    report = dict(source_episode=str(source), physical_operations=False,
        scope='FULL timeline; only wrist images outside the active window are black-masked',
        before_close_s=before_close, after_open_s=after_open,
        close_timestamp_s=close, open_timestamp_s=opened,
        wrist_valid_start_timestamp_s=start, wrist_valid_end_timestamp_s=end,
        wrist_valid_relative_start_s=start-meta['start_timestamp_s'],
        wrist_valid_relative_end_s=end-meta['start_timestamp_s'],
        wrist_valid_duration_s=end-start,
        start_timestamp_s=meta['start_timestamp_s'], end_timestamp_s=meta['end_timestamp_s'],
        full_duration_s=meta['end_timestamp_s']-meta['start_timestamp_s'],
        camera_counts=counts, source_sha256=hashes,
        source_image_sha256=source_image_hashes,
        camera_jsonl_byte_identical=True, other_jsonl_byte_identical=True,
        timestamps_rebased=False, frames_deleted=False,
        mask_sidecar='camera_validity.jsonl', mask_semantics='1=original image, 0=black image',
        mask_is_not_yet_a_model_input=True, inference_gate_not_implemented=True,
        existing_model_compatible=False, preview_only=True)
    preview_meta = dict(meta)
    # Do not expose active full-plan/hardware-replay pointers from a preview.
    for key in ('complete_trajectory_path', 'complete_trajectory_sha256', 'cycle_result'):
        preview_meta.pop(key, None)
    preview_meta.update(trial_id=output.name, review_status='preview_only', diagnostic_only=True,
        operator_marked_success=False, automatic_task_contract_passed=False,
        data_quality_passed=False, terminal_label='full_timeline_mask_preview',
        source_episode_path=str(source), camera_mask={
            'wrist_valid_start_timestamp_s': start, 'wrist_valid_end_timestamp_s': end,
            'sidecar': 'camera_validity.jsonl', 'masked_pixel_value': 0,
            'model_mask_input_implemented': False})
    write_json(output/'metadata.json', preview_meta)
    if not (output/'gripper_event.jsonl').exists():
        with (output/'gripper_event.jsonl').open('x') as stream:
            for name, t in anchors.items():
                stream.write(json.dumps(dict(timestamp_s=t, event=name,
                    source='derived from original events.jsonl'))+'\n')
    if any(digest(source/name) != value for name, value in hashes.items()):
        raise ValueError('Source metadata/stream changed during preview generation')
    if any(digest(source/relative) != value for relative, value in source_image_hashes.items()):
        raise ValueError('Source image changed during preview generation')
    for file in source.glob('*.jsonl'):
        if digest(output/file.name) != hashes[file.name]:
            raise ValueError('Copied stream differs from source: '+file.name)
    write_json(output/'mask_report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--before-close', type=float, default=3.)
    parser.add_argument('--after-open', type=float, default=5.)
    args = parser.parse_args()
    report = mask_preview(args.source, args.output, args.before_close, args.after_open)
    from vision_grasp_baseline.collection_io import audit_episode
    quality = audit_episode(args.output)
    write_json(args.output/'quality.json', quality)
    if not quality['passed']:
        raise ValueError('Preview timing/path audit failed; inspect quality.json')
    print(json.dumps({key: report[key] for key in (
        'source_episode', 'full_duration_s', 'wrist_valid_relative_start_s',
        'wrist_valid_relative_end_s', 'wrist_valid_duration_s', 'camera_counts',
        'camera_jsonl_byte_identical', 'other_jsonl_byte_identical', 'frames_deleted',
        'existing_model_compatible', 'physical_operations')}, indent=2))


if __name__ == '__main__':
    main()
