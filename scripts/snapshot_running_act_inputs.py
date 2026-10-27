#!/usr/bin/env python3
"""Read existing camera shared-memory outputs and recorded feedback, never control.

Only JPEG payloads whose complete SHA256 matches the camera producer's audit log
are accepted. This does not attach a Dora node, open a camera/CAN/socket, signal
processes, or enable inference in the stopped physical dataflow.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
import pyarrow as pa

ROOT = Path(__file__).resolve().parents[1]


def recent_rows(path, size=262144):
    with Path(path).open('rb') as stream:
        length = stream.seek(0, 2)
        stream.seek(max(0, length-size))
        lines = stream.read().splitlines()
    rows = []
    for line in lines:
        try:
            rows.append(json.loads(line))
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass  # Partial first/last line in a concurrently appended log.
    return rows


def camera_snapshot(pid, log):
    proc = Path(f'/proc/{pid}')
    if b'configured_camera.py' not in (proc/'cmdline').read_bytes():
        raise ValueError('PID is not the expected existing camera producer')
    paths = {line.split()[-1] for line in (proc/'maps').read_text().splitlines()
             if '/dev/shm/shmem_' in line and '(deleted)' not in line}
    payloads = {}
    for name in paths:
        path = Path(name)
        if not str(path).startswith('/dev/shm/shmem_'):
            continue
        try:
            if path.stat().st_size > 2_000_000:
                continue
            content = path.read_bytes()
        except FileNotFoundError:
            continue
        start = content.find(b'\xff\xd8\xff')
        end = content.find(b'\xff\xd9', start)
        if start >= 0 and end >= 0:
            jpeg = content[start:end+2]
            payloads[hashlib.sha256(jpeg).hexdigest()] = (jpeg, str(path))
    matches = [row for row in recent_rows(log) if row.get('jpeg_sha256') in payloads
               and row.get('event') == 'camera_frame_published']
    if not matches:
        raise ValueError('No producer-verified JPEG in existing shared-memory slots')
    row = max(matches, key=lambda r: r['source_monotonic_ns'])
    jpeg, path = payloads[row['jpeg_sha256']]
    if len(jpeg) != row['jpeg_bytes']:
        raise ValueError('JPEG length mismatch')
    return jpeg, row, path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trial', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ceiling-pid', type=int, required=True)
    parser.add_argument('--wrist-pid', type=int, required=True)
    parser.add_argument('--samples', type=int, default=3)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    path = ROOT/'official/dora-openarm-evaluation/nodes/dora-openarm-observer/src/dora_openarm_observer/main.py'
    spec = importlib.util.spec_from_file_location('_snapshot_official_observer', path)
    observer = importlib.util.module_from_spec(spec); spec.loader.exec_module(observer)
    producer = {'ceiling': args.ceiling_pid, 'wrist_right': args.wrist_pid}
    logs = {role: next(args.trial.glob(role+'_*.jsonl')) for role in producer}
    manifest = {'physical_commands_sent': False, 'runtime_configuration_changed': False,
                'scope': 'Reconstructed shadow inputs from verified live camera payloads and passive recorded CAN feedback; not published into the physical policy dataflow.',
                'observer_source': str(path), 'observer_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'samples': []}
    with ThreadPoolExecutor(max_workers=2) as pool:
        for sample_id in range(args.samples):
            deadline = time.monotonic()+5
            while True:
                try:
                    cameras = {role: camera_snapshot(pid, logs[role]) for role,pid in producer.items()}
                    feedback = [r for r in recent_rows(args.trial/'bridge.jsonl', 1_048_576)
                                if r.get('kind')=='post_stop_feedback'][-1]
                    can = feedback['can']
                    stamps = [int(c[1]['source_monotonic_ns']) for c in cameras.values()] + [int(v) for v in can['source_monotonic_ns']]
                    now = time.monotonic_ns()
                    max_age, skew = now-min(stamps), max(stamps)-min(stamps)
                    if min(stamps)>now or max_age>200_000_000 or skew>100_000_000:
                        raise ValueError(f'Snapshot timing rejected: age={max_age/1e6:.1f}ms skew={skew/1e6:.1f}ms')
                    q = np.array(can['positions_rad'], dtype=np.float32)
                    assert q.shape==(16,) and np.isfinite(q).all()
                    break
                except (ValueError, IndexError) as error:
                    if time.monotonic()>deadline:
                        raise RuntimeError('No fresh synchronized snapshot') from error
                    time.sleep(.02)
            observation = {'id': sample_id, 'arm_right': {'value': pa.array(q[:8])}, 'arm_left': {'value': pa.array(q[8:])}}
            evidence = {}
            for role,(jpeg,row,shared_path) in cameras.items():
                (args.output/f'{sample_id:02d}_{role}.jpg').write_bytes(jpeg)
                observation['camera_'+role] = {'value': pa.array(np.frombuffer(jpeg,dtype=np.uint8))}
                evidence[role] = {'producer_log': str(logs[role]), 'producer_record': row, 'shared_memory_path': shared_path}
            metadata = {'observation_monotonic_ns': now, 'source_age_ns': max_age, 'source_skew_ns': skew}
            output = observer._build_output(observation, None, 'pick up cube', metadata, pool)
            batch = pa.RecordBatch.from_struct_array(output)
            target = args.output/f'observation_{sample_id:02d}.arrow'
            with pa.OSFile(str(target), 'wb') as stream:
                with pa.ipc.new_file(stream, batch.schema) as writer:
                    writer.write_batch(batch)
            manifest['samples'].append({'path': str(target), 'sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
                                        'metadata': metadata, 'feedback': feedback, 'camera_evidence': evidence,
                                        'snapshot_wall_ns': time.time_ns()})
            print(f'SNAPSHOT {sample_id+1}/{args.samples} source_age_ms={max_age/1e6:.1f} source_skew_ms={skew/1e6:.1f}',flush=True)
            if sample_id+1<args.samples:
                time.sleep(1)
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('READ_ONLY_SNAPSHOT_COMPLETE',flush=True)


if __name__=='__main__':
    main()
