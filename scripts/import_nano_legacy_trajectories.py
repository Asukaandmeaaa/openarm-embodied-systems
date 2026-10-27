#!/usr/bin/env python3
"""Copy missing Nano trajectory JSONs and verify against old episode hashes.
Operator runs this in a terminal for SSH authentication. No ROS/device access.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile

WS=Path('/home/lushilin/openarm_vla_ws')
REMOTE_DIR='/home/ubuntu/openarm_kinesthetic_data/collections/trajectories'


def expected_sources():
    expected={}
    for path in sorted((WS/'openarm_kinesthetic_data/collections/episodes_pending_review').glob('*/metadata.json')):
        value=json.loads(path.read_text())
        trial=value.get('source_trajectory_trial_id','')
        if not trial.startswith('20261008_'):
            continue
        digest=value.get('trajectory_sha256','')
        if not re.fullmatch(r'20261008_\d{6}_\d{3}',trial) or not re.fullmatch(r'[a-f0-9]{64}',digest):
            raise ValueError(f'invalid historical source identity in {path}')
        if trial in expected and expected[trial]!=digest:
            raise ValueError(f'conflicting historical hashes: {trial}')
        expected[trial]=digest
    if len(expected)!=12:
        raise ValueError(f'expected 12 historical source identities; found {len(expected)}')
    return dict(sorted(expected.items()))


def verify_files(directory,expected):
    errors=[]
    for trial,digest in expected.items():
        path=directory/(trial+'.json')
        if not path.is_file():errors.append(f'{trial}: missing')
        elif hashlib.sha256(path.read_bytes()).hexdigest()!=digest:errors.append(f'{trial}: SHA256 mismatch')
    if errors:raise ValueError('\n'.join(errors))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host',default='192.168.3.112')
    parser.add_argument('--verify-only',action='store_true')
    args=parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9.-]*',args.host):
        parser.error('host must be an IP address or hostname')
    expected=expected_sources()
    archive=WS/'openarm_kinesthetic_data/imports/nano_legacy_20261016'
    dest=archive/'trajectories'
    if args.verify_only:
        verify_files(dest,expected)
        print(f'PASS: {len(expected)} archived source hashes match historical episodes')
        return
    archive.mkdir(parents=True,exist_ok=True)
    staging=Path(tempfile.mkdtemp(prefix='incoming_',dir=archive))
    print(f'Copying trajectory JSONs only into {staging}',flush=True)
    completed=subprocess.run(['scp','-p','-o','ConnectTimeout=10',
                              f'ubuntu@{args.host}:{REMOTE_DIR}/20261008_*.json',str(staging)])
    if completed.returncode:
        raise SystemExit('Copy failed. Check Nano hostname -I, SSH, and password. Staging retained; original data unchanged.')
    verify_files(staging,expected)
    # Reject any conflicting existing destination before committing files.
    for trial,digest in expected.items():
        path=dest/(trial+'.json')
        if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            raise SystemExit(f'Existing archive differs; refusing to overwrite: {path}')
    dest.mkdir(exist_ok=True)
    for trial in expected:
        path=dest/(trial+'.json')
        if not path.exists():
            with path.open('xb') as f:f.write((staging/path.name).read_bytes())
    verify_files(dest,expected)
    rows=[dict(trial_id=trial,path=str(dest/(trial+'.json')),sha256=digest,
               remote_path=f'{REMOTE_DIR}/{trial}.json') for trial,digest in expected.items()]
    (archive/'source_manifest.json').write_text(json.dumps({'host':args.host,'verified_against':'historical episode metadata SHA256','sources':rows},indent=2)+'\n')
    sources=sorted((WS/'openarm_kinesthetic_data/collections/trajectories').glob('*.json'))+[dest/(trial+'.json') for trial in expected]
    (archive/'combined_source_files.txt').write_text(''.join(str(p.resolve())+'\n' for p in sources))
    print(f'PASS: imported {len(expected)} source trajectories; all SHA256 match historical episodes')
    print(f'Combined source list ({len(sources)} files): {archive / "combined_source_files.txt"}')

if __name__=='__main__':main()
