"""Verify/install the offline-tested H1 library only with control processes stopped."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
CANDIDATE = ROOT/'.h1_gripper_candidate_20261023'
LIBRARY = Path('install/openarm_hardware/lib/libopenarm_hardware.so')
TARGET = ROOT/'.control_ownership_overlay/install/openarm_hardware/lib/libopenarm_hardware.so'
SOURCE = ROOT.parent/'openarm_ws/src/openarm_ros2/openarm_hardware'
SOURCE_FILES = ('CMakeLists.txt', 'src/openarm_simple_hardware.cpp',
                'include/openarm_hardware/openarm_simple_hardware.hpp',
                'test/test_gripper_startup_handoff.cpp')
REQUIRED_SYMBOLS=('openarm_can_ownership_protocol_version','openarm_h1_gripper_handoff_version')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def control_processes(proc=Path('/proc')):
    result = []
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            args = [x.decode(errors='replace') for x in (entry/'cmdline').read_bytes().split(b'\0') if x]
        except FileNotFoundError:
            continue
        except PermissionError:
            raise ValueError('cannot inspect process '+entry.name)
        if args and (Path(args[0]).name == 'ros2_control_node' or any(
                args[i:i+2] == ['-m', 'scripts.control_ownership.gateway'] for i in range(len(args)))):
            result.append(int(entry.name))
    return result


def candidate_report():
    symbols = subprocess.run(['nm', '-D', '--defined-only', str(CANDIDATE/LIBRARY)],
                             check=True, capture_output=True, text=True).stdout
    for symbol in REQUIRED_SYMBOLS:
        if ' '+symbol not in symbols:
            raise ValueError('candidate library lacks '+symbol)
    results = {}
    for name in ('test_gripper_contact_detector', 'test_gripper_startup_handoff'):
        path = CANDIDATE/'build/test_results/openarm_hardware'/(name+'.gtest.xml')
        tree = ET.parse(path).getroot()
        if (int(tree.get('tests', '0')) == 0 or int(tree.get('failures', '0'))
                or int(tree.get('errors', '0')) or int(tree.get('disabled', '0'))):
            raise ValueError('candidate C++ tests did not pass: '+str(path))
        results[name] = {'tests': int(tree.get('tests')), 'sha256': digest(path)}
    return {'candidate_sha256': digest(CANDIDATE/LIBRARY), 'cpp_tests': results,
            'source_sha256': {name: digest(SOURCE/name) for name in SOURCE_FILES},
            'physical_test_performed': False}


def install():
    processes = control_processes()
    if processes:
        raise ValueError('Stop H1 and gateway terminals before installing; running PIDs: '+str(processes))
    report = json.loads((CANDIDATE/'candidate.json').read_text())
    if candidate_report() != report:
        raise ValueError('candidate library/source/test results changed; rebuild and retest')
    if TARGET.is_symlink() or not TARGET.is_file():
        raise ValueError('expected a regular installed overlay library: '+str(TARGET))
    backup = Path(tempfile.mkdtemp(prefix='h1_gripper_before_', dir=ROOT/'results'))
    shutil.copy2(TARGET, backup/'libopenarm_hardware.so')
    previous_sha256 = digest(TARGET)
    descriptor, temporary = tempfile.mkstemp(prefix='.h1-gripper-', dir=TARGET.parent)
    os.close(descriptor)
    temporary = Path(temporary)
    try:
        shutil.copyfile(CANDIDATE/LIBRARY, temporary)
        temporary.chmod(0o644)
        if digest(temporary) != report['candidate_sha256']:
            raise ValueError('staged library checksum mismatch')
        if control_processes():
            raise ValueError('control process appeared during installation; refusing replacement')
        os.replace(temporary, TARGET)
    finally:
        if temporary.exists():
            temporary.unlink()
    result = {'installed': str(TARGET), 'previous_sha256': previous_sha256,
              'installed_sha256': digest(TARGET), 'backup': str(backup),
              'hardware_started': False}
    (backup/'install.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


def main():
    global CANDIDATE,REQUIRED_SYMBOLS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record-candidate', action='store_true')
    parser.add_argument('--closed-transit',action='store_true',
                        help='Select the separately built/tested collection startup candidate')
    args = parser.parse_args()
    if args.closed_transit:
        CANDIDATE=ROOT/'.closed_transit_h1_candidate_20261023'
        REQUIRED_SYMBOLS=(*REQUIRED_SYMBOLS,'openarm_closed_transit_startup_version')
    if args.record_candidate:
        report = candidate_report()
        (CANDIDATE/'candidate.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report, indent=2))
    else:
        install()


if __name__ == '__main__':
    main()
