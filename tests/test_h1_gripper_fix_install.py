"""Offline candidate installation tests; no live processes are stopped."""
import json

import pytest

from scripts.control_ownership import h1_gripper_fix as fix


def test_running_backend_blocks_install_before_any_files_are_written(monkeypatch, tmp_path):
    monkeypatch.setattr(fix, 'ROOT', tmp_path)
    monkeypatch.setattr(fix, 'control_processes', lambda: [1234])
    with pytest.raises(ValueError, match='Stop H1 and gateway'):
        fix.install()
    assert not list(tmp_path.iterdir())


@pytest.fixture
def installation(monkeypatch, tmp_path):
    (tmp_path/'results').mkdir()
    candidate = tmp_path/'candidate'; candidate.mkdir()
    library = candidate/fix.LIBRARY; library.parent.mkdir(parents=True)
    library.write_bytes(b'tested candidate library')
    target = tmp_path/'overlay/libopenarm_hardware.so'; target.parent.mkdir()
    target.write_bytes(b'original installed library')
    report = {'candidate_sha256': fix.digest(library)}
    (candidate/'candidate.json').write_text(json.dumps(report))
    monkeypatch.setattr(fix, 'ROOT', tmp_path)
    monkeypatch.setattr(fix, 'CANDIDATE', candidate)
    monkeypatch.setattr(fix, 'TARGET', target)
    monkeypatch.setattr(fix, 'candidate_report', lambda: report)
    monkeypatch.setattr(fix, 'control_processes', lambda: [])
    return tmp_path, target, library


def test_install_retains_original_and_atomically_replaces_library(installation):
    root, target, candidate = installation
    fix.install()
    assert target.read_bytes() == candidate.read_bytes()
    backup, = (root/'results').glob('h1_gripper_before_*')
    assert (backup/'libopenarm_hardware.so').read_bytes() == b'original installed library'
    report = json.loads((backup/'install.json').read_text())
    assert not report['hardware_started']
    assert report['installed_sha256'] == fix.digest(candidate)


def test_changed_candidate_report_blocks_replacement(monkeypatch, installation):
    _, target, _ = installation
    monkeypatch.setattr(fix, 'candidate_report', lambda: {'candidate_sha256': 'changed'})
    with pytest.raises(ValueError, match='changed; rebuild and retest'):
        fix.install()
    assert target.read_bytes() == b'original installed library'


def test_new_backend_during_staging_blocks_replacement(monkeypatch, installation):
    _, target, _ = installation
    processes = iter([[], [1234]])
    monkeypatch.setattr(fix, 'control_processes', lambda: next(processes))
    with pytest.raises(ValueError, match='appeared during installation'):
        fix.install()
    assert target.read_bytes() == b'original installed library'
    assert not list(target.parent.glob('.h1-gripper-*'))


def test_process_scan_recognizes_backend_and_gateway_without_matching_shell_text(tmp_path):
    for pid, args in [('101', b'/opt/ros/jazzy/lib/controller_manager/ros2_control_node\0'),
                      ('102', b'/usr/bin/python3\0-m\0scripts.control_ownership.gateway\0'),
                      ('103', b'bash\0-c\0echo ros2_control_node\0')]:
        directory = tmp_path/pid; directory.mkdir()
        (directory/'cmdline').write_bytes(args)
    assert sorted(fix.control_processes(tmp_path)) == [101, 102]
