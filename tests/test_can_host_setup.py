"""Cold-boot setup regressions; all paths and module loading are fake."""
import subprocess

import pytest

from scripts.control_ownership import host_setup, secure_launch


def test_new_lock_directory_and_existing_lock_evidence_are_preserved(tmp_path):
    root = tmp_path/'locks'
    host_setup.ensure_lock_directory(root)
    lock = root/'can0.lock'
    lock.write_text('existing ownership evidence')
    host_setup.ensure_lock_directory(root)
    assert lock.read_text() == 'existing ownership evidence'


def test_unsafe_directory_and_symlink_are_rejected(tmp_path):
    root = tmp_path/'locks'; root.mkdir(mode=0o777); root.chmod(0o777)
    with pytest.raises(ValueError, match='unsafe'):
        host_setup.ensure_lock_directory(root)
    link = tmp_path/'link'; link.symlink_to(root)
    with pytest.raises(ValueError, match='unsafe'):
        host_setup.ensure_lock_directory(link)


def test_missing_registry_cannot_pass_readonly_preflight(tmp_path):
    with pytest.raises(ValueError, match='prepare-host'):
        secure_launch.existing_can_receivers(tmp_path/'missing')
    with pytest.raises(ValueError, match='prepare-host'):
        secure_launch.existing_can_receivers(tmp_path)


def test_cold_boot_loads_protocol_before_registry_verification(monkeypatch, tmp_path):
    registry = tmp_path/'can'
    monkeypatch.setattr(host_setup, 'CAN_PROC', registry)
    monkeypatch.setattr(host_setup, 'ensure_lock_directory', lambda: None)
    monkeypatch.setattr(host_setup.os, 'geteuid', lambda: 1000)
    monkeypatch.setattr(host_setup.shutil, 'which', lambda _: '/usr/sbin/modprobe')
    calls = []

    def load(command, **kwargs):
        assert kwargs == {'check': True}
        calls.append(command)
        registry.mkdir(); (registry/'rcvlist_all').write_text('(any: no entry)\n')

    monkeypatch.setattr(host_setup.subprocess, 'run', load)
    verify = host_setup.require_can_registry
    monkeypatch.setattr(host_setup, 'require_can_registry', lambda: verify(registry))
    result = host_setup.prepare_host()
    assert calls == [['sudo', '/usr/sbin/modprobe', 'can_raw']]
    assert result['host_prepared'] and not result['CAN_sockets_opened']


def test_module_load_failure_cannot_report_success(monkeypatch, tmp_path):
    monkeypatch.setattr(host_setup, 'CAN_PROC', tmp_path/'missing')
    monkeypatch.setattr(host_setup, 'ensure_lock_directory', lambda: None)
    monkeypatch.setattr(host_setup.shutil, 'which', lambda _: '/usr/sbin/modprobe')

    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(1, 'modprobe')

    monkeypatch.setattr(host_setup.subprocess, 'run', fail)
    with pytest.raises(subprocess.CalledProcessError):
        host_setup.prepare_host()
