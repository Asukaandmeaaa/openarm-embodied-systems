"""Offline permissions/launch contract checks: never start a ROS process."""
import xml.etree.ElementTree as ET
import json
import subprocess
from types import SimpleNamespace
from fnmatch import fnmatchcase
from pathlib import Path

import pytest

from scripts.control_ownership import security, secure_launch


def allowed(role, kind, operation, endpoint):
    tree = ET.parse(security.POLICY)
    profile = tree.find(f"enclaves/enclave[@path='/openarm/{role}']/profiles/profile")
    matching = [group.attrib[operation] for group in profile.findall(kind)
                if operation in group.attrib and any(fnmatchcase(endpoint, item.text) for item in group)]
    return 'ALLOW' in matching and 'DENY' not in matching


@pytest.mark.parametrize('role', ['h2', 'moveit', 'observer', 'bootstrap'])
@pytest.mark.parametrize('controller', secure_launch.CONTROLLERS)
def test_other_sources_cannot_publish_raw_or_call_actions(role, controller):
    assert not allowed(role, 'topics', 'publish', f'/{controller}/joint_trajectory')
    assert not allowed(role, 'actions', 'call', f'/{controller}/follow_joint_trajectory')


@pytest.mark.parametrize('role', ['h2', 'moveit', 'observer', 'gateway'])
@pytest.mark.parametrize('endpoint', ['/controller_manager/switch_controller',
                                     '/controller_manager/set_hardware_component_state',
                                     '/openarm_right_hardware/set_passive',
                                     '/openarm_left_hardware/set_passive'])
def test_sources_cannot_change_hardware_or_controller_lifecycle(role, endpoint):
    assert not allowed(role, 'services', 'request', endpoint)


@pytest.mark.parametrize('role', ['h2', 'moveit', 'observer', 'bootstrap', 'gateway'])
def test_no_source_can_publish_passive_heartbeat(role):
    for side in ('left', 'right'):
        assert not allowed(role, 'topics', 'publish', f'/openarm_{side}_hardware/passive_heartbeat')


def test_authorized_routes_and_bootstrap_are_distinct():
    for controller in secure_launch.CONTROLLERS:
        assert allowed('gateway', 'topics', 'publish', f'/{controller}/joint_trajectory')
        assert allowed('gateway', 'actions', 'call', f'/{controller}/follow_joint_trajectory')
        assert allowed('moveit', 'actions', 'call', f'/authorized/{controller}/follow_joint_trajectory')
    assert allowed('h2', 'actions', 'call', '/execute_trajectory')
    assert allowed('h2', 'actions', 'call', '/authorized/right_gripper_controller/follow_joint_trajectory')
    assert not allowed('h2', 'actions', 'call', '/authorized/right_joint_trajectory_controller/follow_joint_trajectory')
    assert allowed('bootstrap', 'services', 'request', '/controller_manager/switch_controller')


def test_physical_launch_arguments_retain_validated_controller_contract():
    assert secure_launch.VALIDATED_ARGUMENTS['gravity_compensation'] == 'true'
    assert secure_launch.VALIDATED_ARGUMENTS['gravity_scale'] == '1.0'
    assert secure_launch.VALIDATED_ARGUMENTS['robot_controller'] == 'joint_trajectory_controller'
    for name in secure_launch.CONTROLLERS:
        assert (f'/{name}/follow_joint_trajectory',
                f'/authorized/{name}/follow_joint_trajectory') in secure_launch.moveit_remaps()


def test_commissioning_bootstrap_skips_known_left_gripper_fault_only():
    assert secure_launch.bootstrap_arguments(['left_gripper_controller', 'right_gripper_controller', '-c', '/controller_manager']) == ['right_gripper_controller', '-c', '/controller_manager']
    arms = ['left_joint_trajectory_controller', 'right_joint_trajectory_controller', '-c', '/controller_manager']
    assert secure_launch.bootstrap_arguments(arms) == arms


def test_owner_scan_uses_actual_argv_not_shell_source(tmp_path):
    task = tmp_path / '120'; task.mkdir()
    task.joinpath('cmdline').write_bytes(b'bash\0-c\0echo ros2_control_node\0')
    assert secure_launch.conflicting_processes(tmp_path) == []
    task.joinpath('cmdline').write_bytes(b'/opt/ros/jazzy/lib/controller_manager/ros2_control_node\0')
    assert len(secure_launch.conflicting_processes(tmp_path)) == 1


def test_can_preflight_rejects_any_existing_can_owner(tmp_path):
    tmp_path.joinpath('rcvlist_all').write_text('device can_id can_mask function userdata matches ident\n  can0 000 000 0 0 0 raw\n')
    assert secure_launch.existing_can_receivers(tmp_path)[0]['file'] == 'rcvlist_all'


def test_missing_or_unknown_keystore_never_falls_back(tmp_path):
    with pytest.raises(FileNotFoundError):
        security.environment(tmp_path)


def test_empty_can_queues_allow_historical_drops_without_reset(monkeypatch):
    calls = []
    def read_stats(command, **kwargs):
        calls.append(command)
        assert kwargs['timeout'] == 5.
        return SimpleNamespace(stdout=json.dumps([
            {'backlog': 0, 'qlen': 0, 'drops': 6285703, 'requeues': 784587}]))
    monkeypatch.setattr(secure_launch.subprocess, 'run', read_stats)
    secure_launch.verify_empty_can_tx_queues()
    assert calls == [['tc', '-j', '-s', 'qdisc', 'show', 'dev', interface]
                     for interface in ('can0', 'can1')]


@pytest.mark.parametrize('blocked_interface', ['can0', 'can1'])
@pytest.mark.parametrize('backlog,qlen', [(792, 11), (72, 0), (0, 1)])
def test_pending_can_commands_block_startup(monkeypatch, blocked_interface, backlog, qlen):
    def read_stats(command, **kwargs):
        occupied = command[-1] == blocked_interface
        return SimpleNamespace(stdout=json.dumps([
            {'backlog': backlog if occupied else 0, 'qlen': qlen if occupied else 0}]))
    monkeypatch.setattr(secure_launch.subprocess, 'run', read_stats)
    with pytest.raises(ValueError, match='pending CAN TX frames on ' + blocked_interface):
        secure_launch.verify_empty_can_tx_queues()


@pytest.mark.parametrize('output', ['[]', '{}', 'invalid JSON', '[{}]',
                                  '[{"backlog":0,"qlen":null}]',
                                  '[{"backlog":-1,"qlen":0}]'])
def test_unavailable_queue_evidence_blocks_startup(monkeypatch, output):
    monkeypatch.setattr(secure_launch.subprocess, 'run',
                        lambda *args, **kwargs: SimpleNamespace(stdout=output))
    with pytest.raises(ValueError, match='cannot verify CAN TX queue'):
        secure_launch.verify_empty_can_tx_queues()


@pytest.mark.parametrize('error', [FileNotFoundError('tc'),
                                 subprocess.TimeoutExpired('tc', 5.),
                                 subprocess.CalledProcessError(1, 'tc')])
def test_queue_inspection_failure_is_fail_closed(monkeypatch, error):
    def failed(*args, **kwargs):
        raise error
    monkeypatch.setattr(secure_launch.subprocess, 'run', failed)
    with pytest.raises(ValueError, match='cannot verify CAN TX queue'):
        secure_launch.verify_empty_can_tx_queues()


def test_physical_preflight_propagates_pending_queue_block(monkeypatch):
    monkeypatch.setattr(secure_launch, 'environment', lambda _: {})
    monkeypatch.setattr(secure_launch, 'verify_overlay', lambda: 'mock-library')
    monkeypatch.setattr(secure_launch, 'conflicting_processes', lambda: [])
    monkeypatch.setattr(secure_launch, 'existing_can_receivers', lambda: [])
    monkeypatch.setattr(secure_launch.Path, 'is_dir', lambda _: True)
    monkeypatch.setattr(secure_launch.os, 'access', lambda *args: True)
    def blocked():
        raise ValueError('pending CAN TX frames on can0')
    monkeypatch.setattr(secure_launch, 'verify_empty_can_tx_queues', blocked)
    with pytest.raises(ValueError, match='pending CAN TX frames'):
        secure_launch.preflight('unused-test-keystore')


@pytest.mark.parametrize('domain', [0, 97, 117, 149, 233])
def test_mock_stack_rejects_non_test_domains_before_creating_anything(tmp_path, monkeypatch, domain):
    from scripts.control_ownership import test_secure_stack
    target = tmp_path / 'never_created'
    monkeypatch.setattr('sys.argv', ['mock-test', '--output', str(target), '--domain', str(domain)])
    with pytest.raises(SystemExit) as error:
        test_secure_stack.main()
    assert error.value.code == 2
    assert not target.exists()
