"""File-lock and fake-driver tests: no CAN sockets or hardware operations."""
import functools
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
NODE_SRC = ROOT / "official/dora-openarm-evaluation/nodes/dora-openarm/src"
sys.path.insert(0, str(NODE_SRC))
from dora_openarm.can_ownership import CanOwnership, reject_existing_receivers

HEADER = Path("/home/lushilin/openarm_ws/src/openarm_ros2/openarm_hardware/include")


@pytest.fixture(scope="session")
def cpp_owner(tmp_path_factory):
    directory = tmp_path_factory.mktemp("cpp-owner")
    source = directory / "owner.cpp"
    source.write_text(r'''
#define OPENARM_CAN_OWNERSHIP_TESTING
#include "openarm_hardware/can_ownership.hpp"
#include <iostream>
int main(int argc, char** argv) {
  if (argc != 3) return 2;
  try {
    openarm_hardware::CanOwnership owner(argv[2], "test-cpp", argv[1]);
    std::cout << "owned" << std::endl;
    std::string input;
    std::getline(std::cin, input);
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return 3;
  }
}
''')
    executable = directory / "owner"
    subprocess.run(["g++", "-std=c++17", "-Wall", "-Wextra", "-Werror", "-I", str(HEADER),
                    str(source), "-o", str(executable)], check=True, capture_output=True, text=True)
    return executable


def test_second_python_writer_refused_without_overwriting_owner(tmp_path):
    with CanOwnership("can0", "first", _test_root=tmp_path) as owner:
        before = owner.path.read_text()
        with pytest.raises(RuntimeError, match='"backend": "first"'):
            CanOwnership("can0", "second", _test_root=tmp_path)
        assert owner.path.read_text() == before
        assert json.loads(before)["pid"] == os.getpid()
        assert not os.get_inheritable(owner._fd)
    assert owner.path.exists()  # Persistent inode; stale metadata is not a lock.
    with CanOwnership("can0", "third", _test_root=tmp_path):
        pass


def test_independent_interfaces_can_share_one_ros_process(tmp_path):
    with CanOwnership("can0", "right", _test_root=tmp_path), \
            CanOwnership("can1", "left", _test_root=tmp_path):
        assert len(list(tmp_path.glob("*.lock"))) == 2


def test_cpp_refuses_python_owner_before_any_hardware(tmp_path, cpp_owner):
    with CanOwnership("can0", "test-python", _test_root=tmp_path):
        result = subprocess.run([str(cpp_owner), str(tmp_path), "can0"], input="\n",
                                text=True, capture_output=True)
        assert result.returncode == 3
        assert "test-python" in result.stderr and "owned\n" not in result.stdout


def test_python_and_second_cpp_refuse_cpp_owner(tmp_path, cpp_owner):
    process = subprocess.Popen([str(cpp_owner), str(tmp_path), "can0"],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        assert process.stdout.readline() == "owned\n"
        with pytest.raises(RuntimeError, match="test-cpp"):
            CanOwnership("can0", "python", _test_root=tmp_path)
        second = subprocess.run([str(cpp_owner), str(tmp_path), "can0"], input="\n",
                                capture_output=True, text=True)
        assert second.returncode == 3
        # Other arm remains independently lockable.
        with CanOwnership("can1", "other-arm", _test_root=tmp_path):
            pass
    finally:
        process.communicate("\n", timeout=5)
    with CanOwnership("can0", "successor", _test_root=tmp_path):
        pass


def test_process_crash_releases_kernel_lock(tmp_path):
    code = "from dora_openarm.can_ownership import CanOwnership; import os,sys; " \
           "owner=CanOwnership('can0','crashed',_test_root=sys.argv[1]); os._exit(17)"
    env = dict(os.environ, PYTHONPATH=str(NODE_SRC))
    result = subprocess.run([sys.executable, "-c", code, str(tmp_path)], env=env)
    assert result.returncode == 17
    with CanOwnership("can0", "after-crash", _test_root=tmp_path):
        pass


@pytest.mark.parametrize("name", ["", "../can0", "can0/one", ".", "..", "x" * 16])
def test_invalid_interface_rejected_before_file_creation(tmp_path, name):
    with pytest.raises(ValueError):
        CanOwnership(name, "test", _test_root=tmp_path)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("cpp", [False, True])
def test_symlink_and_unsafe_directory_fail_closed(tmp_path, cpp_owner, cpp):
    root = tmp_path / "locks"
    root.mkdir()
    target = tmp_path / "target"
    target.write_text("untouched")
    (root / "can0.lock").symlink_to(target)
    def rejected():
        if cpp:
            assert subprocess.run([str(cpp_owner), str(root), "can0"], input="\n",
                                  text=True, capture_output=True).returncode == 3
        else:
            with pytest.raises((OSError, RuntimeError)):
                CanOwnership("can0", "test", _test_root=root)
    rejected()
    assert target.read_text() == "untouched"
    (root / "can0.lock").unlink()
    root.chmod(0o777)
    rejected()
    assert not list(root.iterdir())


def _fake_driver(monkeypatch, tmp_path):
    from dora_openarm import feedback_driver
    monkeypatch.setattr(feedback_driver, "require_paced_can", lambda: None)
    monkeypatch.setattr(feedback_driver, "CanOwnership", functools.partial(CanOwnership, _test_root=tmp_path))
    class Config:
        def get_can_interface(self, side):
            return "can0" if side == "right_arm" else "can1"
    return feedback_driver, Config()


def test_dora_constructor_cannot_run_under_ros_lock(monkeypatch, tmp_path):
    driver, config = _fake_driver(monkeypatch, tmp_path)
    entered = []
    monkeypatch.setattr(driver.BaseSingleArmDriver, "__init__", lambda *a, **k: entered.append(True))
    with CanOwnership("can0", "ros2_control:right", _test_root=tmp_path):
        with pytest.raises(RuntimeError, match="ros2_control:right"):
            driver.SingleArmDriver("right_arm", config)
    assert entered == []


def test_dora_stop_keeps_lock_until_disable_and_socket_destruction(monkeypatch, tmp_path):
    driver, config = _fake_driver(monkeypatch, tmp_path)
    events = []
    def assert_locked():
        with pytest.raises(RuntimeError):
            CanOwnership("can0", "competitor", _test_root=tmp_path)
    class FakeSocket:
        def __del__(self):
            assert_locked()
            events.append("socket-closed-under-lock")
    def initialize(self, *_):
        assert_locked()
        events.append("constructor-under-lock")
        self.openarm = FakeSocket()
    def disable(self):
        assert_locked()
        events.append("disable-under-lock")
    monkeypatch.setattr(driver.BaseSingleArmDriver, "__init__", initialize)
    monkeypatch.setattr(driver.BaseSingleArmDriver, "stop", disable)
    arm = driver.SingleArmDriver("right_arm", config)
    arm.stop()
    with CanOwnership("can0", "successor", _test_root=tmp_path):
        pass
    assert events == ["constructor-under-lock", "disable-under-lock", "socket-closed-under-lock"]
    # Existing ordinary START can replace a stopped session with no self-lockout.
    replacement = driver.SingleArmDriver("right_arm", config)
    replacement.stop()


def test_constructor_failure_closes_socket_and_releases_lock(monkeypatch, tmp_path):
    driver, config = _fake_driver(monkeypatch, tmp_path)
    def broken(self, *_):
        self.openarm = object()
        raise RuntimeError("fake SDK initialization failure")
    monkeypatch.setattr(driver.BaseSingleArmDriver, "__init__", broken)
    with pytest.raises(RuntimeError, match="fake SDK"):
        driver.SingleArmDriver("right_arm", config)
    with CanOwnership("can0", "successor", _test_root=tmp_path):
        pass


def test_stop_failure_does_not_admit_another_writer(monkeypatch, tmp_path):
    driver, config = _fake_driver(monkeypatch, tmp_path)
    monkeypatch.setattr(driver.BaseSingleArmDriver, "__init__", lambda *a, **k: None)
    def broken(self):
        raise RuntimeError("fake disable failure")
    monkeypatch.setattr(driver.BaseSingleArmDriver, "stop", broken)
    arm = driver.SingleArmDriver("right_arm", config)
    with pytest.raises(RuntimeError, match="disable failure"):
        arm.stop()
    with pytest.raises(RuntimeError):
        CanOwnership("can0", "competitor", _test_root=tmp_path)
    arm._close_transport()  # Test cleanup only; no live driver exists.


def test_parameter_probe_does_not_create_socket_if_owned(monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location("probe_for_ownership_test", ROOT / "scripts/probe_openarm_can.py")
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    monkeypatch.setattr(probe, "CanOwnership", functools.partial(CanOwnership, _test_root=tmp_path))
    monkeypatch.setattr(probe, "snapshot", lambda _: {})
    monkeypatch.setattr(probe, "check_link", lambda _: None)
    sockets = []
    monkeypatch.setattr(probe.socket, "socket", lambda *a, **k: sockets.append(True))
    with CanOwnership("can0", "ros-owner", _test_root=tmp_path):
        result = probe.probe("can0", [1])
    assert not result["ok"] and "ros-owner" in result["error"]
    assert sockets == []


@pytest.mark.parametrize("device", ["can0", "(any)", "any"])
@pytest.mark.parametrize("cpp", [False, True])
def test_legacy_receiver_blocks_even_when_no_lock_owner_exists(tmp_path, cpp_owner, device, cpp):
    registry = tmp_path / "rcvlist_all"
    registry.write_text(f"device can_id can_mask function userdata matches ident\n {device} 000 00000000 0 0 17 raw\n")
    if cpp:
        result = subprocess.run([str(cpp_owner), str(tmp_path), "can0"], input="\n",
                                text=True, capture_output=True)
        assert result.returncode == 3 and "unpatched writer" in result.stderr
        assert "owned\n" not in result.stdout
    else:
        with pytest.raises(RuntimeError, match="unpatched writer"):
            CanOwnership("can0", "new-writer", _test_root=tmp_path)
    # Refusal closes its lock descriptor; subsequent safe acquisition works.
    registry.write_text("(any: no entry)\n(can0: no entry)\n")
    with CanOwnership("can0", "after-legacy-exit", _test_root=tmp_path):
        pass


@pytest.mark.parametrize("cpp", [False, True])
def test_other_bus_registration_allows_bimanual_initialization(tmp_path, cpp_owner, cpp):
    (tmp_path / "rcvlist_all").write_text(
        "receive list 'rx_all':\n(any: no entry)\n"
        "device can_id can_mask function userdata matches ident\n"
        "can0 000 00000000 0 0 17 raw\n(can1: no entry)\n")
    if cpp:
        result = subprocess.run([str(cpp_owner), str(tmp_path), "can1"], input="\n",
                                text=True, capture_output=True)
        assert result.returncode == 0 and result.stdout == "owned\n"
    else:
        with CanOwnership("can1", "left", _test_root=tmp_path):
            pass


def test_dora_legacy_refusal_precedes_base_constructor(monkeypatch, tmp_path):
    driver, config = _fake_driver(monkeypatch, tmp_path)
    (tmp_path / "rcvlist_sff").write_text("can0 001 000007ff 0 0 12 raw\n")
    entered = []
    monkeypatch.setattr(driver.BaseSingleArmDriver, "__init__", lambda *a, **k: entered.append(True))
    with pytest.raises(RuntimeError, match="unpatched writer"):
        driver.SingleArmDriver("right_arm", config)
    assert entered == []


def test_missing_receiver_registry_fails_closed(tmp_path):
    with pytest.raises(RuntimeError, match="registry unavailable"):
        reject_existing_receivers("can0", tmp_path)
    with pytest.raises(FileNotFoundError):
        reject_existing_receivers("can0", tmp_path / "missing")
