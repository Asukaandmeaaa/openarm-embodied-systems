"""Opt-in, local-only synchronized position trajectory generation.

No ROS/CAN, intermediate waypoints/cloud API, motor enable, or dispatch here.
Ruckig's acceleration-limited mode retains the existing speed/acceleration caps.
Phase synchronization is requested; non-collinear moving states can fall back to
time synchronization. This is not a path planner or a physical stopping proof.
"""
from functools import lru_cache
import hashlib
import importlib.util
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
STREAM_MODE = "position_stream_v1"
MODES = ("legacy", "synchronized_v1", STREAM_MODE)
LIBRARY = ROOT/".act_motion_deps/ruckig-0.12.2/ruckig.cpython-312-x86_64-linux-gnu.so"
LIBRARY_SHA256 = "02c63a2986f3e326a6c09f7ba24066270cc15c093ee84fdcfebe1f30c6daefaa"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@lru_cache(maxsize=1)
def load_ruckig():
    if not LIBRARY.is_file() or digest(LIBRARY) != LIBRARY_SHA256:
        raise ValueError("pinned local Ruckig dependency missing/changed; do not fall back to legacy")
    existing = sys.modules.get("ruckig")
    if existing is not None:
        if Path(existing.__file__).resolve() != LIBRARY.resolve():
            raise ValueError("unexpected Ruckig already imported")
        return existing
    spec = importlib.util.spec_from_file_location("ruckig", LIBRARY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if module.__version__ != "0.12.2":
        raise ValueError("unexpected Ruckig version")
    sys.modules["ruckig"] = module
    return module


def contract(name):
    if name not in MODES:
        raise ValueError("unknown motion generator")
    if name == "legacy":
        return {"name": name}
    if name == STREAM_MODE:
        return {
            "name": name,
            "position": "forward bounded rolling-curve samples without extra slew",
            "velocity": "analytic derivative of the same time-aligned rolling position curve",
            "constraints": "canonical URDF/MoveIt velocity rejection; no added acceleration cap",
            "stop_behavior": "latched last-reference hold with zero qd; not a braking guarantee",
        }
    load_ruckig()
    return {
        "name": name, "library": str(LIBRARY), "library_sha256": LIBRARY_SHA256,
        "source_sha256": {str(p): digest(p) for p in
                          (Path(__file__), Path(__file__).with_name("guard.py"))},
        "source_hash_policy": "record only; source changes do not block repeated testing",
        "synchronization": "phase requested; time fallback for incompatible moving states",
        "constraints": "existing velocity/acceleration limits; no jerk limit claimed",
        "collision": "task: MuJoCo checks disabled; commissioning: measured/command sweeps; braking reserve diagnostic only",
        "braking_reserve_policy": "diagnostic only; no position or collision rejection",
        "stop_behavior": "unchanged latched last-reference hold; not a dynamic braking guarantee",
    }


def selected_name(config):
    """Read the selected backend; recorded hashes are provenance only."""
    stored = config.get("motion_generator", {"name": "legacy"})
    if not isinstance(stored, dict) or stored.get("name") not in MODES:
        raise ValueError("invalid motion generator selection")
    return stored["name"]


def from_config(config):
    """Use the selected backend without comparing saved implementation metadata.

    Source edits must not require re-preparing a trial during repeated tests.
    The current local dependency is still loaded and validated by load_ruckig.
    """
    name = selected_name(config)
    if name == "synchronized_v1":
        load_ruckig()
    return name


def attach(config, report, name):
    """Bind selection before preparation; never change a live control mode."""
    config["motion_generator"] = contract(name)
    report["motion_generator"] = config["motion_generator"]


def training_processes(proc=Path("/proc")):
    """Read-only host contention check; never stop or signal training."""
    names = {"train_act_real7030.py", "train_act_latest_real_finetune.py",
             "lerobot_train.py", "train.py"}
    found = []
    for directory in proc.iterdir():
        if not directory.name.isdigit():
            continue
        try:
            if directory.stat().st_uid != os.getuid():
                continue
            args = directory.joinpath("cmdline").read_bytes().decode(errors="replace").split("\0")
        except (OSError, ProcessLookupError):
            continue
        executable = Path(args[0]).name if args else ""
        if (executable.startswith("python") and len(args) > 1
                and (Path(args[1]).name in names
                     or (args[1] == "-m" and len(args) > 2
                         and args[2] in ("lerobot.scripts.train", "lerobot.scripts.lerobot_train")))):
            found.append(int(directory.name))
    return found


def check_live_host(name):
    if name in ("synchronized_v1", STREAM_MODE):
        processes = training_processes()
        if processes:
            raise ValueError("ACT live test refused while training is running: "
                             +str(processes)+". Wait for training to finish/checkpoint; no process was stopped.")


def braking_reserve(q, v, acceleration):
    """Synchronized constant-deceleration straight line in joint space.

    Each joint decelerates by -v/T, T=max(abs(v)/a). Therefore |deceleration|<=a.
    This defines diagnostic geometry only; it does not reject commands.
    Existing ROS fault/stop behavior does NOT execute this polynomial.
    """
    q, v = np.asarray(q, float), np.asarray(v, float)
    if (q.shape != v.shape or q.ndim != 1 or not np.isfinite(q).all()
            or not np.isfinite(v).all() or not np.isfinite(acceleration) or acceleration <= 0):
        raise ValueError("invalid braking reserve state")
    duration = float(np.max(np.abs(v))/acceleration)
    return q+v*duration/2, duration


class SynchronizedGenerator:
    def __init__(self, speed, acceleration, lower, upper, command_gripper):
        self.library = load_ruckig()
        self.count = 8 if command_gripper else 7
        self.speed = np.asarray(speed, float)[:self.count].copy()
        self.acceleration = float(acceleration)
        self.lower = np.asarray(lower, float)[:self.count].copy()
        self.upper = np.asarray(upper, float)[:self.count].copy()
        if (not np.isfinite(self.speed).all() or np.any(self.speed <= 0)
                or not np.isfinite(self.acceleration) or self.acceleration <= 0):
            raise ValueError("invalid synchronized motion limits")
        self.solver = self.library.Ruckig(self.count, .01)
        self.input = self.library.InputParameter(self.count)
        self.input.max_velocity = self.speed.tolist()
        self.input.max_acceleration = [self.acceleration]*self.count
        # max_jerk defaults to infinity: acceleration-limited mode, matching the
        # existing contract. Do not invent a hardware jerk specification.
        self.input.target_velocity = [0.]*self.count
        self.input.synchronization = self.library.Synchronization.Phase
        self.trajectory = self.library.Trajectory(self.count)
        self.diagnostics = {}
        self._cached_target = None
        self._expected_q = None
        self._expected_v = None
        self._elapsed = 0.

    def propose(self, command, velocity, target, dt):
        q, v, target = (np.asarray(x, float) for x in (command, velocity, target))
        if any(x.shape != (8,) or not np.isfinite(x).all() for x in (q, v, target)):
            raise ValueError("synchronized generator requires finite right8 vectors")
        if not np.isfinite(dt) or not 0 < dt <= .01+1e-12:
            raise ValueError("invalid synchronized control period")
        n = self.count
        reuse = (self._cached_target is not None
                 and np.array_equal(target[:n], self._cached_target)
                 and np.allclose(q[:n], self._expected_q, atol=1e-12, rtol=0.)
                 and np.allclose(v[:n], self._expected_v, atol=1e-12, rtol=0.))
        if not reuse:
            self.input.current_position = q[:n].tolist()
            self.input.current_velocity = v[:n].tolist()
            self.input.target_position = target[:n].tolist()
            self.solver.validate_input(self.input, True, True)
            result = self.solver.calculate(self.input, self.trajectory)
            if result.value < 0:
                raise ValueError("synchronized trajectory generation failed: "+str(result))
            self._elapsed = 0.
        # Retain a compatible unchanged-target trajectory instead of repeatedly
        # solving the same path and introducing numerical phase fallback.
        elapsed = min(self._elapsed+dt, self.trajectory.duration)
        extrema = self.trajectory.position_extrema
        if any(e.min < self.lower[i]-1e-10 or e.max > self.upper[i]+1e-10
               for i, e in enumerate(extrema)):
            raise ValueError("synchronized trajectory exceeds command position limits")
        position, speed, acceleration = map(np.asarray, self.trajectory.at_time(elapsed))
        if (not np.isfinite(np.r_[position, speed, acceleration]).all()
                or np.any(np.abs(speed) > self.speed+1e-10)
                or np.any(np.abs(speed-v[:n]) > self.acceleration*dt+1e-10)
                or np.any(np.abs(acceleration) > self.acceleration+1e-10)
                or np.any(np.abs(position-q[:n]) > self.speed*dt+1e-10)):
            raise ValueError("synchronized output violates independent derivative checks")
        candidate, next_velocity = q.copy(), np.zeros(8)
        candidate[:n], next_velocity[:n] = position, speed
        end, stop_time = braking_reserve(position, speed, self.acceleration)
        reserve = candidate.copy()
        reserve[:n] = end
        self.diagnostics = {
            "name": "synchronized_v1", "trajectory_duration_s": self.trajectory.duration,
            "trajectory_elapsed_s": elapsed, "trajectory_reused": reuse,
            "step_acceleration_rad_s2": acceleration.tolist(),
            "braking_reserve_endpoint_rad": reserve.tolist(),
            "braking_reserve_duration_s": stop_time,
            "braking_reserve_policy": "diagnostic_only",
            "braking_reserve_limit_excess_rad": np.maximum(
                np.maximum(self.lower-end, end-self.upper), 0.).tolist(),
            "planned_horizon_is_not_permission_to_execute_stale_actions": True,
        }
        desired_velocity = np.zeros(8)
        desired_velocity[:n] = np.clip((target[:n]-q[:n])/dt, -self.speed, self.speed)
        self._elapsed = elapsed
        self._cached_target = target[:n].copy()
        self._expected_q, self._expected_v = position.copy(), speed.copy()
        return candidate, next_velocity, desired_velocity, dt
