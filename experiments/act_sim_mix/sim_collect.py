"""Simulation-only expert collector for the 100 nominal + 50 corrective set.

No CAN socket, camera device, ROS or Dora process is created.  Expert actions
are written through MuJoCo position actuators and every transition is advanced
with mj_step.  Images are rendered offline from the exact saved qpos states.

Usage examples (pilot only; the operator runs production batches):
  cd openarm-embodied-systems
  MUJOCO_GL=egl uv run --frozen --group sim-mix python -m experiments.act_sim_mix.sim_collect normal \
    --output experiments/act_sim_mix/artifacts/sim_pilot_normal_01 --episodes 3
  MUJOCO_GL=egl uv run --frozen --group sim-mix python -m experiments.act_sim_mix.sim_collect recovery \
    --output experiments/act_sim_mix/artifacts/sim_pilot_recovery_01 --families 1
"""
from __future__ import annotations

import argparse
import hashlib
from functools import wraps
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

import cv2
import mujoco
import numpy as np

from .sim_expert.geometry import yaw_from_quat
from .sim_expert.mapping import SimMapper
from .sim_expert.world import FPS, ROOT, SCENE, SimWorld, load_config

HERE = Path(__file__).resolve().parent
TASK = "Put the blue cube inside the black box with the right arm, then return home."
RECOVERY_TYPES = (
    "closed_approach",
    "lateral_offset",
    "bad_height",
    "recoverable_drop",
    "pre_release_hold",
)
CAMERAS = ("ceiling", "wrist_right")
IMAGE_HEIGHT, IMAGE_WIDTH = 480, 640


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path = Path(path)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def append_jsonl(path: Path, value: dict) -> None:
    with Path(path).open("a") as f:
        f.write(json.dumps(value, allow_nan=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def code_hashes() -> dict[str, str]:
    # Generated runs, audits and tests are not collector source dependencies.
    files = sorted(HERE.glob("*.py")) + sorted((HERE / "sim_expert").glob("*.py"))
    files += sorted((HERE / "sim_expert").glob("*.yaml"))
    files += sorted((HERE / "config").glob("*.json"))
    hashes = {str(p.relative_to(HERE)): digest(p) for p in files if p.is_file()}
    hashes["sim/scenes/act_fixed_20261015/scene.xml"] = digest(SCENE)
    from .contracts import robot_asset_binding
    hashes.update(robot_asset_binding())
    return hashes


def binding(args, domain: str) -> dict:
    return {
        "schema": "act_sim_mix_collection_v2",
        "created_ns": time.time_ns(),
        "domain": domain,
        "source_domain": "simulation",
        "scene": str(SCENE),
        "scene_sha256": digest(SCENE),
        "task": TASK,
        "fps": FPS,
        "image_size_hw": [IMAGE_HEIGHT, IMAGE_WIDTH],
        "cameras": list(CAMERAS),
        "state_action_order": "right J1..J7, right gripper, left J1..J7, left gripper",
        "units": "rad, absolute position; state is feedback, action is issued target",
        "expert_version": "act_sim_mix_expert_v2",
        "code_sha256": code_hashes(),
        "port_manifest_sha256": digest(HERE / "config/port_manifest.json"),
        "mujoco": mujoco.__version__,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "seed": int(args.seed),
        "jitter": {
            "cube_xy_jitter_m": args.cube_xy_jitter_m,
            "cube_yaw_jitter_deg": args.cube_yaw_jitter_deg,
            "table_xy_jitter_m": args.table_xy_jitter_m,
            "table_yaw_jitter_deg": args.table_yaw_jitter_deg,
        },
        "calibrated_dynamics": False,
        "limitations": [
            "Approximate scene; camera extrinsics/intrinsics, table alignment, mass/friction and gains are not identified.",
            "Ideal body gravity compensation is enabled in simulation; this is not a measured hardware response.",
            "Simulation success is not physical success.",
        ],
        "hardware_accessed": False,
    }


def stable_binding(value: dict) -> dict:
    return {k: v for k, v in value.items() if k != "created_ns"}


def prepare_output(output: Path, expected: dict):
    """Acquire the collector lock before any metadata write; keep it at the caller."""
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / ".collector.lock").open("a")
    import fcntl

    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise SystemExit("Another collector holds this output directory")
    try:
        run = output / "run.json"
        if run.exists():
            existing = json.loads(run.read_text())
            if stable_binding(existing) != stable_binding(expected):
                raise ValueError("Existing run has different code/config/scene; use a separate output")
        else:
            if any(p.name not in {".collector.lock"} for p in output.iterdir()):
                raise ValueError("Non-empty output without run.json; refuse to overwrite")
            write_json(run, expected)
        (output / "attempts").mkdir(exist_ok=True)
        (output / "episodes").mkdir(exist_ok=True)
        (output / "episodes/index.jsonl").touch(exist_ok=True)
    except BaseException:
        lock.close()
        raise
    return lock


def collection_run(function):
    @wraps(function)
    def locked(args):
        with prepare_output(args.output.resolve(), binding(args, args.command)):
            for name in ("attempts.jsonl", "episodes/index.jsonl"):
                recover_jsonl_tail(args.output.resolve() / name)
            reconcile_committed_episodes(args.output.resolve())
            return function(args)
    return locked


def recover_jsonl_tail(path: Path) -> None:
    """Keep a torn final append as evidence; committed directories repair the index."""
    if not path.exists():
        return
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):
        end = data.rfind(b"\n") + 1
        evidence = path.with_name(path.name + f".interrupted_{time.time_ns()}")
        with evidence.open("xb") as f:
            f.write(data[end:])
        with path.open("r+b") as f:
            f.truncate(end)
            f.flush()
            os.fsync(f.fileno())
        data = data[:end]
    for line in data.splitlines():
        if line.strip():
            json.loads(line)  # Interior corruption must fail explicitly.


def reconcile_committed_episodes(output: Path) -> None:
    """Recover a crash between atomic episode rename and index publication."""
    index = output / "episodes/index.jsonl"
    entries = [json.loads(line) for line in index.read_text().splitlines() if line.strip()]
    known = {entry["episode_id"] for entry in entries}
    for directory in sorted((output / "episodes").iterdir()):
        if not directory.is_dir() or not directory.name.isdigit():
            continue
        if int(directory.name) in known:
            continue
        meta = json.loads((directory / "metadata.json").read_text())
        replay = json.loads((directory / "replay.json").read_text())
        if not meta["success"] or not replay["success"] or not (directory / "numeric.npz").is_file():
            raise ValueError(f"Unindexed episode is incomplete: {directory}; preserve for diagnosis")
        for role in CAMERAS:
            if len(list((directory / "cameras" / role).glob("*.jpg"))) != meta["frames"]:
                raise ValueError(f"Unindexed episode has missing images: {directory}")
        append_jsonl(index, {
            **{key: meta.get(key) for key in (
                "episode_id", "domain", "group_id", "family_id", "recovery_type", "source",
                "success", "frames", "duration_s", "valid_from_frame", "success_gates")},
            "path": str(directory.relative_to(output)), "replay_success": True,
        })


def attempt_state(output: Path) -> tuple[int, int]:
    numbers = [int(p.name) for p in (output / "attempts").iterdir()
               if p.is_dir() and p.name.isdigit()]
    return len(numbers), max(numbers, default=-1) + 1


class ExpertEpisode:
    def __init__(self, world: SimWorld, domain: str, *, recovery_type: str | None = None):
        self.world = world
        self.domain = domain
        self.recovery_type = recovery_type
        self.takeover_index: int | None = None
        self.takeover_reason: str | None = None
        self.selected_candidate: int | None = None
        self.initial_qpos = world.data.qpos.copy()
        self.initial_ctrl = world.data.ctrl.copy()
        self.phase_log: list[dict] = []

    @property
    def cfg(self):
        return self.world.cfg

    def phase(self, name: str) -> None:
        self.phase_log.append(dict(t=float(self.world.data.time), phase=name))

    def home(self, phase: str = "STABILIZE", duration: float = 1.0, *, valid: bool = True) -> None:
        self.phase(phase)
        self.world.follow(phase, [self.world.home_driver[:8]], duration, valid=valid)

    def current_cube_yaw(self) -> float:
        quat = self.world.data.qpos[
            self.world.cube_qadr + 3 : self.world.cube_qadr + 7
        ]
        return yaw_from_quat(quat)

    def plan_current(self):
        selected, candidates = self.world.planner.plan(
            self.world.cube_xyz(), self.current_cube_yaw(), self.world.state16()[:7].copy()
        )
        if selected is None:
            reasons = [c.failure_reasons for c in candidates]
            raise RuntimeError("no_grasp_candidate:" + json.dumps(reasons))
        self.selected_candidate = selected.index
        self.world.event(
            "grasp_planned",
            candidate=selected.index,
            yaw_rad=selected.yaw,
            candidate_failures=[
                {"index": c.index, "reasons": c.failure_reasons} for c in candidates
            ],
        )
        return selected

    def go_pregrasp(self, selected, gripper: float, *, valid: bool) -> None:
        self.phase("PREGRASP")
        self.world.follow(
            "PREGRASP",
            [np.r_[q, gripper] for q in selected.transit_waypoints],
            valid=valid,
        )

    def go_approach(self, selected, gripper: float, *, valid: bool) -> None:
        self.phase("APPROACH")
        self.world.follow(
            "APPROACH",
            [np.r_[q, gripper] for q in selected.approach_waypoints],
            valid=valid,
        )

    def close_gripper(self, *, valid: bool, duration: float = 1.5) -> None:
        self.phase("CLOSE")
        q = self.world.last_q[:7].copy()
        self.world.follow(
            "CLOSE",
            [np.r_[q, self.cfg["closed_right_gripper_rad"]]],
            duration,
            valid=valid,
        )
        self.phase("VERIFY")
        self.world.follow("VERIFY", [self.world.last_q], 1.0, valid=valid)

    def mark_takeover(self, reason: str, **detail) -> None:
        if self.takeover_index is not None:
            return
        self.takeover_index = len(self.world.samples)
        self.takeover_reason = reason
        self.world.event(
            "expert_takeover",
            reason=reason,
            sample_index=self.takeover_index,
            **detail,
        )

    def complete_after_grasp(self, *, valid: bool = True) -> None:
        world = self.world
        bilateral, _ = world.contacts()
        if not bilateral:
            raise RuntimeError("no_bilateral_grasp")
        world.grasp_relative = (
            world.data.site_xmat[world.site].reshape(3, 3).T
            @ (world.cube_xyz() - world.data.site_xpos[world.site])
        )
        world.event("grasp_confirmed", tcp=world.tcp_xyz().tolist())
        self.phase("LIFT")
        lift = world.tcp_xyz() + np.array([0, 0, 0.065])
        world.cartesian("LIFT", lift, world.data.site_xmat[world.site].reshape(3, 3).copy(), self.cfg["closed_right_gripper_rad"], valid=valid)
        self.phase("HOLD")
        world.follow("HOLD", [world.last_q], 2.0, valid=valid)
        if world.contacts_duration < self.cfg["hold_s"]:
            raise RuntimeError("lift_hold_failed")
        rot = (
            _rot_z(self.cfg["transport_yaw_z_deg"])
            @ _rot_y(self.cfg["transport_roll_y_deg"])
        )
        box_position = world.data.xpos[world.box].copy()
        self.phase("TRANSPORT")
        world.cartesian("TRANSPORT", box_position + [0, 0, 0.13], rot, self.cfg["closed_right_gripper_rad"], valid=valid)
        if not world.contacts()[0]:
            raise RuntimeError("grasp_lost_before_planned_release")
        cube = world.cube_xyz().copy()
        target = world.tcp_xyz().copy()
        target += box_position + [0, 0, self.cfg["release_above_box_m"]] - cube
        self.phase("PLACE_ABOVE_BOX")
        world.cartesian("PLACE_ABOVE_BOX", target, rot, self.cfg["closed_right_gripper_rad"], valid=valid)
        if not world.contacts()[0]:
            raise RuntimeError("grasp_lost_before_planned_release")
        world.event("release_started", cube=cube.tolist(), tcp=world.tcp_xyz().tolist())
        self.phase("RELEASE")
        world.follow(
            "RELEASE",
            [np.r_[world.last_q[:7], self.cfg["pregrasp_right_gripper_rad"]]],
            1.5,
            valid=valid,
        )
        self.phase("SETTLE")
        world.follow("SETTLE", [world.last_q], 2.0, valid=valid)
        self.return_home_after_release(rot, self.cfg["pregrasp_right_gripper_rad"], valid=valid)

    def return_home_after_release(self, rot, gripper: float, *, valid: bool) -> None:
        world = self.world
        home = world.home_driver[:8]
        # Prefer a direct joint-space home path when it is collision-free.  A
        # Cartesian retreat near the box can cross an IK branch and is not
        # required by the task contract.
        safe = True
        for s in np.linspace(0.0, 1.0, 201):
            q = world.last_q[:7] * (1.0 - s) + home[:7] * s
            if world.planner._has_forbidden_collision(
                q, allow_cube=True, gripper=gripper
            ):
                safe = False
                break
        if safe:
            self.phase("RETURN_HOME")
            world.follow("RETURN_HOME", [home], 3.0, valid=valid)
        else:
            self.phase("RETREAT")
            world.cartesian(
                "RETREAT",
                world.tcp_xyz() + [0, 0, 0.07],
                rot,
                gripper,
                valid=valid,
            )
            self.phase("RETURN_HOME")
            world.follow("RETURN_HOME", [home], 3.0, valid=valid)
        self.phase("TAIL")
        world.follow("TAIL", [home], self.cfg["post_success_tail_s"], valid=valid)

    def run_normal(self) -> dict:
        self.world.event("episode_start", domain="normal")
        self.home("STABILIZE", 1.0, valid=True)
        selected = self.plan_current()
        self.go_pregrasp(selected, self.cfg["pregrasp_right_gripper_rad"], valid=True)
        self.go_approach(selected, self.cfg["pregrasp_right_gripper_rad"], valid=True)
        self.close_gripper(valid=True)
        self.complete_after_grasp(valid=True)
        return {}

    # ----- corrective starts -------------------------------------------------
    def retreat_to_pregrasp(self, selected, gripper: float, *, valid: bool) -> None:
        """Open in place, then return to the known pregrasp joint configuration.

        A Cartesian retreat near the failure state can cross an IK branch; the
        joint-space return uses the configuration the expert already traversed.
        """
        world = self.world
        self.phase("RECOVERY_OPEN")
        world.follow(
            "RECOVERY_OPEN",
            [np.r_[world.last_q[:7], gripper]],
            0.8,
            valid=valid,
        )
        qpre = np.r_[selected.transit_waypoints[-1], gripper]
        if world.planner._has_forbidden_collision(
            qpre[:7], allow_cube=True, gripper=qpre[7]
        ):
            raise RuntimeError("recovery_pregrasp_collision")
        self.phase("RECOVERY_PREGRASP")
        world.follow("RECOVERY_PREGRASP", [qpre], 1.5, valid=valid)

    def _wrong_grasp_then_correct(self, *, mode: str, valid=False) -> None:
        world = self.world
        self.home("STABILIZE", 1.0, valid=valid)
        selected = self.plan_current()
        if mode == "closed_approach":
            self.go_pregrasp(selected, self.cfg["return_closed_right_gripper_rad"], valid=valid)
            self.go_approach(selected, self.cfg["return_closed_right_gripper_rad"], valid=valid)
            self.close_gripper(valid=valid)
        elif mode == "lateral_offset":
            self.go_pregrasp(selected, self.cfg["pregrasp_right_gripper_rad"], valid=valid)
            target = world.cube_xyz().copy()
            target[2] += self.cfg["grasp_center_z_offset_m"]
            target[1] += 0.050
            world.cartesian(
                "APPROACH_OFFSET",
                target,
                world.planner.target_rotation(selected.yaw),
                self.cfg["pregrasp_right_gripper_rad"],
                valid=valid,
            )
            self.close_gripper(valid=valid)
        elif mode == "bad_height":
            self.go_pregrasp(selected, self.cfg["pregrasp_right_gripper_rad"], valid=valid)
            target = world.cube_xyz().copy()
            target[2] += self.cfg["grasp_center_z_offset_m"] + 0.045
            world.cartesian(
                "APPROACH_HIGH",
                target,
                world.planner.target_rotation(selected.yaw),
                self.cfg["pregrasp_right_gripper_rad"],
                valid=valid,
            )
            self.close_gripper(valid=valid)
        else:
            raise ValueError(mode)
        # Deliberately small lift attempt; failure is detected from actual contact.
        self.phase("FAILED_LIFT")
        world.cartesian(
            "FAILED_LIFT",
            world.tcp_xyz() + [0, 0, 0.03],
            world.data.site_xmat[world.site].reshape(3, 3).copy(),
            self.cfg["closed_right_gripper_rad"],
            valid=valid,
        )
        bilateral, _ = world.contacts()
        if bilateral:
            raise RuntimeError("expected_no_grasp_but_bilateral_contact_present")
        self.mark_takeover(
            f"{mode}_no_grasp",
            bilateral=bool(bilateral),
            cube=world.cube_xyz().tolist(),
            tcp=world.tcp_xyz().tolist(),
        )
        # Expert recovery: retreat, open, re-plan from the actual disturbed cube.
        self.phase("RECOVERY_RETREAT")
        self.retreat_to_pregrasp(
            selected, self.cfg["pregrasp_right_gripper_rad"], valid=True
        )
        selected = self.plan_current()
        self.go_pregrasp(selected, self.cfg["pregrasp_right_gripper_rad"], valid=True)
        self.go_approach(selected, self.cfg["pregrasp_right_gripper_rad"], valid=True)
        self.close_gripper(valid=True)
        self.complete_after_grasp(valid=True)

    def _recoverable_drop(self) -> None:
        world = self.world
        self.home("STABILIZE", 1.0, valid=False)
        selected = self.plan_current()
        self.go_pregrasp(selected, self.cfg["pregrasp_right_gripper_rad"], valid=False)
        self.go_approach(selected, self.cfg["pregrasp_right_gripper_rad"], valid=False)
        self.close_gripper(valid=False)
        bilateral, _ = world.contacts()
        if not bilateral:
            raise RuntimeError("drop_scenario_failed_to_grasp")
        world.grasp_relative = (
            world.data.site_xmat[world.site].reshape(3, 3).T
            @ (world.cube_xyz() - world.data.site_xpos[world.site])
        )
        self.phase("LIFT_FOR_DROP")
        world.cartesian(
            "LIFT_FOR_DROP",
            world.tcp_xyz() + [0, 0, 0.08],
            world.data.site_xmat[world.site].reshape(3, 3).copy(),
            self.cfg["closed_right_gripper_rad"],
            valid=False,
        )
        self.phase("DROP")
        world.follow(
            "DROP",
            [np.r_[world.last_q[:7], self.cfg["pregrasp_right_gripper_rad"]]],
            1.0,
            valid=False,
        )
        # Let the cube fall and settle on the table; no teleporting.
        world.follow("WAIT_DROP_SETTLE", [world.last_q], 1.5, valid=False)
        cube = world.cube_xyz()
        if not _on_table(world, cube):
            raise RuntimeError("drop_not_recoverable: cube left the table")
        if float(np.linalg.norm(world.data.cvel[world.cube][3:])) > 0.02:
            raise RuntimeError("drop_not_recoverable: cube still moving")
        self.mark_takeover(
            "recoverable_drop_cube_on_table",
            cube=cube.tolist(),
            tcp=world.tcp_xyz().tolist(),
        )
        self.phase("RECOVERY_RETREAT")
        self.retreat_to_pregrasp(
            selected, self.cfg["pregrasp_right_gripper_rad"], valid=True
        )
        selected = self.plan_current()
        self.go_pregrasp(selected, self.cfg["pregrasp_right_gripper_rad"], valid=True)
        self.go_approach(selected, self.cfg["pregrasp_right_gripper_rad"], valid=True)
        self.close_gripper(valid=True)
        self.complete_after_grasp(valid=True)

    def _pre_release_hold(self) -> None:
        world = self.world
        self.home("STABILIZE", 1.0, valid=False)
        selected = self.plan_current()
        self.go_pregrasp(selected, self.cfg["pregrasp_right_gripper_rad"], valid=False)
        self.go_approach(selected, self.cfg["pregrasp_right_gripper_rad"], valid=False)
        self.close_gripper(valid=False)
        bilateral, _ = world.contacts()
        if not bilateral:
            raise RuntimeError("pre_release_scenario_failed_to_grasp")
        world.grasp_relative = (
            world.data.site_xmat[world.site].reshape(3, 3).T
            @ (world.cube_xyz() - world.data.site_xpos[world.site])
        )
        self.phase("LIFT")
        world.cartesian(
            "LIFT",
            world.tcp_xyz() + [0, 0, 0.065],
            world.data.site_xmat[world.site].reshape(3, 3).copy(),
            self.cfg["closed_right_gripper_rad"],
            valid=False,
        )
        self.phase("HOLD")
        world.follow("HOLD", [world.last_q], 2.0, valid=False)
        rot = _rot_z(self.cfg["transport_yaw_z_deg"]) @ _rot_y(self.cfg["transport_roll_y_deg"])
        box_position = world.data.xpos[world.box].copy()
        self.phase("TRANSPORT")
        world.cartesian("TRANSPORT", box_position + [0, 0, 0.13], rot, self.cfg["closed_right_gripper_rad"], valid=False)
        cube = world.cube_xyz().copy()
        target = world.tcp_xyz().copy() + box_position + [0, 0, self.cfg["release_above_box_m"]] - cube
        self.phase("PLACE_ABOVE_BOX")
        world.cartesian("PLACE_ABOVE_BOX", target, rot, self.cfg["closed_right_gripper_rad"], valid=False)
        self.phase("PRE_RELEASE_HOLD")
        world.follow("PRE_RELEASE_HOLD", [world.last_q], 0.5, valid=False)
        self.mark_takeover(
            "at_place_not_released",
            cube=world.cube_xyz().tolist(),
            tcp=world.tcp_xyz().tolist(),
        )
        world.event("release_started", cube=world.cube_xyz().tolist(), tcp=world.tcp_xyz().tolist())
        self.phase("RELEASE")
        world.follow(
            "RELEASE",
            [np.r_[world.last_q[:7], self.cfg["pregrasp_right_gripper_rad"]]],
            1.5,
            valid=True,
        )
        self.phase("SETTLE")
        world.follow("SETTLE", [world.last_q], 2.0, valid=True)
        self.return_home_after_release(rot, self.cfg["pregrasp_right_gripper_rad"], valid=True)

    def run_recovery(self, recovery_type: str) -> dict:
        self.world.event(
            "episode_start",
            domain="recovery",
            recovery_type=recovery_type,
            source="scripted_perturbation",
        )
        if recovery_type in ("closed_approach", "lateral_offset", "bad_height"):
            self._wrong_grasp_then_correct(mode=recovery_type)
        elif recovery_type == "recoverable_drop":
            self._recoverable_drop()
        elif recovery_type == "pre_release_hold":
            self._pre_release_hold()
        else:
            raise ValueError(recovery_type)
        return {"recovery_type": recovery_type}


def _rot_z(degrees: float) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    return Rotation.from_euler("z", degrees, degrees=True).as_matrix()


def _rot_y(degrees: float) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    return Rotation.from_euler("y", degrees, degrees=True).as_matrix()


def _on_table(world: SimWorld, cube: np.ndarray) -> bool:
    box = world.data.xpos[world.box]
    # Table top is z=0.27; cube half edge 0.02.
    return bool(cube[2] < 0.34 and abs(cube[0] - 0.385) < 0.30 and abs(cube[1]) < 0.32)


def _array(samples: list[dict], key: str) -> np.ndarray:
    return np.asarray([s[key] for s in samples])


def data_quality(world: SimWorld, samples: list[dict], success: dict) -> dict:
    times = _array(samples, "time")
    actions = _array(samples, "action")
    states = _array(samples, "state")
    valid = _array(samples, "valid")
    dt = np.diff(times)
    if len(times) < 2:
        raise ValueError("episode too short")
    velocity30 = np.diff(actions[:, :8], axis=0) * FPS
    acceleration30 = np.diff(velocity30, axis=0) * FPS
    return {
        "frames": len(samples),
        "duration_s": float(times[-1] - times[0]),
        "sample_dt_min_s": float(dt.min()),
        "sample_dt_max_s": float(dt.max()),
        "sample_dt_nominal_s": 1.0 / FPS,
        "finite": bool(np.isfinite(states).all() and np.isfinite(actions).all()),
        "state_min": states.min(axis=0).tolist(),
        "state_max": states.max(axis=0).tolist(),
        "action_min": actions.min(axis=0).tolist(),
        "action_max": actions.max(axis=0).tolist(),
        "valid_frames": int(valid.sum()),
        "invalid_frames": int((~valid).sum()),
        "max_internal_command_velocity_rad_s": world.max_v,
        "max_internal_command_acceleration_rad_s2": world.max_a,
        "max_30hz_command_velocity_rad_s": float(np.max(np.abs(velocity30))),
        "max_30hz_command_acceleration_rad_s2": float(np.max(np.abs(acceleration30))),
        "forbidden_contact_events": len(world.forbidden_contacts),
        "gravity_compensation": "ideal body gravcomp=1 (simulation only)",
        "success_gates": success,
        "images_expected": len(samples),
        "images_saved": 0,
    }


def replay_episode(world: SimWorld, samples: list[dict], initial_qpos, initial_ctrl) -> dict:
    """Fresh dynamics rollout of the recorded 30 Hz targets (no qpos writes)."""
    model = world.model
    data = mujoco.MjData(model)
    data.qpos[:] = initial_qpos
    data.ctrl[:] = initial_ctrl
    mujoco.mj_forward(model, data)
    mapper = SimMapper(model)
    times = _array(samples, "time")
    actions = _array(samples, "action")
    phases = [s["phase"] for s in samples]
    valid = _array(samples, "valid").astype(bool)
    dt = float(model.opt.timestep)
    n = len(samples)
    cube_start_z = float(world.nominal_cube_xyz[2])
    peak = 0.0
    peak_valid = 0.0
    hold = 0.0
    max_hold = 0.0
    early_drop = False
    early_drop_at = None
    forbidden = []
    try:
        for i in range(n):
            t0 = times[i]
            t1 = times[i + 1] if i + 1 < n else t0 + 1.0 / FPS
            steps = max(1, int(round((t1 - t0) / dt)))
            for k in range(steps):
                u = (k + 1) / steps
                command = actions[i] * (1 - u) + (actions[i + 1] if i + 1 < n else actions[i]) * u
                mapper.set_ctrl(data.ctrl, command[:8], "right")
                mapper.set_ctrl(data.ctrl, command[8:], "left")
                mujoco.mj_step(model, data)
                if not np.isfinite(data.qpos).all():
                    raise RuntimeError("nonfinite replay physics")
                cube_z = float(data.xpos[world.cube][2])
                peak = max(peak, cube_z - cube_start_z)
                fingers_here: set[str] = set()
                for contact in data.contact:
                    names = [
                        model.body(model.geom_bodyid[x]).name or ""
                        for x in (contact.geom1, contact.geom2)
                    ]
                    if "orange_cube" in names:
                        fingers_here.update(
                            name
                            for name in names
                            if name
                            in (
                                "openarm_right_ee_inner_finger",
                                "openarm_right_ee_outer_finger",
                            )
                        )
                    elif contact.dist < -0.001 and any(n.startswith("openarm_") for n in names):
                        if not all("finger" in n for n in names):
                            forbidden.append(names)
                fingers = len(fingers_here) == 2
                if fingers and cube_z - cube_start_z >= world.cfg["hold_height_m"]:
                    hold += dt
                    max_hold = max(max_hold, hold)
                else:
                    hold = 0.0
                if bool(valid[i]):
                    peak_valid = max(peak_valid, cube_z - cube_start_z)
                if (
                    phases[i] in {"LIFT", "HOLD", "TRANSPORT"}
                    and bool(valid[i])
                    and peak_valid > 0.05
                    and cube_z - cube_start_z < peak_valid - 0.03
                ):
                    early_drop = True
                    if early_drop_at is None:
                        early_drop_at = {
                            "sample": i,
                            "phase": phases[i],
                            "valid": bool(valid[i]),
                            "cube_z_rel": float(cube_z - cube_start_z),
                            "peak_valid": float(peak_valid),
                        }
        rotation = data.xmat[world.box].reshape(3, 3)
        local = rotation.T @ (data.xpos[world.cube] - data.xpos[world.box])
        half = np.abs(rotation.T @ data.xmat[world.cube].reshape(3, 3)) @ np.full(3, 0.02)
        inside = bool(np.all(np.abs(local[:2]) + half[:2] <= 0.06))
        supported = bool(abs(local[2] - half[2] - 0.01) < 0.002)
        home_error = float(np.max(np.abs(
            np.r_[mapper.get_driver(data.qpos, "right")[0], mapper.get_driver(data.qpos, "left")[0]]
            - world.home_driver
        )))
        cube_speed = float(np.linalg.norm(data.cvel[world.cube][3:]))
        success = bool(
            inside and supported and max_hold >= world.cfg["hold_s"] and peak >= world.cfg["hold_height_m"]
            and home_error <= world.cfg["max_home_error_rad"] and not forbidden and not early_drop
            and cube_speed < 0.01
        )
        return {
            "success": success,
            "cube_inside": inside,
            "cube_supported": supported,
            "peak_lift_m": peak,
            "bilateral_lift_s": max_hold,
            "home_error_rad": home_error,
            "early_drop": early_drop,
            "early_drop_at": early_drop_at,
            "forbidden_contacts": forbidden[:5],
            "final_cube": data.xpos[world.cube].tolist(),
            "final_cube_speed_m_s": cube_speed,
            "method": "linear interpolation of recorded 30 Hz targets through fresh mj_step contact dynamics",
        }
    except Exception as exc:
        return {
            "success": False,
            "failure": f"{type(exc).__name__}: {exc}",
            "method": "linear interpolation of recorded 30 Hz targets through fresh mj_step contact dynamics",
        }


def write_episode(
    output: Path,
    episode_id: int,
    world: SimWorld,
    episode: ExpertEpisode,
    meta: dict,
    success: dict,
    replay: dict,
) -> dict:
    samples = world.samples
    directory = output / "episodes" / f"{episode_id:06d}"
    if directory.exists():
        raise FileExistsError(directory)
    staging = output / "episodes" / f".{episode_id:06d}.incomplete"
    if staging.exists():
        staging.rename(staging.with_name(staging.name + f".interrupted_{time.time_ns()}"))
    staging.mkdir()
    valid = _array(samples, "valid").astype(bool)
    valid_from = int(np.argmax(valid)) if valid.any() else None
    if valid_from is not None and not valid[valid_from:].all():
        raise ValueError("valid mask must be a suffix after expert takeover")
    if valid_from is None:
        raise ValueError("episode has no expert-valid segment")
    arrays = {
        "time": _array(samples, "time").astype(np.float64),
        "state": _array(samples, "state").astype(np.float32),
        "action": _array(samples, "action").astype(np.float32),
        "valid": valid,
        "phase": np.asarray([s["phase"] for s in samples], dtype="U32"),
        "cube_xyz": _array(samples, "cube").astype(np.float64),
        "tcp_xyz": _array(samples, "tcp").astype(np.float64),
        "bilateral": _array(samples, "bilateral").astype(bool),
        "qpos": _array(samples, "qpos").astype(np.float64),
        "qvel": _array(samples, "qvel").astype(np.float64),
        "cube_speed": _array(samples, "cube_speed").astype(np.float64),
        "finger_count": _array(samples, "finger_count").astype(np.int64),
        "cube_half_box": _array(samples, "cube_half_box").astype(np.float64),
    }
    np.savez_compressed(staging / "numeric.npz", **arrays)
    # Render the exact saved states, never a separately simulated trajectory.
    data = mujoco.MjData(world.model)
    renderer = mujoco.Renderer(world.model, height=IMAGE_HEIGHT, width=IMAGE_WIDTH)
    try:
        for role in CAMERAS:
            (staging / "cameras" / role).mkdir(parents=True, exist_ok=True)
        for i, sample in enumerate(samples):
            data.qpos[:] = sample["qpos"]
            mujoco.mj_forward(world.model, data)
            for role in CAMERAS:
                renderer.update_scene(data, camera=f"camera_{role}")
                rgb = renderer.render().copy()
                if rgb.shape != (IMAGE_HEIGHT, IMAGE_WIDTH, 3):
                    raise ValueError("unexpected rendered image shape")
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                ok = cv2.imwrite(
                    str(staging / "cameras" / role / f"{i:06d}.jpg"),
                    bgr,
                    [cv2.IMWRITE_JPEG_QUALITY, 90],
                )
                if not ok:
                    raise RuntimeError("JPEG write failed")
    finally:
        renderer.close()
    quality = data_quality(world, samples, success)
    quality["images_saved"] = len(samples)
    quality["replay"] = replay
    quality["valid_from_frame"] = valid_from
    meta = {
        **meta,
        "episode_id": episode_id,
        "task": TASK,
        "fps": FPS,
        "image_size_hw": [IMAGE_HEIGHT, IMAGE_WIDTH],
        "cameras": list(CAMERAS),
        "frames": len(samples),
        "duration_s": float(arrays["time"][-1] - arrays["time"][0]),
        "valid_from_frame": valid_from,
        "takeover_index": episode.takeover_index,
        "takeover_reason": episode.takeover_reason,
        "selected_candidate": episode.selected_candidate,
        "events": world.events,
        "phase_log": episode.phase_log,
        "initial_qpos": episode.initial_qpos.tolist(),
        "initial_ctrl": episode.initial_ctrl.tolist(),
        "success": bool(success["success"]),
        "success_gates": success,
        "randomization": world.randomization,
        "state_action_order": "right J1..J7, right gripper, left J1..J7, left gripper",
        "timing": "state/action sampled at the same 30 Hz instant; image rendered from that exact qpos; action is the issued target, state is feedback",
        "hardware_accessed": False,
        "calibrated_dynamics": False,
    }
    write_json(staging / "metadata.json", meta)
    write_json(staging / "quality.json", quality)
    write_json(staging / "replay.json", replay)
    staging.rename(directory)
    entry = {
        "episode_id": episode_id,
        "path": str(directory.relative_to(output)),
        "domain": meta["domain"],
        "group_id": meta["group_id"],
        "family_id": meta["family_id"],
        "recovery_type": meta.get("recovery_type"),
        "source": meta["source"],
        "success": meta["success"],
        "frames": meta["frames"],
        "duration_s": meta["duration_s"],
        "valid_from_frame": valid_from,
        "success_gates": success,
        "replay_success": replay["success"],
    }
    append_jsonl(output / "episodes/index.jsonl", entry)
    return entry


def run_one(
    output: Path,
    domain: str,
    seed: int,
    cfg: dict,
    *,
    episode_id: int,
    family_id: str,
    recovery_type: str | None = None,
    attempt_id: int,
    jitter: dict,
) -> dict:
    world = SimWorld(seed, cfg, jitter=jitter)
    episode = ExpertEpisode(world, domain, recovery_type=recovery_type)
    result: dict = {"success": False}
    try:
        if domain == "normal":
            episode.run_normal()
            source = "expert_nominal"
            group_id = family_id
        else:
            episode.run_recovery(recovery_type)
            source = "expert_recovery"
            group_id = family_id
        success = world.final_success()
        result.update(success=success["success"], success_gates=success)
        if not success["success"]:
            result["failure"] = "final_success_gate"
            raise RuntimeError("final_success_gate")
        replay = replay_episode(
            world, world.samples, episode.initial_qpos, episode.initial_ctrl
        )
        result["replay"] = replay
        if not replay["success"]:
            result["success"] = False
            result["failure"] = "independent_replay_failed:" + str(replay.get("failure", ""))
            raise RuntimeError("independent_replay_failed")
        meta = {
            "domain": domain,
            "group_id": group_id,
            "family_id": family_id,
            "source": source,
            "failure_state_source": "scripted_perturbation" if domain == "recovery" else None,
            "recovery_type": recovery_type,
            "seed": seed,
            "attempt_id": attempt_id,
            "jitter": jitter,
            "scene_fidelity": {
                "use": "auxiliary_approximate_simulation",
                "camera_alignment_verified": False,
                "table_alignment_verified": False,
                "physical_contact_parameters_identified": False,
            },
        }
        entry = write_episode(output, episode_id, world, episode, meta, success, replay)
        result.update(entry)
        return result
    except Exception as exc:
        result["success"] = False
        result.setdefault("failure", f"{type(exc).__name__}: {exc}")
        result["seed"] = seed
        result["domain"] = domain
        result["recovery_type"] = recovery_type
        result["frames_completed"] = len(world.samples)
        result["randomization"] = world.randomization
        result["takeover_index"] = episode.takeover_index
        result["takeover_reason"] = episode.takeover_reason
        result["events"] = world.events[-20:]
        # Keep the failed rollout for diagnosis without treating it as a label.
        if world.samples:
            attempt_dir = output / "attempts" / f"{attempt_id:06d}"
            attempt_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                attempt_dir / "trace.npz",
                time=_array(world.samples, "time"),
                state=_array(world.samples, "state"),
                action=_array(world.samples, "action"),
                valid=_array(world.samples, "valid"),
                phase=np.asarray([s["phase"] for s in world.samples], dtype="U32"),
                cube_xyz=_array(world.samples, "cube"),
                qpos=_array(world.samples, "qpos"),
            )
        return result


def _jitter_from_args(args) -> dict:
    return {
        "cube_xy_jitter_m": args.cube_xy_jitter_m,
        "cube_yaw_jitter_deg": args.cube_yaw_jitter_deg,
        "table_xy_jitter_m": args.table_xy_jitter_m,
        "table_yaw_jitter_deg": args.table_yaw_jitter_deg,
    }


@collection_run
def cmd_normal(args) -> int:
    output = args.output.resolve()
    cfg = load_config()
    index = [
        json.loads(line)
        for line in (output / "episodes/index.jsonl").read_text().splitlines()
        if line.strip()
    ]
    successes = len(index)
    next_id = max((e["episode_id"] for e in index), default=-1) + 1
    attempts_used, next_attempt = attempt_state(output)
    if successes >= args.episodes:
        print(f"already complete: {successes}/{args.episodes}")
        return 0
    if args.validate_only:
        attempts = 1
    else:
        attempts = args.max_attempts - attempts_used
    jitter = _jitter_from_args(args)
    for attempt in range(attempts):
        number = next_attempt + attempt
        seed = args.seed + number
        family = f"nominal_{seed:08d}"
        attempt_dir = output / "attempts" / f"{number:06d}"
        attempt_dir.mkdir(exist_ok=False)
        write_json(attempt_dir / "request.json", {
            "attempt": number, "seed": seed, "domain": "normal", "family_id": family,
        })
        result = run_one(
            output,
            "normal",
            seed,
            cfg,
            episode_id=next_id,
            family_id=family,
            attempt_id=number,
            jitter=jitter,
        )
        write_json(attempt_dir / "result.json", result)
        append_jsonl(output / "attempts.jsonl", {**result, "attempt": number})
        if result["success"]:
            next_id += 1
            successes += 1
            print(
                f"committed={successes}/{args.episodes}; attempt={number}; seed={seed}; "
                f"frames={result['frames']}; duration={result['duration_s']:.1f}s",
                flush=True,
            )
        else:
            print(
                f"attempt={number}; success=False; failure={result.get('failure')}",
                flush=True,
            )
        if args.validate_only:
            return 0 if result["success"] else 2
        if successes >= args.episodes:
            return 0
    print(f"INCOMPLETE: {successes}/{args.episodes} nominal successes; attempt budget reached")
    return 3



def recovery_progress(output: Path) -> dict[tuple[str, str], set[int]]:
    """Retries already started (completed or interrupted) for each family/type."""
    progress: dict[tuple[str, str], set[int]] = {}
    attempts = output / "attempts.jsonl"
    if attempts.exists():
        for row in [
            json.loads(line) for line in attempts.read_text().splitlines() if line.strip()
        ]:
            if row.get("family_id") and row.get("recovery_type") is not None:
                progress.setdefault(
                    (row["family_id"], row["recovery_type"]), set()
                ).add(int(row.get("retry", 0)))
    for path in sorted((output / "attempts").glob("[0-9]*/request.json")):
        row = json.loads(path.read_text())
        if row.get("family_id") and row.get("recovery_type") is not None:
            progress.setdefault(
                (row["family_id"], row["recovery_type"]), set()
            ).add(int(row.get("retry", 0)))
    return progress


@collection_run
def cmd_recovery(args) -> int:
    output = args.output.resolve()
    cfg = load_config()
    index = [
        json.loads(line)
        for line in (output / "episodes/index.jsonl").read_text().splitlines()
        if line.strip()
    ]
    next_id = max((e["episode_id"] for e in index), default=-1) + 1
    attempts_used, next_attempt = attempt_state(output)
    jitter = _jitter_from_args(args)
    types = [t.strip() for t in args.types.split(",") if t.strip()]
    if not types or len(set(types)) != len(types) or any(t not in RECOVERY_TYPES for t in types):
        raise SystemExit("unknown recovery type")
    total_attempts = 0
    progress = recovery_progress(output)
    budget_exhausted = False
    for family_index in range(args.families):
        family_seed = args.seed + family_index * 1_000_000
        family = f"family_{family_seed:08d}"
        for recovery_type in types:
            type_index = RECOVERY_TYPES.index(recovery_type)
            done = [
                e
                for e in index
                if e["family_id"] == family and e["recovery_type"] == recovery_type
            ]
            if done:
                continue
            tried = progress.get((family, recovery_type), set())
            for retry in range(args.retries):
                if retry in tried:
                    continue
                if attempts_used + total_attempts >= args.max_attempts:
                    print("attempt budget reached")
                    budget_exhausted = True
                    break
                number = next_attempt + total_attempts
                seed = family_seed + type_index + retry * 100
                attempt_dir = output / "attempts" / f"{number:06d}"
                attempt_dir.mkdir(parents=True, exist_ok=False)
                write_json(
                    attempt_dir / "request.json",
                    {
                        "attempt": number,
                        "family_id": family,
                        "recovery_type": recovery_type,
                        "retry": retry,
                        "seed": seed,
                    },
                )
                result = run_one(
                    output,
                    "recovery",
                    seed,
                    cfg,
                    episode_id=next_id,
                    family_id=family,
                    recovery_type=recovery_type,
                    attempt_id=number,
                    jitter=jitter,
                )
                total_attempts += 1
                write_json(attempt_dir / "result.json", result)
                append_jsonl(
                    output / "attempts.jsonl",
                    {
                        **result,
                        "attempt": number,
                        "family_id": family,
                        "recovery_type": recovery_type,
                        "retry": retry,
                        "seed": seed,
                    },
                )
                if result["success"]:
                    index.append(result)
                    next_id += 1
                    print(
                        f"committed {family}/{recovery_type}; frames={result['frames']}; "
                        f"takeover={result['valid_from_frame']}",
                        flush=True,
                    )
                    break
                print(
                    f"failed {family}/{recovery_type} retry={retry}: {result.get('failure')}",
                    flush=True,
                )
            if budget_exhausted:
                break
        if budget_exhausted:
            break
    counts = {
        t: sum(
            1
            for e in index
            if e["domain"] == "recovery" and e["recovery_type"] == t
        )
        for t in types
    }
    complete = all(any(e["family_id"] == f"family_{args.seed + fi * 1_000_000:08d}"
                       and e["recovery_type"] == kind for e in index)
                   for fi in range(args.families) for kind in types)
    print(json.dumps({"counts": counts, "families": args.families, "complete": complete}, indent=2))
    if args.validate_only:
        return 0 if complete else 4
    if not complete:
        print("INCOMPLETE: at least one recovery type is below the requested family count")
        return 4
    return 0


def cmd_validate(args) -> int:
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit("validation output must be new or empty")
    cfg = load_config()
    jitter = _jitter_from_args(args)
    report: dict = {
        "hardware_accessed": False,
        "mujoco": mujoco.__version__,
        "note": "No images written; expert/contact/replay validation only.",
    }
    all_ok = True
    world = SimWorld(args.seed, cfg, jitter=jitter)
    episode = ExpertEpisode(world, "normal")
    episode.run_normal()
    success = world.final_success()
    replay = replay_episode(world, world.samples, episode.initial_qpos, episode.initial_ctrl)
    report["normal"] = {
        "success_gates": success,
        "replay": replay,
        "frames": len(world.samples),
        "duration_s": float(world.samples[-1]["time"] - world.samples[0]["time"]),
    }
    all_ok &= bool(success["success"] and replay["success"])
    report["recovery"] = {}
    for index, recovery_type in enumerate(RECOVERY_TYPES):
        world = SimWorld(args.seed + index, cfg, jitter=jitter)
        episode = ExpertEpisode(world, "recovery", recovery_type=recovery_type)
        try:
            episode.run_recovery(recovery_type)
            success = world.final_success()
            replay = replay_episode(world, world.samples, episode.initial_qpos, episode.initial_ctrl)
            valid = _array(world.samples, "valid").astype(bool)
            row = {
                "success_gates": success,
                "replay": replay,
                "frames": len(world.samples),
                "takeover_index": episode.takeover_index,
                "takeover_reason": episode.takeover_reason,
                "valid_from_frame": int(np.argmax(valid)) if valid.any() else None,
                "duration_s": float(world.samples[-1]["time"] - world.samples[0]["time"]),
            }
            all_ok &= bool(success["success"] and replay["success"] and valid.any())
        except Exception as exc:
            row = {"failure": f"{type(exc).__name__}: {exc}"}
            all_ok = False
        report["recovery"][recovery_type] = row
        print(json.dumps(row, indent=2), flush=True)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "validate_normal.json", report)
    print(json.dumps(report, indent=2))
    return 0 if all_ok else 2


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("normal", "recovery", "validate"):
        q = sub.add_parser(name)
        q.add_argument("--output", type=Path, required=True)
        q.add_argument("--seed", type=int, default=20261028)
        q.add_argument("--cube-xy-jitter-m", type=float, default=0.01)
        q.add_argument("--cube-yaw-jitter-deg", type=float, default=30.0)
        q.add_argument("--table-xy-jitter-m", type=float, default=0.02)
        q.add_argument("--table-yaw-jitter-deg", type=float, default=5.0)
        if name == "normal":
            q.add_argument("--episodes", type=int, default=100)
            q.add_argument("--max-attempts", type=int, default=1800)
            q.add_argument("--validate-only", action="store_true")
        if name == "recovery":
            q.add_argument("--families", type=int, default=10)
            q.add_argument("--types", default=",".join(RECOVERY_TYPES))
            q.add_argument("--retries", type=int, default=8,
                           help="Total attempts per family/type, including previous invocations")
            q.add_argument("--max-attempts", type=int, default=500)
            q.add_argument("--validate-only", action="store_true")
    return p


def main() -> int:
    args = build_parser().parse_args()
    if args.command != "validate" and args.max_attempts < 1:
        raise SystemExit("max-attempts must be positive")
    if args.command == "normal" and args.episodes < 1:
        raise SystemExit("episodes must be positive")
    if args.command == "recovery" and (args.families < 1 or not 1 <= args.retries <= 10_000):
        raise SystemExit("families must be positive and retries must be in 1..10000")
    if args.command == "normal":
        return cmd_normal(args)
    if args.command == "recovery":
        return cmd_recovery(args)
    return cmd_validate(args)


if __name__ == "__main__":
    sys.exit(main())
