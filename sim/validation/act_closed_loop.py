#!/usr/bin/env python3
"""Validate a reloaded LeRobot ACT checkpoint in closed-loop OpenArm MuJoCo.

The default execution path mirrors the official OpenArm simulation stack:
30 Hz ACT chunks -> official Hermite upsampler -> official biquad low-pass at
250 Hz -> data.ctrl -> 1000 Hz MuJoCo position actuators.  A legacy ``guard``
mode is retained as an explicit diagnostic; G5.5.2 showed that applying the
dynamic ActionGuard again to already-safe position targets breaks expert
semantic equivalence.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import sys
import time
import types

if "--viewer" in sys.argv:
    os.environ["MUJOCO_GL"] = "glfw"
else:
    os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[2]
OPENARM_MUJOCO_SRC = ROOT / "official/openarm_mujoco/src"
EXECUTOR_SRC = (
    ROOT
    / "official/dora-openarm-evaluation/nodes/dora-openarm-actions-executor/src"
)
sys.path[:0] = [str(ROOT), str(OPENARM_MUJOCO_SRC), str(EXECUTOR_SRC)]

import mujoco
import mujoco.viewer
import numpy as np
import pyarrow.parquet as pq
import torch
import yaml
from lerobot.policies import get_policy_class, make_pre_post_processors
from openarm_mujoco.v2 import JointResolver
from safetensors.torch import load_file

from openarm_cube_pick_expert.config import sha256
from openarm_cube_pick_expert.geometry import combine_action
from openarm_cube_pick_expert.guard import ActionGuard
from openarm_cube_pick_expert.task import PickCubeEvaluator


SCENE = ROOT / "official/openarm_mujoco/v2/cell/demo.xml"
PARAMS = ROOT / "configs/collection/openarm_cube_pick_expert_params.yaml"
BLACK_BOX_PARAMS = (
    ROOT / "configs/collection/openarm_cube_into_black_box_expert_params.yaml"
)
REFERENCE_DATASET = (
    ROOT / "data/raw/openarm_cube_pick/expert_pilot_g3_frozen_20260903"
)
EXECUTOR_IMPL = EXECUTOR_SRC / "dora_openarm_actions_executor/main.py"
ACTION_INTERVAL_NS = 33_333_333
EXECUTOR_CONTROL_HZ = 250


def load_official_executor_classes():
    """Import math helpers from the official executor without requiring Dora."""
    # The interpolation/filter classes do not use Dora, but its module imports
    # dora at import time. A minimal placeholder keeps this validation isolated
    # from the message bus without copying either implementation.
    sys.modules.setdefault("dora", types.ModuleType("dora"))
    from dora_openarm_actions_executor.main import BiquadLowpass, HermiteUpsampler

    return HermiteUpsampler, BiquadLowpass


def stats_delta(after: Counter, before: Counter) -> dict[str, int]:
    return {
        key: int(after.get(key, 0) - before.get(key, 0))
        for key in sorted(set(after) | set(before))
        if after.get(key, 0) != before.get(key, 0)
    }


def set_wrist_picture_in_picture(viewer, rgb: np.ndarray) -> None:
    """Overlay the live wrist camera in the native MuJoCo framebuffer."""
    import cv2

    viewport = viewer.viewport
    if viewport is None:
        return
    width = min(420, max(240, int(viewport.width * 0.32)))
    height = int(round(width * rgb.shape[0] / rgb.shape[1]))
    image = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
    image = np.ascontiguousarray(image)
    image[[0, -1], :, :] = [0, 255, 0]
    image[:, [0, -1], :] = [0, 255, 0]
    margin = 14
    rect = mujoco.MjrRect(
        viewport.left + margin,
        viewport.bottom + viewport.height - height - margin,
        width,
        height,
    )
    viewer.set_images((rect, image))


def reference_episode_states(dataset: Path, seed: int) -> dict[str, np.ndarray] | None:
    """Load reset metadata and frame-0 observation/action for one raw episode."""
    metadata_path = dataset / "metadata.yaml"
    if not metadata_path.exists():
        return None
    metadata = yaml.safe_load(metadata_path.read_text())
    for entry in metadata.get("episodes", []):
        if int(entry.get("seed", -1)) == seed:
            reset = np.asarray(entry.get("initial_qpos"), dtype=np.float32)
            episode_dir = dataset / "episodes" / str(entry["id"])

            def first_vector(kind: str) -> np.ndarray:
                parts = []
                for side in ("right", "left"):
                    path = episode_dir / kind / "arms" / side / "state.parquet"
                    table = pq.read_table(path, columns=["qpos"])
                    parts.append(
                        np.asarray(table["qpos"][0].as_py(), dtype=np.float32)
                    )
                return np.concatenate(parts)

            result = {
                "reset_metadata": reset,
                "frame0_observation": first_vector("obs"),
                "frame0_action": first_vector("action"),
            }
            if any(value.shape != (16,) for value in result.values()):
                raise ValueError("reference episode states must all be float32[16]")
            return result
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--max-seconds", type=float, default=45.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--task-contract",
        choices=("auto", "pick-place", "cube-into-black-box"),
        default="auto",
        help="task geometry and success evaluator; auto inspects train_config.json",
    )
    parser.add_argument("--viewer", action="store_true", help="show live MuJoCo rollout with wrist-camera picture-in-picture")
    parser.add_argument("--viewer-speed", type=float, default=1.0, help="viewer playback speed multiplier")
    parser.add_argument("--viewer-hz", type=float, default=30.0, help="viewer/PIP refresh rate")
    parser.add_argument("--viewer-hold-s", type=float, default=3.0, help="keep the viewer open after rollout completion")
    parser.add_argument("--stabilize-s", type=float, help="override reset stabilization duration")
    parser.add_argument("--reference-dataset", type=Path, default=REFERENCE_DATASET)
    parser.add_argument(
        "--control-mode",
        choices=("direct", "guard"),
        default="direct",
        help="direct matches official dora-openarm-mujoco; guard is a legacy diagnostic",
    )
    parser.add_argument(
        "--initial-state-mode",
        choices=("reset", "dataset-frame0-teleport"),
        default="reset",
        help=(
            "reset is deployable; dataset-frame0-teleport is a privileged "
            "diagnostic for dataset action/observation alignment"
        ),
    )
    parser.add_argument(
        "--chunk-schedule",
        choices=("whole", "official-stream"),
        default="whole",
        help=(
            "whole executes all 100 actions before replanning; official-stream "
            "models the official Dora executor cancellation/blending path"
        ),
    )
    parser.add_argument(
        "--replan-hz",
        type=float,
        default=10.0,
        help="simulated new-chunk arrival rate for --chunk-schedule official-stream",
    )
    parser.add_argument(
        "--stabilize-left-gripper",
        choices=("config", "zero", "hold"),
        default="zero",
        help=(
            "zero matches the expert episode action/reset contract; hold preserves "
            "the old reset-state behavior for diagnosis"
        ),
    )
    parser.add_argument(
        "--state-sanitization",
        choices=("none", "low-variance-mean"),
        default="low-variance-mean",
        help=(
            "replace near-constant state dimensions with checkpoint training means "
            "before normalization; physical state and control are unchanged"
        ),
    )
    parser.add_argument(
        "--low-variance-std-threshold",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--place-stable-s",
        type=float,
        default=0.25,
        help="required stable on-support duration after release",
    )
    parser.add_argument(
        "--return-home-stable-s",
        type=float,
        default=0.50,
        help="required duration inside the right-arm home tolerance",
    )
    parser.add_argument(
        "--return-home-tolerance-rad",
        type=float,
        default=0.03,
        help="maximum right-arm joint error for RETURN_HOME",
    )
    parser.add_argument(
        "--release-gripper-rad",
        type=float,
        default=-0.35,
        help="right gripper must reach at least this open angle to count RELEASE",
    )
    args = parser.parse_args()
    if args.replan_hz <= 0:
        parser.error("--replan-hz must be positive")
    if args.low_variance_std_threshold <= 0:
        parser.error("--low-variance-std-threshold must be positive")
    if args.place_stable_s <= 0 or args.return_home_stable_s <= 0:
        parser.error("place/home stable durations must be positive")
    if args.return_home_tolerance_rad <= 0:
        parser.error("return-home tolerance must be positive")
    if args.viewer_speed <= 0 or args.viewer_hz <= 0 or args.viewer_hold_s < 0:
        parser.error("viewer speed/hz must be positive and hold time non-negative")

    task_contract = args.task_contract
    if task_contract == "auto":
        train_config_path = args.checkpoint / "train_config.json"
        train_config = (
            json.loads(train_config_path.read_text())
            if train_config_path.exists()
            else {}
        )
        dataset_root = str(train_config.get("dataset", {}).get("root", ""))
        task_contract = (
            "cube-into-black-box"
            if "cube_into_black_box" in dataset_root
            else "pick-place"
        )
    black_box_task = task_contract == "cube-into-black-box"
    params_path = BLACK_BOX_PARAMS if black_box_task else PARAMS
    cfg = yaml.safe_load(params_path.read_text())
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    if black_box_task:
        from openarm_cube_into_black_box_expert.scene import move_black_box_to_middle

        move_black_box_to_middle(model)
    if not np.isclose(model.opt.timestep, 0.001):
        raise ValueError(f"expected 1000 Hz physics, got dt={model.opt.timestep}")
    data = mujoco.MjData(model)
    mapper = JointResolver(model)
    home = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, home)

    cube = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "orange_cube")
    # The expert collector paints this body blue before recording, so the
    # policy was only ever trained on a blue cube. Without this the rollout
    # renders the scene-default orange cube, which is out of distribution.
    cube_geom_ids = np.flatnonzero(model.geom_bodyid == cube)
    model.geom_rgba[cube_geom_ids] = np.array([0.04, 0.18, 0.90, 1.0])
    cube_joint = int(model.body_jntadr[cube])
    cube_qadr = int(model.jnt_qposadr[cube_joint])
    cube_dadr = int(model.jnt_dofadr[cube_joint])
    tcp_site = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, "right_grasp_frame"
    )

    # Match Collector.reset() exactly.  The former ±0.02 m hard-coded square
    # only tested the easy center of the training distribution.
    rng = np.random.default_rng(args.seed)
    cube_center = np.asarray(cfg["cube_xy_center_m"], dtype=np.float64)
    cube_half_range = np.asarray(cfg["cube_xy_half_range_m"], dtype=np.float64)
    cube_xy = cube_center + rng.uniform(-cube_half_range, cube_half_range)
    data.qpos[cube_qadr:cube_qadr + 3] = [
        cube_xy[0],
        cube_xy[1],
        1.05,
    ]
    yaw = rng.uniform(-math.radians(15), math.radians(15))
    data.qpos[cube_qadr + 3:cube_qadr + 7] = [
        math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)
    ]
    data.qvel.fill(0)
    mujoco.mj_forward(model, data)

    if black_box_task:
        nominal_place_xy = np.asarray(cfg["fixed_place_xy_m"], dtype=np.float64)
        place_half_range = np.zeros(2, dtype=np.float64)
    else:
        # Reproduce the old expert's hidden sampled target for diagnostics
        # only. It cannot be a policy success condition because that target is
        # not rendered or included in the ACT observation.
        place_half_range = np.asarray(cfg["place_xy_half_range_m"], dtype=np.float64)
        nominal_place_xy = None
        for _ in range(100):
            candidate = cube_center + rng.uniform(-place_half_range, place_half_range)
            if np.linalg.norm(candidate - cube_xy) >= cfg["place_min_distance_m"]:
                nominal_place_xy = candidate
                break
        if nominal_place_xy is None:
            direction = np.array([1.0, -1.0]); direction /= np.linalg.norm(direction)
            nominal_place_xy = np.clip(
                cube_xy + cfg["place_min_distance_m"] * direction,
                cube_center - place_half_range,
                cube_center + place_half_range,
            )

    def state() -> np.ndarray:
        right, right_gripper = mapper.get_driver(data.qpos, "right")
        left, left_gripper = mapper.get_driver(data.qpos, "left")
        return combine_action(
            np.r_[right, right_gripper], np.r_[left, left_gripper]
        )

    def contacts() -> set[str]:
        names: set[str] = set()
        expected = {
            "openarm_right_ee_inner_finger",
            "openarm_right_ee_outer_finger",
        }
        for index in range(data.ncon):
            contact = data.contact[index]
            bodies = {
                int(model.geom_bodyid[contact.geom1]),
                int(model.geom_bodyid[contact.geom2]),
            }
            if cube not in bodies or len(bodies) != 2:
                continue
            other = (bodies - {cube}).pop()
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, other) or ""
            if name in expected:
                names.add(name)
        return names

    lower = np.r_[
        model.actuator_ctrlrange[9:17, 0], model.actuator_ctrlrange[1:9, 0]
    ]
    upper = np.r_[
        model.actuator_ctrlrange[9:17, 1], model.actuator_ctrlrange[1:9, 1]
    ]
    guard_cfg = cfg["guard"]
    guard = ActionGuard(
        lower,
        upper,
        guard_cfg["max_velocity"],
        guard_cfg["max_acceleration"],
        guard_cfg["max_step"],
        guard_cfg["watchdog_timeout_s"],
        guard_cfg["limit_policy"],
        guard_cfg.get("target_response_time_s", 0.4),
    )

    # Match expert collection reset/stabilization exactly: a static target is
    # continuously submitted for 1.5 s, so watchdog never substitutes a hold.
    stabilize_target = state()
    stabilize_target[7] = cfg["pregrasp_right_gripper_rad"]
    if args.stabilize_left_gripper == "zero":
        # Every expert episode commands the inactive left gripper to 0.0.  Its
        # observed equilibrium is about -1e-3 rad, while preserving the home
        # reset value leaves it about 5.8e-4 rad away.  The pilot set has almost
        # zero variance on this dimension, so that tiny physical mismatch is a
        # many-thousand-sigma normalized input unless initialization matches.
        stabilize_target[15] = 0.0
    elif args.stabilize_left_gripper == "config":
        stabilize_target[15] = cfg.get(
            "left_gripper_open_rad", stabilize_target[15]
        )
    stabilize_start_state = state().copy()
    stabilize_s = cfg["stabilize_s"] if args.stabilize_s is None else args.stabilize_s
    for _ in range(int(round(stabilize_s / model.opt.timestep))):
        guard.submit(stabilize_target, float(data.time))
        guarded = guard.step(state(), float(data.time))
        mapper.set_ctrl(data.ctrl, guarded.action[:8], "right")
        mapper.set_ctrl(data.ctrl, guarded.action[8:], "left")
        mujoco.mj_step(model, data)
    stabilize_end_state = state().copy()
    stabilize_stats = dict(guard.stats)
    policy_stats_baseline = Counter(guard.stats)
    reference_states = reference_episode_states(args.reference_dataset, args.seed)
    reset_state_error = (
        float(
            np.max(
                np.abs(stabilize_end_state - reference_states["reset_metadata"])
            )
        )
        if reference_states is not None
        else None
    )
    frame0_state_error_before = (
        float(
            np.max(
                np.abs(
                    stabilize_end_state - reference_states["frame0_observation"]
                )
            )
        )
        if reference_states is not None
        else None
    )
    if args.initial_state_mode == "dataset-frame0-teleport":
        if reference_states is None:
            raise RuntimeError("dataset frame-0 reference is unavailable")
        mapper.set_qpos(
            data.qpos, reference_states["frame0_observation"][:8], "right"
        )
        mapper.set_qpos(
            data.qpos, reference_states["frame0_observation"][8:], "left"
        )
        data.qvel.fill(0)
        mujoco.mj_forward(model, data)
    policy_initial_state = state().copy()
    frame0_state_error_after = (
        float(
            np.max(
                np.abs(
                    policy_initial_state - reference_states["frame0_observation"]
                )
            )
        )
        if reference_states is not None
        else None
    )

    reset_z = float(data.qpos[cube_qadr + 2])
    evaluator = PickCubeEvaluator(
        reset_z,
        cfg["hold_height_m"],
        cfg["hold_s"],
        cfg["tcp_distance_m"],
        cfg.get("max_relative_drift_m", 0.025),
    )
    reference_relative = None
    bilateral_physics_steps = 0
    lift_hold_success = False
    lift_hold_completed_at_s = None
    release_detected = False
    release_detected_at_s = None
    placement_complete = False
    placement_completed_at_s = None
    place_stable_since_s = None
    return_home_since_s = None
    return_home_completed_at_s = None
    full_sequence_success = False
    max_horizontal_displacement_m = 0.0
    min_place_height_error_m = float("inf")
    min_return_home_error_rad = float("inf")
    release_min_gripper_rad = float("inf")

    if black_box_task:
        from openarm_cube_into_black_box_expert.contract import (
            BLACK_BOX_CUBE_CENTER_Z_M,
            BLACK_BOX_SAFE_HALF_RANGE_M,
            TaskContract,
        )

        box_contract = TaskContract()
        place_min_displacement_m = 0.0
        place_low = nominal_place_xy - np.asarray(BLACK_BOX_SAFE_HALF_RANGE_M)
        place_high = nominal_place_xy + np.asarray(BLACK_BOX_SAFE_HALF_RANGE_M)
        place_height_target_m = BLACK_BOX_CUBE_CENTER_Z_M
    else:
        box_contract = None
        place_min_displacement_m = max(
            0.0, cfg["place_min_distance_m"] - cfg["place_tolerance_m"]
        )
        place_workspace_tolerance = cfg["place_tolerance_m"]
        place_low = cube_center - place_half_range - place_workspace_tolerance
        place_high = cube_center + place_half_range + place_workspace_tolerance
        place_height_target_m = reset_z

    def evaluate_physics_step() -> None:
        nonlocal reference_relative, bilateral_physics_steps
        nonlocal lift_hold_success, lift_hold_completed_at_s
        nonlocal release_detected, release_detected_at_s, release_min_gripper_rad
        nonlocal placement_complete, placement_completed_at_s, place_stable_since_s
        nonlocal return_home_since_s, return_home_completed_at_s
        nonlocal full_sequence_success
        nonlocal max_horizontal_displacement_m, min_place_height_error_m
        nonlocal min_return_home_error_rad
        relative = data.xpos[cube] - data.site_xpos[tcp_site]
        finger_contacts = contacts()
        bilateral = len(finger_contacts) == 2
        if bilateral:
            bilateral_physics_steps += 1
            if reference_relative is None:
                reference_relative = relative.copy()
        drift = (
            float(np.linalg.norm(relative - reference_relative))
            if reference_relative is not None
            else float("inf")
        )
        now_s = float(data.time)
        just_completed_lift_hold = False
        # Freeze the successful hold evidence. Continuing to feed RELEASE and
        # PLACE frames into PickCubeEvaluator would reset its active hold
        # window and misleadingly serialize hold_duration_s as zero.
        if not lift_hold_success:
            just_completed_lift_hold = evaluator.update(
                now_s=float(data.time),
                cube_z=float(data.qpos[cube_qadr + 2]),
                bilateral_contact=bilateral,
                tcp_distance_m=float(np.linalg.norm(relative)),
                relative_drift_m=drift,
                cube_vertical_velocity_m_s=float(data.qvel[cube_dadr + 2]),
            )
        if just_completed_lift_hold and not lift_hold_success:
            lift_hold_success = True
            lift_hold_completed_at_s = now_s

        if not lift_hold_success:
            return

        current_state = state()
        right_gripper = float(current_state[7])
        release_min_gripper_rad = min(release_min_gripper_rad, right_gripper)
        if (
            not release_detected
            and not finger_contacts
            and right_gripper <= args.release_gripper_rad
        ):
            release_detected = True
            release_detected_at_s = now_s

        cube_position = data.xpos[cube]
        horizontal_displacement = float(np.linalg.norm(cube_position[:2] - cube_xy))
        max_horizontal_displacement_m = max(
            max_horizontal_displacement_m, horizontal_displacement
        )
        place_height_error = abs(float(cube_position[2] - place_height_target_m))
        min_place_height_error_m = min(min_place_height_error_m, place_height_error)
        inside_place_workspace = bool(
            np.all(cube_position[:2] >= place_low)
            and np.all(cube_position[:2] <= place_high)
        )
        placement_region_valid = (
            box_contract.box_contains(cube_position)
            if black_box_task
            else (
                horizontal_displacement >= place_min_displacement_m
                and inside_place_workspace
                and place_height_error <= cfg["place_height_tolerance_m"]
            )
        )
        stable_place_now = bool(
            release_detected
            and not finger_contacts
            and placement_region_valid
            and np.linalg.norm(data.qvel[cube_dadr:cube_dadr + 3]) <= 0.03
        )
        if stable_place_now:
            if place_stable_since_s is None:
                place_stable_since_s = now_s
            if now_s - place_stable_since_s >= args.place_stable_s:
                if not placement_complete:
                    placement_complete = True
                    placement_completed_at_s = now_s
        else:
            place_stable_since_s = None

        if not placement_complete:
            return

        return_home_error = float(
            np.max(np.abs(current_state[:7] - rollout_start_state[:7]))
        )
        min_return_home_error_rad = min(
            min_return_home_error_rad, return_home_error
        )
        if return_home_error <= args.return_home_tolerance_rad:
            if return_home_since_s is None:
                return_home_since_s = now_s
            if now_s - return_home_since_s >= args.return_home_stable_s:
                full_sequence_success = True
                return_home_completed_at_s = now_s
        else:
            return_home_since_s = None

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    policy_class = get_policy_class("act")
    policy = policy_class.from_pretrained(args.checkpoint).to(device).eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=args.checkpoint,
        preprocessor_overrides={"device_processor": {"device": device}},
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )
    policy.reset()
    normalizer_state_path = (
        args.checkpoint
        / "policy_preprocessor_step_3_normalizer_processor.safetensors"
    )
    normalizer_state = load_file(normalizer_state_path)
    state_mean = normalizer_state["observation.state.mean"].cpu().numpy()
    state_std = normalizer_state["observation.state.std"].cpu().numpy()
    low_variance_axes = np.flatnonzero(
        state_std < args.low_variance_std_threshold
    )

    renderers = {
        name: mujoco.Renderer(model, height=600, width=960)
        for name in ("ceiling", "wrist_right")
    }
    viewer = mujoco.viewer.launch_passive(model, data) if args.viewer else None
    if viewer is not None:
        camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "camera_ceiling")
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = camera_id
    HermiteUpsampler, BiquadLowpass = load_official_executor_classes()
    chunk_hz = 1e9 / ACTION_INTERVAL_NS
    horizon_s = (policy.config.chunk_size - 1) / chunk_hz
    upsampler = HermiteUpsampler(chunk_hz=chunk_hz, horizon_sec=horizon_s)
    executor_dt = 1.0 / EXECUTOR_CONTROL_HZ
    executor_times = np.arange(0.0, horizon_s + 1e-9, executor_dt)
    lowpass = BiquadLowpass(fs=EXECUTOR_CONTROL_HZ, fc=15.0)

    rollout_start_s = float(data.time)
    deadline_s = rollout_start_s + args.max_seconds
    rollout_start_state = state().copy()
    chunks = 0
    raw_actions = 0
    executor_targets = 0
    physics_steps = 0
    max_state_displacement = 0.0
    action_min = np.full(16, np.inf)
    action_max = np.full(16, -np.inf)
    completed = False
    canceled_positions = None
    viewer_closed = False
    next_viewer_sync = float(data.time)
    next_pip_render = float(data.time)
    viewer_wall_anchor = time.monotonic()
    viewer_sim_anchor = float(data.time)
    stream_executor_steps = max(
        1, int(round(EXECUTOR_CONTROL_HZ / args.replan_hz))
    )

    try:
        while data.time < deadline_s and not full_sequence_success and not viewer_closed:
            policy_state = state().copy()
            if args.state_sanitization == "low-variance-mean":
                policy_state[low_variance_axes] = state_mean[low_variance_axes]
            batch = {
                "observation.state": torch.from_numpy(policy_state).unsqueeze(0)
            }
            for key, camera in (
                ("ceiling", "camera_ceiling"),
                ("wrist_right", "camera_wrist_right"),
            ):
                renderers[key].update_scene(data, camera=camera)
                rgb = renderers[key].render().copy()
                batch[f"observation.images.{key}"] = (
                    torch.from_numpy(rgb).permute(2, 0, 1).float().div(255).unsqueeze(0)
                )
            with torch.inference_mode():
                chunk = postprocessor(
                    policy.predict_action_chunk(preprocessor(batch))
                ).numpy()[0]
            if chunk.shape != (policy.config.chunk_size, 16) or not np.isfinite(chunk).all():
                raise RuntimeError(f"invalid ACT chunk: shape={chunk.shape}")
            chunks += 1
            raw_actions += len(chunk)
            action_min = np.minimum(action_min, chunk.min(axis=0))
            action_max = np.maximum(action_max, chunk.max(axis=0))

            positions = chunk
            if args.chunk_schedule == "official-stream" and canceled_positions is not None:
                # Match dora_openarm_actions_executor._main_executor.blend():
                # the unconsumed old raw trajectory fades into the new prediction.
                n_blend = len(canceled_positions)
                old_weight = np.linspace(1, 0, n_blend, dtype=np.float32)[:, None]
                blended = (
                    canceled_positions * old_weight
                    + positions[:n_blend] * (1 - old_weight)
                )
                positions = np.concatenate([blended, positions[n_blend:]])

            executor_chunk = upsampler.upsample(positions, executor_times)
            if chunks == 1:
                lowpass.reset_state(executor_chunk[0])
            target_limit = (
                len(executor_chunk)
                if args.chunk_schedule == "whole"
                else min(stream_executor_steps, len(executor_chunk))
            )
            targets_executed_this_chunk = 0
            for target in executor_chunk[:target_limit]:
                filtered_target = lowpass.step(target)
                executor_targets += 1
                targets_executed_this_chunk += 1
                if args.control_mode == "guard":
                    guard.submit(filtered_target, float(data.time))
                else:
                    mapper.set_ctrl(data.ctrl, filtered_target[:8], "right")
                    mapper.set_ctrl(data.ctrl, filtered_target[8:], "left")
                for _ in range(int(round(executor_dt / model.opt.timestep))):
                    if args.control_mode == "guard":
                        guarded = guard.step(state(), float(data.time))
                        mapper.set_ctrl(data.ctrl, guarded.action[:8], "right")
                        mapper.set_ctrl(data.ctrl, guarded.action[8:], "left")
                    mujoco.mj_step(model, data)
                    physics_steps += 1
                    max_state_displacement = max(
                        max_state_displacement,
                        float(np.max(np.abs(state() - rollout_start_state))),
                    )
                    evaluate_physics_step()
                    if viewer is not None and data.time + 1e-12 >= next_pip_render:
                        renderers["wrist_right"].update_scene(data, camera="camera_wrist_right")
                        set_wrist_picture_in_picture(viewer, renderers["wrist_right"].render().copy())
                        next_pip_render += 1.0 / args.viewer_hz
                    if viewer is not None and data.time + 1e-12 >= next_viewer_sync:
                        viewer.sync()
                        target_wall = viewer_wall_anchor + (float(data.time) - viewer_sim_anchor) / args.viewer_speed
                        remaining = target_wall - time.monotonic()
                        if remaining > 0:
                            time.sleep(remaining)
                        viewer_wall_anchor = time.monotonic()
                        viewer_sim_anchor = float(data.time)
                        next_viewer_sync += 1.0 / args.viewer_hz
                        viewer_closed = not viewer.is_running()
                    if full_sequence_success or data.time >= deadline_s or viewer_closed:
                        break
                if full_sequence_success or data.time >= deadline_s or viewer_closed:
                    break
            if args.chunk_schedule == "official-stream":
                consumed_time_s = targets_executed_this_chunk * executor_dt
                consumed_raw_steps = int(consumed_time_s * chunk_hz)
                canceled_positions = positions[consumed_raw_steps:]
        completed = not viewer_closed
        if viewer is not None and not viewer_closed and args.viewer_hold_s:
            hold_deadline = time.monotonic() + args.viewer_hold_s
            while viewer.is_running() and time.monotonic() < hold_deadline:
                viewer.sync()
                time.sleep(min(1.0 / args.viewer_hz, hold_deadline - time.monotonic()))
    finally:
        if viewer is not None:
            viewer.close()
        for renderer in renderers.values():
            renderer.close()

    evidence = evaluator.evidence()
    valid_hold_frames = [frame for frame in evidence.pop("hold_frames") if frame["valid"]]
    policy_stats = (
        stats_delta(guard.stats, policy_stats_baseline)
        if args.control_mode == "guard"
        else {}
    )
    total_policy_interventions = sum(
        count for key, count in policy_stats.items() if key.startswith(("clip_", "hold_", "reject_"))
    )
    task_success = bool(placement_complete)
    if task_success:
        failure_reason = None
    elif not lift_hold_success:
        failure_reason = "lift_hold_condition_not_met"
    elif not release_detected:
        failure_reason = "release_not_detected"
    else:
        failure_reason = "placement_condition_not_met"

    if full_sequence_success:
        full_sequence_failure_reason = None
    elif not task_success:
        full_sequence_failure_reason = failure_reason
    else:
        full_sequence_failure_reason = "return_home_condition_not_met"

    final_cube_position = data.xpos[cube].copy()
    final_horizontal_displacement_m = float(
        np.linalg.norm(final_cube_position[:2] - cube_xy)
    )
    final_place_height_error_m = abs(
        float(final_cube_position[2] - place_height_target_m)
    )
    final_inside_black_box = bool(
        box_contract.box_contains(final_cube_position)
        if black_box_task else False
    )
    final_return_home_error_rad = float(
        np.max(np.abs(state()[:7] - rollout_start_state[:7]))
    )
    report = {
        "seed": args.seed,
        "rollout_completed": completed,
        "task_success": bool(task_success),
        "failure_reason": failure_reason,
        "task_contract": (
            "openarm_cube_into_black_box_v3"
            if black_box_task else "pick_hold_place_release"
        ),
        "full_sequence_success": bool(full_sequence_success),
        "full_sequence_failure_reason": full_sequence_failure_reason,
        "full_sequence_contract": (
            "openarm_cube_into_black_box_v3_return_home"
            if black_box_task else "pick_hold_place_release_return_home"
        ),
        "lift_hold_success": bool(lift_hold_success),
        "release_detected": bool(release_detected),
        "placement_complete": bool(placement_complete),
        "return_home_complete": bool(full_sequence_success),
        "phase_timestamps_s": {
            "lift_hold_complete": lift_hold_completed_at_s,
            "release_detected": release_detected_at_s,
            "placement_complete": placement_completed_at_s,
            "return_home_complete": return_home_completed_at_s,
        },
        "randomization": {
            "cube_xy_center_m": cube_center.tolist(),
            "cube_xy_half_range_m": cube_half_range.tolist(),
            "initial_cube_xy_m": cube_xy.tolist(),
            "initial_cube_yaw_rad": float(yaw),
            "nominal_expert_place_target_xy_m": nominal_place_xy.tolist(),
            "nominal_target_is_success_condition": bool(black_box_task),
        },
        "placement": {
            "minimum_displacement_m": float(place_min_displacement_m),
            "maximum_horizontal_displacement_m": float(max_horizontal_displacement_m),
            "final_horizontal_displacement_m": final_horizontal_displacement_m,
            "height_tolerance_m": float(cfg["place_height_tolerance_m"]),
            "minimum_height_error_m": (
                None if math.isinf(min_place_height_error_m)
                else float(min_place_height_error_m)
            ),
            "final_height_error_m": final_place_height_error_m,
            "inside_black_box": final_inside_black_box,
            "stable_duration_required_s": float(args.place_stable_s),
            "workspace_low_xy_m": place_low.tolist(),
            "workspace_high_xy_m": place_high.tolist(),
        },
        "release": {
            "open_threshold_rad": float(args.release_gripper_rad),
            "minimum_gripper_after_lift_rad": (
                None if math.isinf(release_min_gripper_rad)
                else float(release_min_gripper_rad)
            ),
        },
        "return_home": {
            "tolerance_rad": float(args.return_home_tolerance_rad),
            "stable_duration_required_s": float(args.return_home_stable_s),
            "minimum_error_rad": (
                None if math.isinf(min_return_home_error_rad)
                else float(min_return_home_error_rad)
            ),
            "final_error_rad": final_return_home_error_rad,
        },
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_model_sha256": sha256(args.checkpoint / "model.safetensors"),
        "scene_sha256": sha256(SCENE),
        "params_sha256": sha256(params_path),
        "executor_equivalence": {
            "source": str(EXECUTOR_IMPL.relative_to(ROOT)),
            "source_sha256": sha256(EXECUTOR_IMPL),
            "upsampler_class": f"{HermiteUpsampler.__module__}.{HermiteUpsampler.__name__}",
            "filter_class": f"{BiquadLowpass.__module__}.{BiquadLowpass.__name__}",
            "action_interval_ns": ACTION_INTERVAL_NS,
            "chunk_hz": chunk_hz,
            "control_hz": EXECUTOR_CONTROL_HZ,
            "filter_cutoff_hz": 15.0,
            "physics_hz": int(round(1.0 / model.opt.timestep)),
            "control_mode": args.control_mode,
            "chunk_schedule": args.chunk_schedule,
            "replan_hz": (
                args.replan_hz if args.chunk_schedule == "official-stream" else None
            ),
            "stream_executor_steps_per_chunk": (
                stream_executor_steps
                if args.chunk_schedule == "official-stream"
                else None
            ),
            "chain": (
                "ACT -> official Hermite -> official Biquad -> data.ctrl -> mj_step"
                if args.control_mode == "direct"
                else "ACT -> official Hermite -> official Biquad -> ActionGuard -> data.ctrl -> mj_step"
            ),
        },
        "stabilize": {
            "duration_s": stabilize_s,
            "left_gripper_target_mode": args.stabilize_left_gripper,
            "left_gripper_target_rad": float(stabilize_target[15]),
            "initial_state_mode": args.initial_state_mode,
            "privileged_diagnostic": (
                args.initial_state_mode == "dataset-frame0-teleport"
            ),
            "state_displacement_max_rad": float(
                np.max(np.abs(stabilize_end_state - stabilize_start_state))
            ),
            "reset_metadata_error_max_rad": reset_state_error,
            "frame0_observation_error_before_adjustment_max_rad": (
                frame0_state_error_before
            ),
            "frame0_observation_error_at_policy_start_max_rad": (
                frame0_state_error_after
            ),
            "guard_stats": stabilize_stats,
        },
        "policy": {
            "rollout_start_s": rollout_start_s,
            "rollout_end_s": float(data.time),
            "duration_s": float(data.time - rollout_start_s),
            "chunks": chunks,
            "raw_actions": raw_actions,
            "executor_targets": executor_targets,
            "physics_steps": physics_steps,
            "guard_stats": policy_stats,
            "guard_interventions": total_policy_interventions,
            "max_state_displacement_rad": max_state_displacement,
            "bilateral_contact_physics_steps": bilateral_physics_steps,
            "state_sanitization": args.state_sanitization,
            "low_variance_std_threshold": args.low_variance_std_threshold,
            "low_variance_axes": low_variance_axes.tolist(),
            "low_variance_axis_std": state_std[low_variance_axes].tolist(),
            "raw_action_min": action_min.tolist(),
            "raw_action_max": action_max.tolist(),
        },
        **evidence,
        "valid_hold_frame_count": len(valid_hold_frames),
        "valid_hold_first": valid_hold_frames[0] if valid_hold_frames else None,
        "valid_hold_last": valid_hold_frames[-1] if valid_hold_frames else None,
        "final_state": state().tolist(),
        "final_cube_pose_wxyz": data.qpos[cube_qadr:cube_qadr + 7].tolist(),
    }
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")
    # A smoke rollout may be a valid recorded task failure. Runtime/data errors
    # raise above; task failure itself does not make the validator command fail.


if __name__ == "__main__":
    main()
