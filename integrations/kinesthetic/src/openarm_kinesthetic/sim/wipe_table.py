"""Replay a measured right-arm trajectory in the OpenArm v2 MuJoCo scene."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Iterable

import numpy as np

from ..paths import workspace_path


SCENE = workspace_path("sim", "scenes", "wipe_table.xml")
GLOBAL_CAMERA = "camera_ceiling"
LOCAL_CAMERA = "camera_wrist_right"
FPS = 30
WIDTH = 1280
HEIGHT = 720
INSET_WIDTH = 384
INSET_HEIGHT = 240
INSET_MARGIN = 18
INSET_BORDER = 4
LEAD_S = 1.0
TAIL_S = 1.0
PREVIEW_FPS = 60
TABLE_TOP_Z = 0.270
# The recorded gripper tip never descends below z = 0.2912 m, so a cloth lying
# on the 0.270 m tabletop has to be at least ~24 mm thick before the tip can
# reach into it at all; 28 mm gives a stable ~6.8 mm press through both wipes.
CLOTH_THICKNESS_M = 0.028
CLOTH_HALF_THICKNESS_M = CLOTH_THICKNESS_M / 2.0
TABLE_CLOTH_CENTER_Z = TABLE_TOP_Z + CLOTH_HALF_THICKNESS_M
CLOTH_TOP_Z = TABLE_TOP_Z + CLOTH_THICKNESS_M
CLOTH_HALF_SIZE_XY_M = np.asarray((0.100, 0.070), dtype=float)
CLOTH_GEOM_NAME = "purple_wipe_cloth_visual"
# Collision meshes of the gripper head, taken from the vendor OpenArm v2 model.
GRIPPER_TIP_PREFIXES = (
    "ee_base_link_right_collision",
    "finger_inner_right_collision",
    "finger_outer_right_collision",
)
# How deep the gripper tip has to be inside the cloth volume to drag it.
CLOTH_PRESS_DEPTH_M = 0.001


def _add_source_dependency(path: Path) -> None:
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)


def load_trajectory(path: Path) -> dict:
    """Load with the existing real-recorder contract; never rewrite the file."""
    _add_source_dependency(
        workspace_path("ros2_ws", "src", "openarm_kinesthetic_recorder")
    )
    from openarm_kinesthetic_recorder.recorded_motion import load_recorded_motion

    return load_recorded_motion(path.resolve())


def sampled_positions(payload: dict, source_times_s: Iterable[float]) -> np.ndarray:
    """Linearly sample q(t) on arbitrary times using the recorded timestamps."""
    samples = payload["samples"]
    recorded_t = np.asarray(
        [sample["time_from_start_s"] for sample in samples], dtype=float
    )
    recorded_q = np.asarray(
        [sample["positions_rad"] for sample in samples], dtype=float
    )
    query = np.clip(
        np.asarray(tuple(source_times_s), dtype=float), recorded_t[0], recorded_t[-1]
    )
    return np.column_stack(
        [np.interp(query, recorded_t, recorded_q[:, joint]) for joint in range(8)]
    )


def video_source_times(duration_s: float, fps: int = FPS) -> np.ndarray:
    frame_count = int(math.ceil((LEAD_S + duration_s + TAIL_S) * fps))
    video_t = np.arange(frame_count, dtype=float) / fps
    return np.clip(video_t - LEAD_S, 0.0, duration_s)


def _dependencies():
    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco

    _add_source_dependency(workspace_path("vendor", "openarm_mujoco", "src"))
    from openarm_mujoco.v2.joint_resolver import JointResolver

    return mujoco, JointResolver


def _scene(payload: dict):
    mujoco, resolver_type = _dependencies()
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    resolver = resolver_type(model)
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key_id < 0:
        raise RuntimeError("wipe scene is missing its fixed-left-arm home keyframe")
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    cloth_body = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "purple_wipe_cloth"
    )
    cloth_mocap = int(model.body_mocapid[cloth_body])
    grasp_site = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, "right_grasp_frame"
    )
    if cloth_mocap < 0 or grasp_site < 0:
        raise RuntimeError("wipe scene is missing the cloth or right grasp frame")
    # The vendor site's bright green marker dominates the wrist close-up.  Its
    # pose stays in use as the point the cloth is dragged by; only the debug
    # drawing is hidden.
    model.site_rgba[grasp_site, 3] = 0.0
    first = np.asarray(payload["samples"][0]["positions_rad"], dtype=float)
    resolver.set_qpos(data.qpos, first, "right")
    mujoco.mj_forward(model, data)
    cloth_home = np.array(data.mocap_pos[cloth_mocap], dtype=float)
    return mujoco, model, data, resolver, cloth_mocap, grasp_site, cloth_home


def _tcp_sequence(mujoco, model, data, resolver, grasp_site, positions) -> np.ndarray:
    """Forward-kinematics of the right grasp frame for every replay frame."""
    tcp = np.empty((len(positions), 3), dtype=float)
    for index, q in enumerate(positions):
        resolver.set_qpos(data.qpos, q, "right")
        mujoco.mj_forward(model, data)
        tcp[index] = data.site_xpos[grasp_site]
    return tcp


def press_and_track(
    mujoco, model, data, resolver, cloth_mocap, cloth_geom, tip_geoms,
    positions, tcp, fallback_xy,
) -> tuple[np.ndarray, np.ndarray]:
    """Press mask and cloth centres, in one interleaved pass.

    The press test needs the cloth where it currently is, and the cloth's
    position depends on the presses, so both are advanced together.  A seed
    scan first places the cloth beneath the gripper to find the frame where the
    tip first sinks in, which is where the cloth gets laid out.
    """
    seed_frame = None
    for index in range(len(positions)):
        resolver.set_qpos(data.qpos, positions[index], "right")
        _set_cloth_position(
            data, cloth_mocap,
            np.asarray((tcp[index][0], tcp[index][1], TABLE_CLOTH_CENTER_Z)),
        )
        mujoco.mj_forward(model, data)
        if presses_cloth(mujoco, data, cloth_geom, tip_geoms):
            seed_frame = index
            break
    center = np.asarray(
        tcp[seed_frame, :2] if seed_frame is not None else fallback_xy, dtype=float
    ).copy()

    pressing = np.zeros(len(positions), dtype=bool)
    centers = np.empty((len(positions), 2), dtype=float)
    for index in range(len(positions)):
        resolver.set_qpos(data.qpos, positions[index], "right")
        _set_cloth_position(
            data, cloth_mocap,
            np.asarray((center[0], center[1], TABLE_CLOTH_CENTER_Z)),
        )
        mujoco.mj_forward(model, data)
        pressing[index] = presses_cloth(mujoco, data, cloth_geom, tip_geoms)
        if index > 0:
            moved, _ = next_cloth_position(
                np.append(center, TABLE_CLOTH_CENTER_Z),
                tcp[index],
                tcp[index - 1],
                pressed=bool(pressing[index]),
                previously_pressed=bool(pressing[index - 1]),
            )
            center = moved[:2].copy()
        centers[index] = center
    return pressing, centers


def cloth_replay(payload: dict, positions: np.ndarray):
    """Scene handles plus the cloth pose decided for every replay frame."""
    (
        mujoco, model, data, resolver, cloth_mocap, grasp_site, cloth_home
    ) = _scene(payload)
    cloth_geom, tip_geoms = cloth_and_tip_geoms(mujoco, model)
    tcp = _tcp_sequence(mujoco, model, data, resolver, grasp_site, positions)
    pressing, centers = press_and_track(
        mujoco, model, data, resolver, cloth_mocap, cloth_geom, tip_geoms,
        positions, tcp, cloth_home[:2],
    )
    return mujoco, model, data, resolver, cloth_mocap, centers, pressing


def _set_cloth_position(data, mocap_id: int, xyz: np.ndarray) -> None:
    data.mocap_pos[mocap_id] = xyz
    data.mocap_quat[mocap_id] = (1.0, 0.0, 0.0, 0.0)


def cloth_and_tip_geoms(mujoco, model) -> tuple[int, frozenset[int]]:
    """The cloth's collision box and the gripper-head collision meshes."""
    cloth = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, CLOTH_GEOM_NAME)
    if cloth < 0:
        raise RuntimeError(f"wipe scene is missing geom {CLOTH_GEOM_NAME!r}")
    tips = set()
    for geom in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) or ""
        if name.startswith(GRIPPER_TIP_PREFIXES):
            tips.add(geom)
    if not tips:
        raise RuntimeError("wipe scene has no right gripper-head collision geoms")
    return cloth, frozenset(tips)


def presses_cloth(mujoco, data, cloth_geom: int, tip_geoms) -> bool:
    """True while a gripper-head mesh is sunk into the cloth volume.

    This is the real contact test, not a height proxy: it uses the fingertip
    meshes from the arm model, so it follows the gripper's pose and only fires
    once the tip is genuinely inside the cloth.
    """
    for index in range(data.ncon):
        contact = data.contact[index]
        first, second = contact.geom1, contact.geom2
        if first == cloth_geom:
            other = second
        elif second == cloth_geom:
            other = first
        else:
            continue
        if other in tip_geoms and -float(contact.dist) >= CLOTH_PRESS_DEPTH_M:
            return True
    return False


def next_cloth_position(
    current_xyz: np.ndarray,
    tcp_xyz: np.ndarray,
    previous_tcp_xyz: np.ndarray,
    *,
    pressed: bool,
    previously_pressed: bool,
) -> tuple[np.ndarray, bool]:
    """Drag the cloth by the gripper's own motion while it presses the cloth.

    The cloth inherits the TCP increment instead of snapping to the TCP.  A
    lift-and-replace therefore leaves the cloth where it was released and only
    resumes dragging once the gripper is back inside the cloth, and the cloth
    can never move faster than the gripper that drags it.
    """
    current = np.asarray(current_xyz, dtype=float)
    if not (pressed and previously_pressed):
        return current.copy(), False
    dragged = current.copy()
    dragged[:2] += (
        np.asarray(tcp_xyz, dtype=float)[:2]
        - np.asarray(previous_tcp_xyz, dtype=float)[:2]
    )
    dragged[2] = TABLE_CLOTH_CENTER_Z
    return dragged, True


def cloth_track(
    tcp_xyz: np.ndarray, pressing: np.ndarray
) -> np.ndarray:
    """Cloth centre per frame from an already-known press mask.

    Offline counterpart of the interleaved pass: same rule, but it can be
    replayed on a stored mask without a MuJoCo scene.
    """
    tcp = np.asarray(tcp_xyz, dtype=float)
    pressed = np.asarray(pressing, dtype=bool)
    low = np.flatnonzero(pressed)
    center = (tcp[int(low[0]), :2] if low.size else tcp[0, :2]).copy()
    centers = np.empty((len(tcp), 2), dtype=float)
    for index in range(len(tcp)):
        if index > 0:
            moved, _ = next_cloth_position(
                np.append(center, TABLE_CLOTH_CENTER_Z),
                tcp[index],
                tcp[index - 1],
                pressed=bool(pressed[index]),
                previously_pressed=bool(pressed[index - 1]),
            )
            center = moved[:2].copy()
        centers[index] = center
    return centers


def compose_picture_in_picture(
    main_rgb: np.ndarray,
    inset_rgb: np.ndarray,
    *,
    margin: int = INSET_MARGIN,
    border: int = INSET_BORDER,
) -> np.ndarray:
    """Place the wrist image at top left, away from the right-arm task area."""
    main = np.asarray(main_rgb, dtype=np.uint8)
    inset = np.asarray(inset_rgb, dtype=np.uint8)
    if (
        main.ndim != 3
        or inset.ndim != 3
        or main.shape[2:] != (3,)
        or inset.shape[2:] != (3,)
    ):
        raise ValueError("main and inset frames must be RGB images")
    height, width = inset.shape[:2]
    y0 = margin + border
    x0 = margin + border
    if x0 + width + border > main.shape[1] or y0 + height + border > main.shape[0]:
        raise ValueError("inset does not fit inside the main frame")
    output = main.copy()
    output[y0 - border : y0 + height + border, x0 - border : x0 + width + border] = 245
    output[y0 : y0 + height, x0 : x0 + width] = inset
    return output


def _apply_frame(
    mujoco, model, data, resolver, cloth_mocap, q: np.ndarray, cloth_xy: np.ndarray
) -> None:
    resolver.set_qpos(data.qpos, q, "right")
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    _set_cloth_position(
        data,
        cloth_mocap,
        np.asarray((cloth_xy[0], cloth_xy[1], TABLE_CLOTH_CENTER_Z)),
    )
    mujoco.mj_forward(model, data)


def inset_rect(mujoco, viewport):
    """Top-left inset rectangle, mirroring the recorded video layout."""
    left = INSET_MARGIN + INSET_BORDER
    bottom = viewport.height - (INSET_MARGIN + INSET_BORDER + INSET_HEIGHT)
    return mujoco.MjrRect(max(left, 0), max(bottom, 0), INSET_WIDTH, INSET_HEIGHT)


def trajectory_report(payload: dict) -> dict:
    t = np.asarray(
        [sample["time_from_start_s"] for sample in payload["samples"]], dtype=float
    )
    q = np.asarray(
        [sample["positions_rad"] for sample in payload["samples"]], dtype=float
    )
    dt = np.diff(t)
    return {
        "duration_s": float(t[-1]),
        "sample_count": int(len(t)),
        "estimated_sample_rate_hz": float((len(t) - 1) / t[-1]),
        "joint_names": list(payload["joint_names"]),
        "joint_min_rad": q.min(axis=0).tolist(),
        "joint_max_rad": q.max(axis=0).tolist(),
        "start_positions_rad": q[0].tolist(),
        "end_positions_rad": q[-1].tolist(),
        "finite": bool(np.isfinite(t).all() and np.isfinite(q).all()),
        "strictly_increasing_time": bool(np.all(dt > 0.0)),
        "maximum_joint_step_rad": float(np.max(np.abs(np.diff(q, axis=0)))),
    }


def preview(payload: dict) -> None:
    """Run the complete replay once in the viewer at original timing.

    The wrist camera is rendered offscreen and overlaid inside the same window,
    so the GUI shows the same picture-in-picture the recording writes out.
    """
    # This mode opens a window, so it must not inherit the EGL default that the
    # offscreen recording needs: an EGL context cannot be made current on a
    # thread that already holds the viewer's window context.  Set before the
    # mujoco import below, which is what reads MUJOCO_GL.
    os.environ.setdefault("MUJOCO_GL", "glfw")
    source_t = video_source_times(float(payload["duration_s"]), fps=PREVIEW_FPS)
    positions = sampled_positions(payload, source_t)
    mujoco, model, data, resolver, cloth_mocap, centers, _ = cloth_replay(
        payload, positions
    )
    import mujoco.viewer

    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, GLOBAL_CAMERA)
    # The left UI panel would sit on top of the inset, which belongs at top left.
    renderer = mujoco.Renderer(model, height=INSET_HEIGHT, width=INSET_WIDTH)
    viewer = mujoco.viewer.launch_passive(model, data, show_left_ui=False)
    try:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = camera_id
        started = time.monotonic()
        for index, (q, cloth_xy) in enumerate(zip(positions, centers)):
            if not viewer.is_running():
                break
            deadline = started + index / PREVIEW_FPS
            delay = deadline - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            _apply_frame(mujoco, model, data, resolver, cloth_mocap, q, cloth_xy)
            renderer.update_scene(data, camera=LOCAL_CAMERA)
            viewer.set_images(
                (inset_rect(mujoco, viewer.viewport), renderer.render())
            )
            viewer.sync()
    finally:
        renderer.close()
        viewer.close()


def record(payload: dict, output: Path) -> None:
    """Render the deterministic replay and stream RGB frames directly to ffmpeg."""
    source_t = video_source_times(float(payload["duration_s"]))
    positions = sampled_positions(payload, source_t)
    mujoco, model, data, resolver, cloth_mocap, centers, _ = cloth_replay(
        payload, positions
    )
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pixel_format", "rgb24",
        "-video_size", f"{WIDTH}x{HEIGHT}", "-framerate", str(FPS),
        "-i", "-", "-an", "-c:v", "libx264", "-preset", "medium",
        "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(output),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        with (
            mujoco.Renderer(model, height=HEIGHT, width=WIDTH) as global_renderer,
            mujoco.Renderer(
                model, height=INSET_HEIGHT, width=INSET_WIDTH
            ) as local_renderer,
        ):
            for q, cloth_xy in zip(positions, centers):
                _apply_frame(
                    mujoco, model, data, resolver, cloth_mocap, q, cloth_xy
                )
                global_renderer.update_scene(data, camera=GLOBAL_CAMERA)
                local_renderer.update_scene(data, camera=LOCAL_CAMERA)
                frame = compose_picture_in_picture(
                    global_renderer.render(), local_renderer.render()
                )
                assert process.stdin is not None
                process.stdin.write(frame.tobytes())
        assert process.stdin is not None
        process.stdin.close()
        return_code = process.wait()
    except BaseException:
        if process.stdin is not None:
            process.stdin.close()
        process.kill()
        process.wait()
        raise
    if return_code != 0:
        raise RuntimeError(f"ffmpeg failed with exit status {return_code}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay an original OpenArm right-arm motion in the isolated wipe scene."
    )
    parser.add_argument("--trajectory", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preview", action="store_true", help="run once in the GUI")
    mode.add_argument("--output", type=Path, help="render a 1280x720 30 FPS MP4")
    mode.add_argument("--inspect", action="store_true", help="validate and print metadata only")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    payload = load_trajectory(args.trajectory)
    print(json.dumps(trajectory_report(payload), indent=2))
    if args.output:
        record(payload, args.output)
        print(f"video={args.output.resolve()}")
    elif args.preview:
        preview(payload)
        # GLFW teardown segfaults on roughly one run in four on this GL stack,
        # after the replay has already finished.  Skip finalisation instead.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    elif not args.inspect:
        raise SystemExit("choose --preview, --output FILE, or --inspect")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
