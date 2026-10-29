"""Closed-loop simulation evaluation on fixed scenes.

Each scene is advanced only by MuJoCo contact dynamics.  The policy predicts a
full chunk (100 for the candidate, 50 for the 60K baseline). The first
``--exec-steps`` targets on the 30 Hz grid use the deployment's official Hermite
upsampling and 250 Hz/15 Hz lowpass, then a fresh observation is captured.
Physics holds the last target during measured render/inference latency.
Nothing teleports the cube or writes qpos during a rollout.

This is simulation task success, not physical success.  The 60K baseline here
executes its full 50-step chunk; it is not the original official_blend runtime,
so comparisons describe the whole changed scheme, not one isolated factor.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from .sim_reference import BiquadLowpass, prepare_chunk
from .sim_collect import write_json
from .sim_expert.world import SimWorld, load_config
from .metrics import difference_metrics, distribution, recovery_success
from .paths import recorded_path
from .policy_limits import DeploymentEnvelope, DeploymentPolicyWorld, range_diagnostics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FPS = 30


class Runner:
    def __init__(self, model: Path):
        from lerobot.configs import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors

        self.model = Path(model)
        cfg = PreTrainedConfig.from_pretrained(self.model)
        cfg.pretrained_path = self.model
        cfg.device = "cuda"
        self.policy = get_policy_class(cfg.type).from_pretrained(
            config=cfg, pretrained_name_or_path=self.model
        )
        self.pre, self.post = make_pre_post_processors(
            cfg,
            pretrained_path=self.model,
            preprocessor_overrides={"device_processor": {"device": "cuda"}},
        )
        self.chunk_size = int(cfg.chunk_size)
        self.policy.eval()

    def reset(self) -> None:
        self.policy.reset()
        self.pre.reset()
        self.post.reset()

    def predict(self, images: dict[str, np.ndarray], state: np.ndarray) -> tuple[np.ndarray, float]:
        import torch
        batch = {
            "observation.state": torch.from_numpy(state.astype(np.float32)),
            "observation.images.ceiling": torch.from_numpy(images["ceiling"]).permute(2, 0, 1).float() / 255,
            "observation.images.wrist_right": torch.from_numpy(images["wrist_right"]).permute(2, 0, 1).float() / 255,
        }
        began = time.monotonic()
        with torch.inference_mode():
            proc = self.pre(batch)
            out = self.policy.predict_action_chunk(proc)
            out = self.post(out.reshape(-1, out.shape[-1])).reshape(out.shape)
        return out[0].cpu().numpy(), time.monotonic() - began


def set_recovery_start(world: SimWorld, collection: Path, scene: dict) -> None:
    root = recorded_path(scene.get("collection_root", collection))
    with np.load(root / scene["source_episode"] / "numeric.npz") as z:
        qpos = z["qpos"][scene["takeover_frame"]].copy()
        qvel = z["qvel"][scene["takeover_frame"]].copy() if "qvel" in z.files else np.zeros(world.model.nv)
        command = z["action"][scene["takeover_frame"]].copy()
    world.data.qpos[:] = qpos
    world.data.qvel[:] = qvel
    world.data.time = 0.0
    import mujoco

    mujoco.mj_forward(world.model, world.data)
    world.nominal_cube_xyz = world.cube_xyz()
    state = world.state16()
    world.reset_runtime()
    world.last_q = command[:8].copy()
    world.last_command16 = command.copy()
    world.last_v = np.zeros(8)
    world.mapper.set_ctrl(world.data.ctrl, command[:8], "right")
    world.mapper.set_ctrl(world.data.ctrl, command[8:], "left")


def _cube_speeds(samples: list[dict]) -> np.ndarray:
    if not samples:
        return np.zeros(0)
    if "cube_speed" in samples[0]:
        return np.asarray([float(s["cube_speed"]) for s in samples])
    cube = np.asarray([s["cube"] for s in samples], dtype=float)
    times = np.asarray([s["time"] for s in samples], dtype=float)
    out = np.zeros(len(cube))
    for i in range(1, len(cube)):
        dt = max(float(times[i] - times[i - 1]), 1e-9)
        out[i] = float(np.linalg.norm(cube[i] - cube[i - 1]) / dt)
    return out


def classify_trajectory(
    samples: list[dict],
    *,
    nominal_cube_z: float,
    box_center: np.ndarray,
    box_rotation: np.ndarray,
    cfg: dict,
    recovery_type: str | None,
    forbidden_contacts: list,
) -> dict:
    """Separate a transit drop from a planned release inside the box."""
    if not samples:
        return {
            "success": False,
            "failure_type": "empty",
            "transit_drop": False,
            "inside": False,
            "supported": False,
            "settled": False,
            "forbidden_contacts": forbidden_contacts[:3],
        }
    cube = np.asarray([s["cube"] for s in samples], dtype=float)
    times = np.asarray([s["time"] for s in samples], dtype=float)
    bilateral = np.asarray([bool(s["bilateral"]) for s in samples])
    released = np.asarray([s.get("finger_count", 2 if s["bilateral"] else 0) == 0 for s in samples])
    speeds = _cube_speeds(samples)
    if not np.isfinite(cube).all() or not np.isfinite(speeds).all() or np.any(np.diff(times) <= 0):
        return {"success": False, "failure_type": "invalid_trajectory", "transit_drop": False}
    rotation = np.asarray(box_rotation, dtype=float)
    center = np.asarray(box_center, dtype=float)
    local = (cube - center) @ rotation
    half = np.asarray([s.get("cube_half_box", [0.02, 0.02, 0.02]) for s in samples])
    fully_over_box = np.all(np.abs(local[:, :2]) + half[:, :2] <= 0.06, axis=1)
    over_opening = np.all(np.abs(local[:, :2]) <= 0.06, axis=1)
    supported = np.abs(local[:, 2] - half[:, 2] - 0.01) < 0.002
    rel = cube[:, 2] - float(nominal_cube_z)
    peak = float(rel.max())
    pre_release = recovery_type == "pre_release_hold"
    threshold = float(cfg.get("hold_height_m", 0.05))
    # A release over the opening is allowed; a loss outside it after lifting
    # remains a failure even if the object later happens to bounce into the box.
    lifted = np.maximum.accumulate(rel) >= threshold
    falling = rel < np.maximum.accumulate(rel) - 0.02
    transit_drop = bool(np.any(lifted & falling & (~bilateral) & (~over_opening)))
    held = bilateral & (rel >= threshold)
    longest_hold = 0.0
    held_start = None
    for i, flag in enumerate(held):
        if flag and held_start is None:
            held_start = times[i]
        if flag:
            longest_hold = max(longest_hold, float(times[i] - held_start))
        else:
            held_start = None
    hold_ok = pre_release or longest_hold >= float(cfg.get("hold_s", 2.0)) - 1e-9
    good = fully_over_box & supported & released & (speeds < 0.01)
    last_bad = np.flatnonzero(~good)
    start = int(last_bad[-1] + 1) if len(last_bad) else 0
    stable_s = float(times[-1] - times[start]) if start < len(times) else 0.0
    settled = bool(good[-1] and stable_s >= float(cfg.get("placement_stable_s", 0.5)) - 1e-9)
    collision = bool(forbidden_contacts)
    success = bool(settled and hold_ok and not collision and not transit_drop)
    if collision:
        failure = "forbidden_contact"
    elif transit_drop:
        failure = "transit_drop"
    elif not hold_ok:
        failure = "no_lift" if peak < threshold else "insufficient_hold"
    elif not released[-1]:
        failure = "not_released"
    elif not fully_over_box[-1]:
        failure = "miss_box"
    elif not settled:
        failure = "not_settled"
    else:
        failure = None
    return {
        "success": success, "failure_type": failure,
        "transit_drop": transit_drop, "release_outside": transit_drop,
        "release_in_box": bool(np.any(released & fully_over_box)),
        "inside": bool(fully_over_box[-1]), "supported": bool(supported[-1]),
        "settled": settled, "settled_duration_s": stable_s,
        "released": bool(released[-1]), "peak_lift_m": peak,
        "bilateral_hold_s": longest_hold,
        "final_cube": cube[-1].tolist(), "final_cube_speed_m_s": float(speeds[-1]),
        "forbidden_contacts": forbidden_contacts[:3],
    }


def task_result(world: SimWorld, *, recovery_type: str | None) -> dict:
    return classify_trajectory(
        world.samples,
        nominal_cube_z=float(world.nominal_cube_xyz[2]),
        box_center=world.data.xpos[world.box].copy(),
        box_rotation=world.data.xmat[world.box].reshape(3, 3).copy(),
        cfg=world.cfg,
        recovery_type=recovery_type,
        forbidden_contacts=world.forbidden_contacts,
    )


class _Rows:
    def __init__(self, rows):
        self.rows = rows

    def to_pylist(self):
        return self.rows


def advance_until(world, deadline: float, command: np.ndarray, phase: str) -> None:
    """Advance fixed-step physics to an absolute deadline (error < one physics tick)."""
    while float(world.data.time) + 1e-12 < deadline:
        world.step(command, phase)


def execute_block(world, prediction: np.ndarray, steps: int, lowpass, *, reset: bool, left_hold: np.ndarray, envelope=None) -> dict:
    event = {"value": _Rows(prediction.tolist()),
             "metadata": {"interval": 33_333_333, "reset": reset}}
    offsets, values, duration_ns = prepare_chunk(event, steps, len(prediction))
    if reset:
        lowpass.reset_state(values[0])
    began = float(world.data.time)
    feedback = world.state16().copy()
    tracking = []
    first_filtered = None
    first_commanded = None
    clipped_count = 0
    clipped_axes = np.zeros(8, dtype=int)
    max_saturation = np.zeros(8)
    saturation_examples = []
    for i, value in enumerate(values):
        requested = lowpass.step(value)
        # Physical ACT drives the right arm and holds the measured left pose.
        requested[8:] = left_hold
        command = requested.copy()
        if envelope is not None:
            command, axes = envelope.apply(requested)
            if len(axes):
                clipped_count += 1
                clipped_axes[axes] += 1
                max_saturation = np.maximum(max_saturation, np.abs(command[:8] - requested[:8]))
                if len(saturation_examples) < 5:
                    saturation_examples.append({'sample': i, 'right_axes_1based': (axes + 1).tolist(),
                        'requested_right': requested[:8].tolist(), 'commanded_right': command[:8].tolist()})
        if first_filtered is None:
            first_filtered = requested.copy()
            first_commanded = command.copy()
        try:
            world.validate_command(command)
            end_ns = offsets[i + 1] if i + 1 < len(offsets) else duration_ns
            advance_until(world, began + int(end_ns) / 1e9, command, "POLICY")
        except Exception as exc:
            # Retain the failed block rather than silently dropping its targets
            # and saturation evidence from the report.
            exc.block_diagnostics = {
                'sample_index': i, 'offset_s': int(offsets[i]) / 1e9,
                'completed_reference_slots': len(tracking),
                'actual_partial_duration_s': float(world.data.time) - began,
                'requested_filtered_right': requested[:8].tolist(),
                'commanded_right': command[:8].tolist(),
                'feedback_right': world.state16()[:8].tolist(),
                'raw_prediction_committed_right': prediction[:steps, :8].tolist(),
                'clipped_actions': clipped_count,
                'clipped_counts_by_right_axis': clipped_axes.tolist(),
                'max_saturation_rad_by_right_axis': max_saturation.tolist(),
                'saturation_examples': saturation_examples,
            }
            raise
        tracking.append(world.state16()[:8] - command[:8])
    return {"predicted": len(prediction), "applied": steps, "first_index": 0,
            "last_index": steps - 1, "block_duration_s": duration_ns / 1e9,
            "actual_block_duration_s": float(world.data.time) - began,
            "reference_count": len(values), "clipped_actions": clipped_count,
            "clipped_counts_by_right_axis": clipped_axes.tolist(),
            "max_saturation_rad_by_right_axis": max_saturation.tolist(),
            "saturation_examples": saturation_examples,
            "raw_first_to_feedback": difference_metrics((prediction[0, :8] - feedback[:8])[None]),
            "first_filtered_to_feedback": difference_metrics((first_filtered[:8] - feedback[:8])[None]),
            "first_commanded_to_feedback": difference_metrics((first_commanded[:8] - feedback[:8])[None]),
            "tracking": difference_metrics(np.asarray(tracking)),
            "tracking_sampling": "end of each 250Hz command slot; MuJoCo feedback minus filtered reference",
            "first_raw_right": prediction[0, :8].tolist(),
            "first_filtered_right": first_filtered[:8].tolist(),
            "last_filtered_right": requested[:8].tolist(),
            "first_commanded_right": first_commanded[:8].tolist(),
            "last_commanded_right": command[:8].tolist()}


def run_scene(runner: Runner, cfg: dict, scene: dict, collection: Path, args) -> dict:
    profile = getattr(args, 'command_envelope', 'strict_expert')
    envelope = DeploymentEnvelope() if profile == 'deployment' else None
    world = DeploymentPolicyWorld(scene['seed'], cfg, jitter=scene.get('jitter'), envelope=envelope) if envelope else SimWorld(scene["seed"], cfg, jitter=scene.get("jitter"))
    try:
        if scene["kind"] == "recovery":
            set_recovery_start(world, collection, scene)
        elif scene.get("start", "deployment_zero") == "deployment_zero":
            q = np.r_[np.zeros(7), -0.02]
            world.mapper.set_qpos(world.data.qpos, q, "right")
            world.mapper.set_qpos(world.data.qpos, q, "left")
            world.mapper.set_ctrl(world.data.ctrl, q, "right")
            world.mapper.set_ctrl(world.data.ctrl, q, "left")
            import mujoco
            mujoco.mj_forward(world.model, world.data)
            world.home_driver = np.r_[q, q]
            world.reset_runtime()
        left_hold = world.state16()[8:].copy()
        runner.reset()
        chunks = []
        failure = None
        failed_block = None
        began = time.monotonic()
        lowpass = BiquadLowpass(fs=250, fc=15)
        try:
            for chunk_index in range(args.max_chunks):
                # Explicit approximation to the post-completion fresh-camera wait.
                world.hold_for(args.observation_wait_s, "OBSERVATION_HOLD")
                capture_begin = time.monotonic()
                state = world.state16().copy()
                observation_sim_time = float(world.data.time)
                images = world.render_frames()
                prediction, inference_s = runner.predict(images, state)
                perception_s = time.monotonic() - capture_begin
                world.hold_for(perception_s, "INFERENCE_HOLD")
                if prediction.shape != (runner.chunk_size, 16) or not np.isfinite(prediction).all():
                    raise RuntimeError("nonfinite or wrong-shaped prediction")
                result = task_result(world, recovery_type=scene.get("recovery_type"))
                if result["success"]:
                    break
                if world.data.time >= args.max_seconds:
                    failure = "timeout"
                    break
                block = execute_block(world, prediction, args.exec_steps, lowpass,
                                      reset=chunk_index == 0, left_hold=left_hold, envelope=envelope)
                block['raw_command_ranges'] = range_diagnostics(prediction, args.exec_steps, cfg, envelope)
                if chunks:
                    previous = np.asarray(chunks[-1]["last_filtered_right"])
                    block["first_filtered_to_previous_reference"] = difference_metrics(
                        (np.asarray(block["first_filtered_right"]) - previous)[None])
                    block["raw_first_to_previous_committed_last"] = difference_metrics(
                        (prediction[0, :8] - np.asarray(chunks[-1]["last_committed_right"]))[None])
                    block['commanded_to_previous_commanded'] = difference_metrics(
                        (np.asarray(block['first_commanded_right']) - np.asarray(chunks[-1]['last_commanded_right']))[None])
                else:
                    block["first_filtered_to_previous_reference"] = None
                    block["raw_first_to_previous_committed_last"] = None
                    block['commanded_to_previous_commanded'] = None
                block["last_committed_right"] = prediction[args.exec_steps - 1, :8].tolist()
                block["observation_to_first_reference_sim_ms"] = (
                    float(world.data.time) - block["actual_block_duration_s"] - observation_sim_time) * 1000
                chunks.append({**block, "chunk": chunk_index,
                               "inference_s": inference_s, "inference_hold_s": perception_s,
                               "render_and_preprocess_overhead_s": max(0., perception_s - inference_s),
                               "observation_wait_s": args.observation_wait_s,
                               "sim_time_s": float(world.data.time), "cube": world.cube_xyz().tolist()})
                result = task_result(world, recovery_type=scene.get("recovery_type"))
                if result["success"]:
                    break
                if world.data.time >= args.max_seconds:
                    failure = "timeout"
                    break
            else:
                failure = "chunk_budget"
        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"
            failed_block = getattr(exc, 'block_diagnostics', None)
            if failed_block is not None:
                failed_block.update(chunk=chunk_index, command_envelope=profile,
                    raw_command_ranges=range_diagnostics(prediction, args.exec_steps, cfg, envelope))
        result = task_result(world, recovery_type=scene.get("recovery_type"))
        if failure and failure not in ("timeout", "chunk_budget"):
            result.update(success=False, failure_type=failure)
        return {"scene": scene, "success": bool(result["success"]),
                "failure_type": result["failure_type"] if not result["success"] else None,
                "termination": failure or ("success" if result["success"] else "ended"),
                "chunks": chunks, "failed_block": failed_block,
                "clip_count": sum(c['clipped_actions'] for c in chunks) + (failed_block['clipped_actions'] if failed_block else 0),
                "command_envelope": profile,
                "wall_s": time.monotonic() - began, "result": result,
                "recovery_velocity_source": scene.get("velocity_source")}
    finally:
        world.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--scenes", type=Path, required=True)
    parser.add_argument("--collection", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exec-steps", type=int, default=50)
    parser.add_argument('--command-envelope', choices=('strict_expert', 'deployment'), default='strict_expert',
                        help='Explicit comparison: strict teacher/planner guards, or frozen deployed endpoint saturation after filtering')
    parser.add_argument("--max-seconds", type=float, default=150.0)
    parser.add_argument("--max-chunks", type=int, default=120)
    parser.add_argument("--observation-wait-s", type=float, default=1 / 30,
                        help="Explicit fresh-observation wait approximation between blocks")
    args = parser.parse_args()
    if args.max_seconds <= 0 or args.max_chunks < 1 or args.exec_steps < 2 or args.observation_wait_s < 0:
        parser.error("invalid execution budget or observation wait")
    if args.output.exists():
        raise SystemExit("output exists; choose a new directory")
    args.output.mkdir(parents=True)
    scenes = json.loads(args.scenes.read_text())
    if not scenes.get("fixed") or scenes["count"] != len(scenes.get("scenes", [])) or not scenes.get("scenes"):
        raise SystemExit("nonempty fixed scene file with matching count is required")
    from .prepare_eval_scenes import verify_scene_file
    verify_scene_file(scenes)
    runner = Runner(args.model.resolve())
    if runner.chunk_size < args.exec_steps:
        raise SystemExit("exec_steps exceeds model chunk size")
    cfg = load_config()
    rows = []
    for scene in scenes["scenes"]:
        print("SCENE", scene["id"], flush=True)
        row = run_scene(runner, cfg, scene, args.collection.resolve(), args)
        rows.append(row)
        print(
            json.dumps(
                {
                    "id": scene["id"],
                    "success": row["success"],
                    "failure": row["failure_type"],
                    "chunks": len(row["chunks"]),
                }
            ),
            flush=True,
        )
        with (args.output / "scenes.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
    summary = {
        "model": str(args.model.resolve()),
        "chunk_size": runner.chunk_size,
        "exec_steps": args.exec_steps,
        "command_envelope": args.command_envelope,
        "deployment_position_snapshot": DeploymentEnvelope().snapshot if args.command_envelope == 'deployment' else None,
        "scene_file_sha256": hashlib.sha256(args.scenes.read_bytes()).hexdigest(),
        "runtime_sha256": scenes["runtime_sha256"],
        "scenes": len(rows),
        "success": int(sum(r["success"] for r in rows)),
        "success_rate": sum(r["success"] for r in rows) / len(rows),
        "by_kind": {},
        "by_recovery_type": recovery_success(rows),
        "failure_types": {},
        "clipped_actions": int(sum(r["clip_count"] for r in rows)),
        "execution": {"interpolation": "official Hermite via prepare_chunk",
                      "filter": "official BiquadLowpass 250Hz/15Hz; persists across blocks",
                      "inference_wait": "physics holds last filtered target for measured render+inference wall time",
                      "observation_wait_s": args.observation_wait_s,
                      "left_arm": "hold initial feedback",
                      "out_of_range": "strict teacher guards, fail" if args.command_envelope == 'strict_expert' else 'explicit logged deployed position saturation after filter; unchanged collision failures'},
        "limitations": ["Ideal simulation gravity compensation and uncalibrated gains/contact/cameras",
                        "No CAN/Dora transport, SDK quantization, or motor dynamics equivalence",
                        "Camera freshness wait is configured, not a measured sensor pipeline"],
        "hardware_accessed": False,
        "physical_success_verified": False,
        "note": (
            "Closed-loop simulation with contact dynamics. The 60K baseline "
            "executes a full 50-step chunk, not official_blend."
        ),
    }
    for kind in ("home", "recovery"):
        subset = [r for r in rows if r["scene"]["kind"] == kind]
        summary["by_kind"][kind] = {
            "scenes": len(subset),
            "success": sum(r["success"] for r in subset),
            "success_rate": sum(r["success"] for r in subset) / len(subset) if subset else None,
            "failure_types": {},
        }
        for r in subset:
            if not r["success"]:
                key = r["failure_type"] or "unknown"
                summary["by_kind"][kind]["failure_types"][key] = (
                    summary["by_kind"][kind]["failure_types"].get(key, 0) + 1
                )
    for r in rows:
        if not r["success"]:
            key = r["failure_type"] or "unknown"
            summary["failure_types"][key] = summary["failure_types"].get(key, 0) + 1
    summary["by_split"] = {}
    for split in sorted({r["scene"]["split"] for r in rows}):
        selected = [r for r in rows if r["scene"]["split"] == split]
        summary["by_split"][split] = {"scenes": len(selected),
                                      "success": sum(r["success"] for r in selected),
                                      "success_rate": sum(r["success"] for r in selected) / len(selected)}
        summary["by_split"][split]["by_recovery_type"] = recovery_success(selected)
    all_chunks = [c for row in rows for c in row["chunks"]]
    partial_blocks = [row['failed_block'] for row in rows if row['failed_block'] is not None]
    saturation_records = all_chunks + partial_blocks
    summary['position_saturation'] = {
        'completed_blocks': len(all_chunks), 'failed_blocks_with_targets': len(partial_blocks),
        'clipped_reference_attempts': sum(c['clipped_actions'] for c in saturation_records),
        'clipped_counts_by_right_axis': np.sum([c['clipped_counts_by_right_axis'] for c in saturation_records], axis=0).astype(int).tolist() if saturation_records else [0] * 8,
        'max_saturation_rad_by_right_axis': np.max([c['max_saturation_rad_by_right_axis'] for c in saturation_records], axis=0).tolist() if saturation_records else [0.] * 8,
        'note': 'Includes attempted reference in failed block. Raw, filtered-requested and commanded values remain distinct; saturation is not a learned correction.'}
    summary["diagnostics"] = {
        "inference_ms": distribution([c["inference_s"] * 1000 for c in all_chunks]),
        "render_plus_inference_ms": distribution([c["inference_hold_s"] * 1000 for c in all_chunks]),
        "observation_to_first_reference_sim_ms": distribution([c["observation_to_first_reference_sim_ms"] for c in all_chunks]),
        "raw_first_to_feedback_max_arm_deg": distribution([c["raw_first_to_feedback"]["arm_max_axis_error_deg"]["max"] for c in all_chunks]),
        "filtered_boundary_max_arm_deg": distribution([c["first_filtered_to_previous_reference"]["arm_max_axis_error_deg"]["max"]
                                                      for c in all_chunks if c["first_filtered_to_previous_reference"]]),
        "note": "Per-block tracking and raw/filtered jumps in scenes.jsonl. Simulation timing includes modeled holds; it is not measured hardware latency."}
    write_json(args.output / "report.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
