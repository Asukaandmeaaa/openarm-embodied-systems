"""Run attended dual-camera episodes from one or more gamepad B presses."""

from __future__ import annotations

import json
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger

from openarm_kinesthetic.recording import allocate_trial_id, valid_trial_id

from .recorded_motion import validate_collection_trajectory
from .capture_preflight import capture_preflight, batch_suffix


class EpisodeCaptureButton(Node):
    def __init__(self) -> None:
        super().__init__("episode_capture_button")
        self.declare_parameter("trajectory_file", "")
        self.declare_parameter("trajectory_directory", "")
        self.declare_parameter("trajectory_pattern", "*.json")
        self.declare_parameter("trajectory_start_number", 1)
        self.declare_parameter("validation_mode", "prior_verified")
        self.declare_parameter("position_id_from_source", False)
        self.declare_parameter("batch_id", "fixed_scene_01")
        self.declare_parameter("position_id", "fixed_01")
        self.declare_parameter("cube_forward_m", 0.0)
        self.declare_parameter("cube_lateral_m", 0.0)
        self.declare_parameter("placement_measurement_method", "fixed_marked_origin")
        self.declare_parameter("plan_only_passed", False)
        self.declare_parameter("dry_replay_passed", False)
        self.declare_parameter("replay_button_index", 1)
        self.declare_parameter("keyframe_settle_tolerance_rad", 0.03)
        self.declare_parameter("allocate_new_episode_ids", False)
        preflight = capture_preflight(
            str(self.get_parameter("validation_mode").value),
            self.get_parameter("plan_only_passed").value,
            self.get_parameter("dry_replay_passed").value,
        )
        self.position_id_from_source = bool(self.get_parameter("position_id_from_source").value)
        directory = str(self.get_parameter("trajectory_directory").value).strip()
        if directory:
            trajectory_dir = Path(directory).expanduser().resolve(strict=True)
            if not trajectory_dir.is_dir() or trajectory_dir.name != "trajectories":
                raise ValueError("trajectory_directory must be collections/trajectories")
            pattern = str(self.get_parameter("trajectory_pattern").value)
            self.trajectories = sorted(trajectory_dir.glob(pattern))
        else:
            self.trajectories = [Path(
                str(self.get_parameter("trajectory_file").value)
            ).expanduser().resolve(strict=True)]
        if not self.trajectories:
            raise ValueError("no numbered trajectories found")
        for trajectory in self.trajectories:
            payload = json.loads(trajectory.read_text(encoding="utf-8"))
            validate_collection_trajectory(payload)
            trial_id = str(payload.get("trial_id", ""))
            if not valid_trial_id(trial_id) or trajectory.stem != trial_id:
                raise ValueError("B capture requires numbered filenames matching trial_id")
            if trajectory.parent.name != "trajectories":
                raise ValueError("numbered trajectory must be in collections/trajectories")
        self.trajectories = batch_suffix(
            self.trajectories, self.get_parameter("trajectory_start_number").value
        )
        self.allocate_new_episode_ids = bool(
            self.get_parameter("allocate_new_episode_ids").value
        )
        self.collection_root = self.trajectories[0].parent.parent
        self.trajectory_index = 0
        self.trajectory = self.trajectories[self.trajectory_index]
        self.source_trial_id = self.trajectory.stem
        self.trial_id = self._episode_trial_id()
        self.button_index = int(self.get_parameter("replay_button_index").value)
        self.keyframe_settle_tolerance_rad = float(
            self.get_parameter("keyframe_settle_tolerance_rad").value
        )
        if not 0.0 < self.keyframe_settle_tolerance_rad <= 0.06:
            raise ValueError("keyframe_settle_tolerance_rad must be in (0, 0.06]")
        self.metadata = {
            "batch_id": str(self.get_parameter("batch_id").value),
            "trial_id": self.trial_id,
            "source_trajectory_trial_id": self.source_trial_id,
            "position_id": (f"pose_{self.source_trial_id}" if self.position_id_from_source
                            else str(self.get_parameter("position_id").value)),
            "cube_forward_m": float(self.get_parameter("cube_forward_m").value),
            "cube_lateral_m": float(self.get_parameter("cube_lateral_m").value),
            "placement_measurement_method": str(
                self.get_parameter("placement_measurement_method").value
            ),
            **preflight,
        }
        self.stage = "READY"
        self.deadline = 0.0
        self.last_publish = 0.0
        self.latest_status: dict = {}
        self.latest_status_at = 0.0
        self.gamepad_status: dict = {}
        self.gamepad_status_at = 0.0
        self.start_future = None
        self.auto_finish_future = None
        self.player: subprocess.Popen | None = None
        self.metadata_pub = self.create_publisher(
            String, "/openarm_kinesthetic/trial_metadata", 10
        )
        self.start_client = self.create_client(Trigger, "/real_episode_recorder/start")
        self.auto_finish_client = self.create_client(
            Trigger, "/real_episode_recorder/finish_auto_review"
        )
        self.create_subscription(
            String, "/openarm_kinesthetic/gamepad_events", self.on_gamepad_events, 10
        )
        self.create_subscription(
            String, "/openarm_kinesthetic/gamepad_status", self.on_gamepad_status, 10
        )
        self.create_subscription(
            String, "/openarm_kinesthetic/recording_status", self.on_recording_status, 10
        )
        self.create_timer(0.1, self.on_timer)
        self.get_logger().warn(
            f"B_CAPTURE_READY trial_id={self.trial_id} "
            f"source_trial_id={self.source_trial_id} trajectory={self.trajectory} "
            f"({self.trajectory_index + 1}/{len(self.trajectories)}) "
            f"button_index={self.button_index}; place cube, release LT/LB, then press B once"
        )

    def _episode_trial_id(self) -> str:
        if not self.allocate_new_episode_ids:
            return self.trajectory.stem
        return allocate_trial_id(self.collection_root)

    def select_trajectory(self, index: int) -> None:
        self.trajectory_index = index
        self.trajectory = self.trajectories[index]
        self.source_trial_id = self.trajectory.stem
        self.trial_id = self._episode_trial_id()
        self.metadata["trial_id"] = self.trial_id
        self.metadata["source_trajectory_trial_id"] = self.source_trial_id
        if getattr(self, "position_id_from_source", False):
            self.metadata["position_id"] = f"pose_{self.source_trial_id}"

    def on_gamepad_status(self, message: String) -> None:
        try:
            self.gamepad_status = json.loads(message.data)
            self.gamepad_status_at = time.monotonic()
        except json.JSONDecodeError:
            return

    def on_recording_status(self, message: String) -> None:
        try:
            self.latest_status = json.loads(message.data)
            self.latest_status_at = time.monotonic()
        except json.JSONDecodeError:
            return

    def on_gamepad_events(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
            pressed = any(
                event.get("kind") == "button"
                and event.get("index") == self.button_index
                and event.get("value") == 1
                and event.get("initial") is False
                for event in payload.get("events", [])
            )
        except (AttributeError, TypeError, json.JSONDecodeError):
            return
        if not pressed:
            return
        if self.stage != "READY":
            self.get_logger().warn(f"B_CAPTURE_IGNORED stage={self.stage}")
            return
        now = time.monotonic()
        pad = self.gamepad_status
        status = self.latest_status
        nodes = self.get_node_names()
        if (
            now - self.gamepad_status_at > 1.0
            or pad.get("connected") is not True
            or pad.get("stop_latched") is not False
            or pad.get("control_mode") != "idle"
            or now - self.latest_status_at > 1.5
            or status.get("active") is not False
            or status.get("sources_ready") is not True
            or Path(str(status.get("output_root", ""))).expanduser() != self.collection_root
            or "right_kinesthetic_bridge" in nodes
            or "right_motion_recorder" in nodes
            or "recorded_motion_player" in nodes
            or nodes.count("episode_capture_button") != 1
            or nodes.count("real_episode_recorder") != 1
            or not self.start_client.service_is_ready()
        ):
            self.get_logger().error(
                "B_CAPTURE_REJECTED: gamepad, cameras, recorder, or controller ownership not ready"
            )
            return
        self.stage = "METADATA"
        self.deadline = now + 5.0
        self.last_publish = 0.0
        self.get_logger().warn(f"B_CAPTURE_REQUESTED trial_id={self.trial_id}")

    def on_timer(self) -> None:
        now = time.monotonic()
        if self.stage == "METADATA":
            if self.latest_status.get("pending_trial_id") == self.trial_id:
                self.start_future = self.start_client.call_async(Trigger.Request())
                self.deadline = now + 5.0
                self.stage = "STARTING"
                return
            if now > self.deadline:
                self.stage = "READY"
                self.get_logger().error("B_CAPTURE_REJECTED: trial metadata was not acknowledged")
                return
            if now - self.last_publish >= 0.5:
                self.metadata_pub.publish(String(data=json.dumps(self.metadata)))
                self.last_publish = now
        elif self.stage == "STARTING":
            if self.start_future is not None and self.start_future.done():
                try:
                    response = self.start_future.result()
                except Exception as error:
                    self.stage = "READY"
                    self.get_logger().error(f"B_CAPTURE_REJECTED: recorder start error: {error}")
                    return
                if response is None or not response.success:
                    self.stage = "READY"
                    self.get_logger().error(
                        f"B_CAPTURE_REJECTED: {response.message if response else 'no response'}"
                    )
                    return
                self.get_logger().warn(f"B_CAPTURE_STARTED {response.message}")
                command = [
                    "ros2", "run", "openarm_kinesthetic_recorder", "recorded_motion_player",
                    "--trajectory", str(self.trajectory), "--allow-real-execution",
                    "--playback-rate", "1.0",
                    "--keyframe-settle-tolerance-rad",
                    str(self.keyframe_settle_tolerance_rad),
                ]
                try:
                    self.player = subprocess.Popen(command, start_new_session=True)
                except OSError as error:
                    self.stage = "FAILED"
                    self.get_logger().error(
                        f"B_CAPTURE_PLAYER_LAUNCH_FAILED: {error}; call finish_failure"
                    )
                    return
                self.stage = "REPLAYING"
            elif now > self.deadline:
                self.stage = "FAILED"
                self.get_logger().error("B_CAPTURE_START_TIMEOUT: inspect recorder before any replay")
        elif self.stage == "REPLAYING":
            result = self.player.poll() if self.player else None
            if result is None:
                return
            if result != 0:
                self.stage = "FAILED"
                self.get_logger().error(
                    f"B_CAPTURE_REPLAY_FAILED exit_code={result}; inspect failed episode"
                )
                return
            self.get_logger().warn("B_CAPTURE_REPLAY_COMPLETE: committing raw data pending human review")
            if not self.auto_finish_client.service_is_ready():
                self.stage = "FAILED"
                self.get_logger().error(
                    "B_CAPTURE_AUTO_SAVE_UNAVAILABLE: recorder remains active; inspect before recovery"
                )
                return
            self.auto_finish_future = self.auto_finish_client.call_async(Trigger.Request())
            self.deadline = now + 10.0
            self.stage = "SAVING"
        elif self.stage == "SAVING":
            if self.auto_finish_future is not None and self.auto_finish_future.done():
                try:
                    response = self.auto_finish_future.result()
                except Exception as error:
                    self.stage = "FAILED"
                    self.get_logger().error(f"B_CAPTURE_AUTO_SAVE_FAILED: {error}")
                    return
                if response is None or not response.success:
                    self.stage = "FAILED"
                    self.get_logger().error(
                        "B_CAPTURE_AUTO_SAVE_FAILED: "
                        + (response.message if response else "no response")
                    )
                    return
                self.get_logger().warn(
                    f"B_CAPTURE_SAVED_PENDING_REVIEW {response.message}"
                )
                if not hasattr(self, "trajectories"):
                    self.stage = "SAVED"
                    return
                next_index = self.trajectory_index + 1
                if next_index >= len(self.trajectories):
                    self.stage = "COMPLETE"
                    self.get_logger().warn(
                        "B_CAPTURE_BATCH_COMPLETE: all trajectories saved; no further B accepted"
                    )
                else:
                    self.select_trajectory(next_index)
                    self.stage = "READY"
                    self.get_logger().warn(
                        f"B_CAPTURE_READY trial_id={self.trial_id} "
                        f"source_trial_id={self.source_trial_id} trajectory={self.trajectory} "
                        f"({self.trajectory_index + 1}/{len(self.trajectories)}); "
                        "reset cube and press B once"
                    )
            elif now > self.deadline:
                self.stage = "FAILED"
                self.get_logger().error(
                    "B_CAPTURE_AUTO_SAVE_TIMEOUT: inspect recorder status before any new trial"
                )


def acquire_capture_lock():
    directory = Path(os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}"))
    domain = int(os.environ.get("ROS_DOMAIN_ID", "0"))
    lock = (directory / f"openarm-b-capture-{domain}.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise RuntimeError("another B capture process is already running")
    return lock


def stop_owned_player(player, logger):
    if player is None or player.poll() is not None:
        return
    logger.warn("B_CAPTURE_CANCEL_REQUESTED: stopping active replay")
    try:
        os.killpg(player.pid, signal.SIGINT)
    except ProcessLookupError:
        pass
    while player.poll() is None:
        try:
            player.wait(timeout=5.0)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            logger.error(
                "B_CAPTURE_WAITING_FOR_PLAYER_EXIT: cancellation not yet complete; "
                "supervisor stays here, keep H1; use physical stop if motion persists"
            )


def main() -> int:
    capture_lock = acquire_capture_lock()
    rclpy.init()
    node = EpisodeCaptureButton()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        stop_owned_player(node.player, node.get_logger())
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        capture_lock.close()
    return 0
