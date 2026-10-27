"""Crash-safe episode transactions with success/failure/cancel isolation."""

from __future__ import annotations

import json
import math
import os
import shutil
import uuid
import hashlib
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


class RecorderStateError(RuntimeError):
    pass


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path.name}:{line_number} is not an object")
            rows.append(row)
    return rows


def audit_real_episode_directory(
    episode: Path,
    expected_joint_names: list[str],
    *,
    maximum_camera_skew_s: float = 0.05,
) -> dict[str, Any]:
    """Audit raw files independently before a real episode is committed."""
    episode = Path(episode)
    failures: list[str] = []
    streams: dict[str, list[dict[str, Any]]] = {}
    try:
        metadata = json.loads((episode / "metadata.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return {"passed": False, "failures": [f"metadata invalid: {error}"]}

    for key in (
        "batch_id", "trial_id", "position_id", "cube_forward_m", "cube_lateral_m",
        "placement_measurement_method", "trajectory_path", "trajectory_sha256",
        "marker_summary", "playback_rate",
    ):
        if key not in metadata:
            failures.append(f"metadata missing {key}")
    if metadata.get("source") != "real_right_continuous_marked_replay":
        failures.append("metadata source is not continuous_marked replay")
    if metadata.get("replay_complete") is not True or metadata.get("full_replay") is not True:
        failures.append("metadata does not prove a complete replay")
    if metadata.get("playback_rate") != 1.0:
        failures.append("metadata playback_rate is not 1.0")
    markers = metadata.get("marker_summary")
    if (
        not isinstance(markers, list)
        or len(markers) != 2
        or not all(isinstance(item, dict) for item in markers)
        or [
        (item.get("type"), item.get("source_button"), item.get("result"))
        for item in markers
        ] != [
            ("gripper_close", "Y", "semantic_only"),
            ("gripper_open", "A", "semantic_only"),
        ]
    ):
        failures.append("metadata marker summary is not strict close/open")

    digest = metadata.get("trajectory_sha256")
    trajectory = Path(str(metadata.get("trajectory_path", "")))
    if not isinstance(digest, str) or len(digest) != 64:
        failures.append("trajectory SHA-256 is invalid")
    elif not trajectory.is_file():
        failures.append("trajectory file is unavailable during commit audit")
    else:
        actual_digest = hashlib.sha256(trajectory.read_bytes()).hexdigest()
        if actual_digest != digest:
            failures.append("trajectory file SHA-256 changed during episode")

    for stream_name in (
        "joint_observation", "demonstration_action", "gripper_event", "camera"
    ):
        path = episode / f"{stream_name}.jsonl"
        try:
            rows = _read_jsonl(path)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            failures.append(f"{stream_name} stream invalid: {error}")
            rows = []
        if not rows:
            failures.append(f"{stream_name} stream is empty")
        if stream_name in ("joint_observation", "demonstration_action") and len(rows) < 2:
            failures.append(f"{stream_name} stream has fewer than two rows")
        streams[stream_name] = rows

    stream_times: dict[str, list[float]] = {}
    for stream_name in ("joint_observation", "demonstration_action", "gripper_event"):
        timestamps = []
        for row in streams[stream_name]:
            value = row.get("timestamp_s")
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                failures.append(f"{stream_name} has invalid timestamp")
                timestamps = []
                break
            timestamps.append(float(value))
        if timestamps and any(b <= a for a, b in zip(timestamps, timestamps[1:])):
            failures.append(f"{stream_name} timestamps are not strictly increasing")
            timestamps = []
        stream_times[stream_name] = timestamps

    for stream_name in ("joint_observation", "demonstration_action"):
        for index, row in enumerate(streams[stream_name]):
            positions = row.get("positions")
            if row.get("order") != expected_joint_names or not isinstance(positions, list) or len(positions) != 16:
                failures.append(f"{stream_name} row {index} violates the 16D joint contract")
                break
            if not all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                for value in positions
            ):
                failures.append(f"{stream_name} row {index} contains nonfinite positions")
                break
            if (
                stream_name == "demonstration_action"
                and row.get("source")
                != "recorded_motion_player_controller_reference"
            ):
                failures.append(
                    f"{stream_name} row {index} has the wrong action source"
                )
                break

    events = [row.get("event") for row in streams["gripper_event"]]
    if events != ["close", "open"]:
        failures.append(f"gripper events are not exactly close/open: {events}")

    if streams["joint_observation"]:
        home = metadata.get("home_initialization", {}).get(
            "home_goal_positions_rad"
        )
        terminal = streams["joint_observation"][-1].get("positions")
        if not isinstance(home, list) or len(home) != 7:
            failures.append("metadata has no seven-joint HOME target")
        elif isinstance(terminal, list) and len(terminal) == 16:
            terminal_errors = [
                abs(float(actual) - float(target))
                for actual, target in zip(terminal[:7], home)
            ]
            tolerance = metadata.get("episode_end_home_tolerance_rad")
            if (
                isinstance(tolerance, bool)
                or not isinstance(tolerance, (int, float))
                or max(terminal_errors) > float(tolerance)
            ):
                failures.append("last measured observation is outside HOME tolerance")
            declared_error = metadata.get("episode_end_home_max_error_rad")
            if (
                not isinstance(declared_error, (int, float))
                or isinstance(declared_error, bool)
                or abs(float(declared_error) - max(terminal_errors)) > 1.0e-6
            ):
                failures.append("metadata terminal HOME error does not match raw observation")

    camera_rows: dict[str, list[dict[str, Any]]] = {
        "ceiling": [], "wrist_right": []
    }
    camera_times: dict[str, list[float]] = {
        "ceiling": [], "wrist_right": []
    }
    episode_root = episode.resolve()
    for row in streams["camera"]:
        key = row.get("key")
        if key not in camera_rows:
            failures.append(f"camera row has unknown key: {key}")
            continue
        camera_rows[key].append(row)
        relative = row.get("path")
        if not isinstance(relative, str):
            failures.append(f"camera {key} row has no path")
            continue
        image = (episode / relative).resolve()
        if episode_root not in image.parents or not image.is_file() or image.stat().st_size == 0:
            failures.append(f"camera {key} image is missing/empty/outside episode: {relative}")
    for key, rows in camera_rows.items():
        timestamps = []
        for row in rows:
            value = row.get("timestamp_s")
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                failures.append(f"camera {key} has invalid timestamp")
                timestamps = []
                break
            timestamps.append(float(value))
        if not rows:
            failures.append(f"camera {key} has no rows")
        elif any(b <= a for a, b in zip(timestamps, timestamps[1:])):
            failures.append(f"camera {key} timestamps are not strictly increasing")
            timestamps = []
        camera_times[key] = timestamps

    action_times = stream_times["demonstration_action"]
    observation_times = stream_times["joint_observation"]
    gripper_times = stream_times["gripper_event"]
    if action_times and observation_times and gripper_times:
        activity_start = min(action_times[0], gripper_times[0])
        activity_end = max(action_times[-1], gripper_times[-1])
        if observation_times[0] > activity_start or observation_times[-1] < activity_end:
            failures.append("joint observation does not cover complete replay activity")
        for key, timestamps in camera_times.items():
            if timestamps and (
                timestamps[0] > activity_start + maximum_camera_skew_s
                or timestamps[-1] < activity_end - maximum_camera_skew_s
            ):
                failures.append(f"camera {key} does not cover complete replay activity")

    actual_stream_counts = {
        name: len(rows) for name, rows in streams.items()
    }
    declared_stream_counts = metadata.get("stream_counts")
    for key in ("joint_observation", "demonstration_action", "gripper_event"):
        if (
            not isinstance(declared_stream_counts, dict)
            or declared_stream_counts.get(key) != actual_stream_counts[key]
        ):
            failures.append(f"metadata stream count mismatch for {key}")
    declared_cameras = metadata.get("camera_observation")
    for key, rows in camera_rows.items():
        if (
            not isinstance(declared_cameras, dict)
            or not isinstance(declared_cameras.get(key), dict)
            or declared_cameras[key].get("frame_count") != len(rows)
        ):
            failures.append(f"metadata camera frame count mismatch for {key}")

    return {
        "passed": not failures,
        "failures": failures,
        "stream_counts": actual_stream_counts,
        "camera_frame_counts": {
            key: len(rows) for key, rows in camera_rows.items()
        },
        "maximum_camera_coverage_slack_s": maximum_camera_skew_s,
    }


class EpisodeRecorder:
    STREAMS = ("physics", "joint_observation", "demonstration_action", "gripper_event", "camera", "sidecar", "keyboard_events", "gamepad_events")

    def __init__(self, root: Path):
        self.root = Path(root)
        self.transactions = self.root / ".transactions"
        self.episodes = self.root / "episodes"
        self.pending = self.root / "episodes_pending_review"
        self.failed = self.root / "failed"
        for directory in (self.transactions, self.episodes, self.pending, self.failed): directory.mkdir(parents=True, exist_ok=True)
        self.active: Path | None = None
        self.episode_id: str | None = None
        self._last_timestamp: dict[str, float] = {}

    @staticmethod
    def _atomic_json(path: Path, value: Any) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False); stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)

    def begin(self, metadata: dict[str, Any], episode_id: str | None = None) -> str:
        if self.active is not None: raise RecorderStateError("a transaction is already active")
        episode_id = episode_id or f"episode_{uuid.uuid4().hex}"
        if any((directory / episode_id).exists() for directory in (self.transactions, self.episodes, self.pending, self.failed)):
            raise RecorderStateError(f"episode id already exists: {episode_id}")
        active = self.transactions / episode_id; active.mkdir()
        payload = dict(metadata); payload.update({"episode_id": episode_id, "transaction_state": "ACTIVE", "terminal_label": None})
        self._atomic_json(active / "metadata.json", payload)
        self.active = active; self.episode_id = episode_id
        self._last_timestamp = {}
        return episode_id

    def append(self, stream_name: str, timestamp_s: float, payload: dict[str, Any]) -> None:
        if self.active is None: raise RecorderStateError("no active transaction")
        if stream_name not in self.STREAMS: raise ValueError(f"unknown stream: {stream_name}")
        timestamp_s = float(timestamp_s)
        clock_key = f"{stream_name}:{payload.get('key')}" if stream_name == "camera" else stream_name
        if timestamp_s <= self._last_timestamp.get(clock_key, float("-inf")):
            raise RecorderStateError(f"{stream_name}: timestamps must be strictly monotonic")
        self._last_timestamp[clock_key] = timestamp_s
        record = {"timestamp_s": timestamp_s, **payload}
        with (self.active / f"{stream_name}.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, separators=(",", ":"), ensure_ascii=False) + "\n")

    def _read_metadata(self) -> dict[str, Any]:
        if self.active is None: raise RecorderStateError("no active transaction")
        return json.loads((self.active / "metadata.json").read_text(encoding="utf-8"))

    def update_metadata(self, values: dict[str, Any]) -> None:
        """Atomically enrich an active transaction without changing its state."""
        if self.active is None:
            raise RecorderStateError("no active transaction")
        if not isinstance(values, dict):
            raise TypeError("metadata update must be a dictionary")
        reserved = {"episode_id", "transaction_state", "terminal_label"}
        if reserved.intersection(values):
            raise RecorderStateError("metadata update contains transaction-owned fields")
        metadata = self._read_metadata()
        metadata.update(values)
        self._atomic_json(self.active / "metadata.json", metadata)

    def finish_success(self, evaluator_result: Any) -> Path:
        if self.active is None: raise RecorderStateError("no active transaction")
        result = asdict(evaluator_result) if is_dataclass(evaluator_result) else dict(evaluator_result)
        if not result.get("full_sequence_success"):
            raise RecorderStateError("S rejected: automatic task contract did not pass")
        metadata = self._read_metadata(); metadata.update({"transaction_state": "COMMITTED", "terminal_label": "S", "task_result": result})
        self._atomic_json(self.active / "metadata.json", metadata)
        destination = self.episodes / self.episode_id
        os.replace(self.active, destination); self.active = None; self.episode_id = None
        self._last_timestamp = {}
        self.write_catalog(); return destination

    def finish_failure(self, reason: str) -> Path:
        if self.active is None: raise RecorderStateError("no active transaction")
        metadata = self._read_metadata(); metadata.update({"transaction_state": "FAILED", "terminal_label": "F", "failure_reason": str(reason)})
        self._atomic_json(self.active / "metadata.json", metadata)
        destination = self.failed / self.episode_id
        os.replace(self.active, destination); self.active = None; self.episode_id = None
        self._last_timestamp = {}
        self.write_catalog(); return destination

    def finish_operator_success(self, notes: str = "") -> Path:
        """Save an attended real episode without claiming automatic task validation."""
        if self.active is None: raise RecorderStateError("no active transaction")
        metadata = self._read_metadata(); metadata.update({
            "transaction_state": "COMMITTED_PENDING_REVIEW",
            "terminal_label": "S",
            "operator_marked_success": True,
            "automatic_task_contract_passed": False,
            "review_notes": str(notes),
        })
        self._atomic_json(self.active / "metadata.json", metadata)
        destination = self.pending / self.episode_id
        os.replace(self.active, destination); self.active = None; self.episode_id = None
        self._last_timestamp = {}
        self.write_catalog(); return destination

    def finish_automatic_pending_review(self, notes: str = "") -> Path:
        """Preserve a validated replay pending human task and image review."""
        if self.active is None: raise RecorderStateError("no active transaction")
        metadata = self._read_metadata(); metadata.update({
            "transaction_state": "COMMITTED_PENDING_REVIEW",
            "terminal_label": None,
            "operator_marked_success": False,
            "automatic_task_contract_passed": False,
            "review_status": "pending",
            "review_notes": str(notes),
        })
        self._atomic_json(self.active / "metadata.json", metadata)
        destination = self.pending / self.episode_id
        os.replace(self.active, destination); self.active = None; self.episode_id = None
        self._last_timestamp = {}
        self.write_catalog(); return destination

    def cancel(self) -> None:
        if self.active is None: raise RecorderStateError("no active transaction")
        shutil.rmtree(self.active); self.active = None; self.episode_id = None; self._last_timestamp = {}; self.write_catalog()

    def residual_transactions(self) -> list[dict[str, Any]]:
        residuals = []
        for path in sorted(self.transactions.iterdir()):
            if not path.is_dir(): continue
            metadata_path = path / "metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
            residuals.append({"episode_id": path.name, "state": metadata.get("transaction_state", "CORRUPT")})
        return residuals

    def write_catalog(self) -> dict[str, Any]:
        catalog = {
            "episodes": sorted(path.name for path in self.episodes.iterdir() if path.is_dir()),
            "episodes_pending_review": sorted(path.name for path in self.pending.iterdir() if path.is_dir()),
            "failed": sorted(path.name for path in self.failed.iterdir() if path.is_dir()),
            "residual_transactions": self.residual_transactions(),
        }
        self._atomic_json(self.root / "catalog.json", catalog); return catalog
