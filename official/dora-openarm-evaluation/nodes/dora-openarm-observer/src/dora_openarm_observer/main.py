# Copyright 2026 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Node to collect the last observation."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import cv2
import dora
import datetime
import os
import pyarrow as pa
import time

SUPPORTED_CAMERAS = (
    "camera_wrist_right",
    "camera_wrist_left",
    "camera_head_left",
    "camera_head_right",
    "camera_ceiling",
)

DEFAULT_SOURCE_MAX_AGE_NS = 200_000_000
DEFAULT_SOURCE_MAX_SKEW_NS = 100_000_000


def _reset_observation(observation, arms, cameras=None):
    """Initialize/reset observations to None and ID to 0."""
    if cameras is None:
        cameras = ["camera_head_left", "camera_head_right", "camera_ceiling"]
        if "right" in arms:
            cameras.append("camera_wrist_right")
        if "left" in arms:
            cameras.append("camera_wrist_left")
    if "right" in arms:
        observation["arm_right"] = None
    if "left" in arms:
        observation["arm_left"] = None
    for camera in cameras:
        observation[camera] = None
    observation["id"] = 0


def _decode_camera(encoded):
    image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError("Failed to decode JPEG camera observation")
    return image


def _arm_position_array(event):
    """Normalize flat or OpenArm ``[{qpos: [...] }]`` state events."""
    value = event["value"]
    if pa.types.is_struct(value.type):
        if "qpos" not in value.type.names:
            raise ValueError("arm observation struct has no qpos field")
        value = value.field("qpos")[0]
        if pa.types.is_list(value.type) or pa.types.is_large_list(value.type):
            value = value.values
    return value


def validate_source_timing(
    source_times_ns, required_ids, now_ns, max_age_ns, max_skew_ns
):
    """Return source age/skew, or reject missing, stale, or unsynchronized inputs."""
    missing = [event_id for event_id in required_ids if event_id not in source_times_ns]
    if missing:
        raise ValueError(f"missing source timestamps: {','.join(missing)}")
    timestamps = [int(source_times_ns[event_id]) for event_id in required_ids]
    ages = [int(now_ns) - timestamp for timestamp in timestamps]
    if any(age < 0 for age in ages):
        raise ValueError("source timestamp is from the future")
    max_age = max(ages)
    skew = max(timestamps) - min(timestamps)
    if max_age > max_age_ns:
        raise ValueError(f"source data stale: age_ns={max_age} limit_ns={max_age_ns}")
    if skew > max_skew_ns:
        raise ValueError(
            f"source data unsynchronized: skew_ns={skew} limit_ns={max_skew_ns}"
        )
    return max_age, skew


def event_source_timestamp_ns(event):
    """Return ``(clock, timestamp_ns)`` for a source event.

    Hardware producers attach ``source_monotonic_ns`` at acquisition/read
    time.  Synthetic or legacy producers may only have Dora's epoch HLC
    ``timestamp``.  Missing timestamps are rejected instead of being replaced
    with observer receipt time; otherwise a delayed or cached frame could look
    fresh.  A graph must not mix clock domains in one observation.
    """
    metadata = event.get("metadata") or {}
    source_timestamp = metadata.get("source_monotonic_ns")
    if source_timestamp is not None:
        return "monotonic", int(source_timestamp)
    timestamp = metadata.get("timestamp")
    if timestamp is None:
        return None
    if isinstance(timestamp, datetime.datetime):
        return "epoch", int(timestamp.timestamp() * 1_000_000_000)
    timestamp_ns = int(timestamp)
    # Accept a seconds-form timestamp from test/integration producers too.
    if timestamp_ns < 1_000_000_000_000:
        timestamp_ns *= 1_000_000_000
    return "epoch", timestamp_ns


def _build_output(
    observation, phase_classifier_result, task_prompt, metadata, decode_pool
):
    """Convert observation to Apache Arrow data and fill metadata.

    observation keys (all values are dora events with a "value" field):
      "arm_right"          – pa.array float32, len 8 (7 joints + 1 gripper)
      "arm_left"           – pa.array float32, len 8
      "camera_wrist_right" – JPEG-encoded uint8 flat array, 960×600
      "camera_wrist_left"  – JPEG-encoded uint8 flat array, 960×600
      "camera_head_left"   – JPEG-encoded uint8 flat array, 1280×720
      "camera_head_right"  – JPEG-encoded uint8 flat array, 1280×720
      "camera_ceiling"     – JPEG-encoded uint8 flat array, 960×600
      "id"                 – int64, incremented for each observation

    Output pa.StructArray fields:
      "position"           – concatenated arm positions, list<float32>
      "camera_wrist_right" – decoded RGB flat array, list<uint8>
      "camera_wrist_left"  – decoded RGB flat array, list<uint8>
      "camera_head_left"   – decoded RGB flat array, list<uint8>
      "camera_head_right"  – decoded RGB flat array, list<uint8>
      "camera_ceiling"     – decoded RGB flat array, list<uint8>
      "phase_classifier_result" – StructArray or null
      "task_prompt"        – string (language instruction for the policy)
      "id"                 – int64, incremented for each observation

    metadata is mutated to add per-camera height/width/encoding keys.
    """
    arrays = []
    names = []
    position_arrays = []
    if "arm_right" in observation:
        position_arrays.append(_arm_position_array(observation["arm_right"]))
    if "arm_left" in observation:
        position_arrays.append(_arm_position_array(observation["arm_left"]))
    arrays.append(
        pa.array(
            [pa.concat_arrays(position_arrays)], type=pa.list_(position_arrays[0].type)
        )
    )
    names.append("position")

    camera_names = [name for name in SUPPORTED_CAMERAS if name in observation]
    decode_futures = {
        name: decode_pool.submit(
            _decode_camera,
            observation[name]["value"].to_numpy(),
        )
        for name in camera_names
    }

    def add_camera_observation(name):
        image = decode_futures[name].result()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        metadata[f"{name}.encoding"] = "rgb8"
        metadata[f"{name}.height"] = image.shape[0]
        metadata[f"{name}.width"] = image.shape[1]
        arrays.append(pa.array([image.ravel()], type=pa.list_(pa.uint8())))
        names.append(name)

    for name in camera_names:
        add_camera_observation(name)
    if phase_classifier_result is None:
        arrays.append(pa.array([None]))
    else:
        arrays.append(phase_classifier_result)
    names.append("phase_classifier_result")
    arrays.append(pa.array([task_prompt], type=pa.string()))
    names.append("task_prompt")
    arrays.append(pa.array([observation["id"]], type=pa.int64()))
    names.append("id")
    return pa.StructArray.from_arrays(arrays, names)


def main():
    """Collect the last observation."""
    parser = argparse.ArgumentParser(description="Collect the last observation")
    parser.add_argument(
        "--arms",
        default=os.getenv("ARMS", "right,left"),
        help="The used arms: 'right,left' (default), 'right' or 'left'",
        type=str,
    )
    parser.add_argument(
        "--decode-workers",
        default=int(os.getenv("OBSERVER_DECODE_WORKERS", "4")),
        type=int,
        help="Number of persistent JPEG decode workers (default: 4)",
    )
    parser.add_argument(
        "--cameras",
        default=os.getenv("OBSERVER_CAMERAS"),
        help="Comma-separated camera inputs required for each observation",
    )
    parser.add_argument(
        "--max-source-age-ms",
        default=float(os.getenv("OBSERVER_MAX_SOURCE_AGE_MS", "200")),
        type=float,
        help="Reject observations when any source event is older than this age",
    )
    parser.add_argument(
        "--max-source-skew-ms",
        default=float(os.getenv("OBSERVER_MAX_SOURCE_SKEW_MS", "100")),
        type=float,
        help="Reject observations when source event times differ by more than this",
    )
    args = parser.parse_args()
    if args.decode_workers <= 0:
        raise ValueError("--decode-workers must be positive")
    if args.max_source_age_ms <= 0 or args.max_source_skew_ms < 0:
        parser.error("source timing limits must be positive")

    arms = args.arms.split(",")
    cameras = (
        None if args.cameras is None
        else [name.strip() for name in args.cameras.split(",") if name.strip()]
    )
    if cameras is None:
        cameras = ["camera_head_left", "camera_head_right", "camera_ceiling"]
        if "right" in arms:
            cameras.append("camera_wrist_right")
        if "left" in arms:
            cameras.append("camera_wrist_left")
    if not cameras or len(cameras) != len(set(cameras)) or any(
        name not in SUPPORTED_CAMERAS for name in cameras
    ):
        parser.error("--cameras must contain unique supported camera names")
    node = dora.Node()
    observation = {}
    _reset_observation(observation, arms, cameras)
    cv2.setNumThreads(1)
    decode_pool = ThreadPoolExecutor(
        max_workers=args.decode_workers,
        thread_name_prefix="observer-jpeg",
    )
    episode_number = 0
    last_phase_classifier_result = None
    last_task_prompt = None
    last_arm_right_status = None
    last_arm_left_status = None
    command_status = "stopped"
    source_times_ns = {}
    source_clock = None
    required_ids = [
        *(["arm_right"] if "right" in arms else []),
        *(["arm_left"] if "left" in arms else []),
        *cameras,
    ]
    max_source_age_ns = int(args.max_source_age_ms * 1_000_000)
    max_source_skew_ns = int(args.max_source_skew_ms * 1_000_000)
    last_timing_rejection_log_ns = 0
    try:
        for event in node:
            if event["type"] != "INPUT":
                continue

            # Main process
            event_id = event["id"]
            if event_id == "tick":
                if any(v is None for v in observation.values()):
                    # If any observation isn't ready yet, we skip this tick.
                    continue
                if (
                    ("right" in arms and last_arm_right_status == "stopped")
                    or ("left" in arms and last_arm_left_status == "stopped")
                    or command_status not in ("start", "ready")
                ):
                    _reset_observation(observation, arms, cameras)
                    source_times_ns.clear()
                    source_clock = None
                    continue
                now_ns = (time.monotonic_ns() if source_clock == "monotonic"
                          else time.time_ns())
                try:
                    source_age_ns, source_skew_ns = validate_source_timing(
                        source_times_ns,
                        required_ids,
                        now_ns,
                        max_source_age_ns,
                        max_source_skew_ns,
                    )
                except ValueError as error:
                    if now_ns - last_timing_rejection_log_ns >= 1_000_000_000:
                        print(f"OBSERVATION_REJECTED: {error}", flush=True)
                        last_timing_rejection_log_ns = now_ns
                    continue
                metadata = {
                    "episode_number": episode_number,
                    "timestamp": time.time_ns(),
                    "observation_monotonic_ns": time.monotonic_ns(),
                    "source_age_ns": source_age_ns,
                    "source_skew_ns": source_skew_ns,
                }
                arrow_observation = _build_output(
                    observation,
                    last_phase_classifier_result,
                    last_task_prompt,
                    metadata,
                    decode_pool,
                )
                node.send_output(
                    "observation",
                    arrow_observation,
                    metadata,
                )
                observation["id"] += 1
            elif event_id == "command":
                # started, stopped, aligned
                command_status = event["value"][0].as_py()
            elif event_id == "arm_right_status":
                last_arm_right_status = event["value"][0].as_py()
            elif event_id == "arm_left_status":
                last_arm_left_status = event["value"][0].as_py()
            elif event_id == "phase_classifier_result":
                last_phase_classifier_result = event["value"]
            elif event_id == "task_prompt":
                last_task_prompt = event["value"][0].as_py()
            else:
                observation[event_id] = event
                source_timestamp_ns = event_source_timestamp_ns(event)
                if source_timestamp_ns is None:
                    source_times_ns.pop(event_id, None)
                else:
                    event_clock, timestamp_ns = source_timestamp_ns
                    if source_clock is None:
                        source_clock = event_clock
                    if event_clock != source_clock:
                        source_times_ns.pop(event_id, None)
                    else:
                        source_times_ns[event_id] = timestamp_ns
    finally:
        decode_pool.shutdown(wait=True)


if __name__ == "__main__":
    main()
