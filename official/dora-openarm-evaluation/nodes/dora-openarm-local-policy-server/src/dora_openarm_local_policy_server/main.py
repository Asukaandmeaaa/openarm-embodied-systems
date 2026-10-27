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

"""Node to communicate with a local policy server."""

import argparse
import dora
import json
import os
import pyarrow as pa
import socket
import tempfile
import time

from observation_timing import validate_timing


TIMING_FAULT_OUTPUT = "timing_fault"


def _timing_limit_ns(name, default_ms):
    return int(float(os.getenv(name, str(default_ms))) * 1_000_000)


def _validate_observation_timing(metadata, now_ns):
    """Reject observations without fresh source timing metadata."""
    return validate_timing(
        metadata,
        now_ns,
        max_source_age_ns=_timing_limit_ns("MAX_SOURCE_AGE_MS", 200),
        max_source_skew_ns=_timing_limit_ns("MAX_SOURCE_SKEW_MS", 100),
        max_observation_age_ns=_timing_limit_ns("MAX_OBSERVATION_AGE_MS", 500),
        required=os.getenv("REQUIRE_SOURCE_TIMING", "yes").lower()
        in ("1", "true", "yes"),
    )


def _emit_timing_fault(node, reason):
    """Publish a latched freshness fault for the guarded arm boundary."""
    node.send_output(
        TIMING_FAULT_OUTPUT,
        pa.array([str(reason)], type=pa.string()),
        {"fault": "observation_timing", "timestamp": time.time_ns()},
    )


def _chunk_horizon_ns(actions):
    """Return the time covered by a timestamped position chunk."""
    return int(actions["interval"] * max(0, len(actions["positions"]) - 1))


def _main_dora(io, shared_dir, execute_full_chunk=False):
    n_keep_data = 5  # TODO: Customizable?
    data_files = []

    node = dora.Node()
    previous_observation_id = None
    next_inference_at_ns = 0
    timing_fault_latched = False
    for event in node:
        if event["type"] != "INPUT":
            continue

        # Timing failures are fail-closed for this session.  Dropping a stale
        # observation without notifying the arm would leave its last command
        # active until the much longer controller action timeout.
        if timing_fault_latched:
            continue

        # Main process
        observation = event["value"]
        observation_id = observation.field("id")[0].as_py()
        reset = previous_observation_id is None or observation_id < previous_observation_id
        previous_observation_id = observation_id
        if (
            execute_full_chunk
            and not reset
            and time.monotonic_ns() < next_inference_at_ns
        ):
            continue
        try:
            timing = _validate_observation_timing(
                event.get("metadata", {}), time.monotonic_ns()
            )
        except ValueError as error:
            print(f"OBSERVATION_REJECTED: {error}", flush=True)
            _emit_timing_fault(node, error)
            timing_fault_latched = True
            continue

        def prepare_request():
            nonlocal previous_observation_id
            data_file = tempfile.NamedTemporaryFile(
                suffix=".arrow", dir=shared_dir, delete_on_close=False
            )
            record_batch = pa.RecordBatch.from_struct_array(observation)
            with pa.output_stream(data_file) as output:
                with pa.ipc.new_file(output, record_batch.schema) as writer:
                    writer.write(record_batch)
            data_files.append(data_file)
            if len(data_files) > n_keep_data:
                data_files.pop(0)
            return {
                "name": "inference",
                "data_path": data_file.name,
                "reset": reset,
                "metadata": event["metadata"],
            }

        # dora-rs node -> Policy server: Inference request
        #   {"name": "inference", "data_path": "/data/path.arrow",
        #    "reset": true/false, ...}
        #
        # "/data/path.arrow" has a record batch:
        #   {
        #     # element len: 8 (7 joints + 1 gripper) * 2 (right + left)
        #     # "arm_right" + "arm_left"
        #     "position": pa.list_(pa.float32()),
        #     # element len: 600 (height) * 960 (width) * 3 (RGB)
        #     # element shape: (height, width, color)
        #     "camera_wrist_right": pa.list_(pa.uint8()),
        #     # element len: 600 (height) * 960 (width) * 3 (RGB)
        #     # element shape: (height, width, color)
        #     "camera_wrist_left": pa.list_(pa.uint8()),
        #     # element len: 600 (height) * 960 (width) * 3 (RGB)
        #     # element shape: (height, width, color)
        #     "camera_head": pa.list_(pa.uint8()),
        #     # element len: 600 (height) * 960 (width) * 3 (RGB)
        #     # element shape: (height, width, color)
        #     "camera_ceiling": pa.list_(pa.uint8()),
        #     }
        #   }
        request = prepare_request()
        io.write(json.dumps(request) + "\n")
        io.flush()

        # Policy server -> dora-rs node: Inferred actions
        #   {
        #     "interval": interval_in_ns,
        #     "positions": [
        #        [...],
        #        ...
        #     ]
        #   }
        #
        #   Arm position: Motor positions
        #     Bimanual: 8 (7 joints + 1 gripper) * 2 (right + left)
        #     Unimanual: 8 (7 joints + 1 gripper)
        response = io.readline()
        if not response:
            break
        actions = json.loads(response)
        if actions.get("error"):
            reason = actions.get("error_detail") or actions["error"]
            print(f"POLICY_TIMING_FAULT: {reason}", flush=True)
            _emit_timing_fault(node, reason)
            timing_fault_latched = True
            continue
        if actions["positions"]:
            # "reset" signals that these actions are the first of a new episode,
            # so downstream nodes can drop any state carried over from the
            # previous episode.
            metadata = {"interval": actions["interval"], "reset": request["reset"]}
            if "cutoff_hz" in actions:
                metadata["cutoff_hz"] = actions["cutoff_hz"]
            if timing is not None:
                metadata.update(timing)
            node.send_output(
                "actions",
                pa.array(actions["positions"], type=pa.list_(pa.float32())),
                metadata,
            )
            if execute_full_chunk:
                next_inference_at_ns = time.monotonic_ns() + _chunk_horizon_ns(actions)


def main():
    """Communicate with a local policy server."""
    parser = argparse.ArgumentParser(
        description="Communicate with a local policy server"
    )
    parser.add_argument(
        "--socket",
        default=os.getenv("SOCKET"),
        help="The local socket to communicate",
        type=str,
    )
    parser.add_argument(
        "--execute-full-chunk",
        action=argparse.BooleanOptionalAction,
        default=os.getenv("EXECUTE_FULL_CHUNK", "no").lower() in ("1", "true", "yes"),
        help="Apply backpressure until the emitted action chunk horizon has elapsed",
    )
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(
        prefix="dora-openarm-local-policy-server", dir="/dev/shm"
    ) as shared_dir:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.connect(args.socket)
            with sock.makefile("rw") as io:
                _main_dora(io, shared_dir, args.execute_full_chunk)


if __name__ == "__main__":
    main()
