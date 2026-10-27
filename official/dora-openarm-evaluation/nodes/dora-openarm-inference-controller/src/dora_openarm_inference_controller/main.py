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

"""Node to control an inference."""

import argparse
import dora
import os
import pyarrow as pa
import time


def main():
    """Control inference dataflow."""
    parser = argparse.ArgumentParser(description="Control inference dataflow")
    parser.add_argument(
        "--arms",
        default=os.getenv("ARMS", "right,left"),
        help="The used arms: 'right,left' (default), 'right' or 'left'",
        type=str,
    )
    parser.add_argument(
        "--timeout",
        default=int(os.getenv("TIMEOUT", 60)),
        help="Timeout seconds for 1 try.",
        type=int,
    )
    parser.add_argument(
        "--max-n-retries",
        default=int(os.getenv("MAX_N_RETRIES", 0)),
        help="The max number of retries",
        type=int,
    )
    parser.add_argument(
        "--success-threshold",
        default=float(os.getenv("SUCCESS_THRESHOLD", 0.8)),
        help="The threshold of success confidence (0.0 ~ 1.0)",
        type=float,
    )
    args = parser.parse_args()
    arms = args.arms.split(",")

    ready_arms = {}
    for arm in arms:
        ready_arms[arm] = False

    def is_ready():
        return all(ready_arms.values())

    timeout_ns = int(args.timeout * 1e9)
    start_time_ns = None
    n_retries = 0

    node = dora.Node()
    for event in node:
        if event["type"] != "INPUT":
            continue

        # Main process
        event_id = event["id"]
        if event_id == "arm_right_status" or event_id == "arm_left_status":
            if is_ready():
                continue
            side = event_id.removeprefix("arm_").removesuffix("_status")
            if event["value"][0].as_py() in ("aligned", "ready"):
                ready_arms[side] = True
                if is_ready():
                    start_time_ns = time.monotonic_ns()
                    node.send_output(
                        "command", pa.array(["start"]), {"episode_number": n_retries}
                    )
        else:
            if not is_ready():
                continue
            if event_id == "phase_classifier_result":
                score = event["value"][0].as_py()
                verdict = event["metadata"].get("verdict")
                if verdict == "SUCCESS" and score > args.success_threshold:
                    node.send_output("command", pa.array(["success"]))
                    break
            elif event_id == "progress_tick":
                elapsed_ns = time.monotonic_ns() - start_time_ns
                if timeout_ns <= elapsed_ns:
                    node.send_output("command", pa.array(["fail"]))
                    if n_retries >= args.max_n_retries:
                        break
                    n_retries += 1
                    start_time_ns = time.monotonic_ns()
                    node.send_output(
                        "command", pa.array(["start"]), {"episode_number": n_retries}
                    )
                continue
            node.send_output(event_id, event["value"], event["metadata"])
    node.send_output("command", pa.array(["quit"]))


if __name__ == "__main__":
    main()
