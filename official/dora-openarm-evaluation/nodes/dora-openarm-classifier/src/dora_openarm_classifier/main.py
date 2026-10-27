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

"""dora-rs node that classifies whether the current state completes the task successfully or not.

For each incoming camera frame it runs single-frame TOPReward, keeps a
sliding window of the last WINDOW scores, and reports the window
median as the verdict. When arm positions are provided, success is
only latched once the grippers are open (release) AND the vision
window agrees -- combining vision and proprioception, like the
production classifier node.

Inputs:
  image          JPEG bytes (uint8 array) with width/height/encoding metadata
  position       optional 16-dim arm state [right(8), left(8)]; grippers at 7 and 15
  position_right optional 8-dim right arm state; gripper at 7
  position_left  optional 8-dim left arm state; gripper at 7
Outputs:
  result         Float32 score; metadata has "verdict" (SUCCESS/FAIL) and "frame"

Command line options/Environment variables:
  --question, QUESTION       yes/no success question
  --server-url, SERVER_URL   URL for dora-openarm-classifier-server
  --window, WINDOW           sliding-window size in frames (default 5)
  --threshold, THRESHOLD     P_yes threshold for SUCCESS (default 0.5)
  --gripper-threshold, GRIPPER_THRESHOLD
                             |gripper joint| above which a gripper counts
                             as open (default 0.2)
  --classify-hz, CLASSIFY_HZ max inference rate; drop frames arriving
                             faster (0 = no limit)
"""

import argparse
from collections import deque
import dora
from PIL import Image
import httpx
import io
import numpy as np
import os
import pyarrow as pa
import time

from .topreward import TOPRewardScorer


def _score_remote(args, image_bytes):
    response = httpx.post(
        f"{args.server_url}/classify",
        data={"question": args.question, "threshold": args.threshold},
        files={"image": ("image.jpeg", image_bytes, "image/jpeg")},
        timeout=10,
    )

    if response.status_code != 200:
        print(
            f"ERROR: Server returned {response.status_code}: {response.content}",
            flush=True,
        )
        return None

    return response.json()["score"]


def main():
    """Run the real-time TOPReword based success classifier."""
    parser = argparse.ArgumentParser(
        description="Classify whether the current task is completed successully or not"
    )
    default_question = os.getenv("QUESTION")
    parser.add_argument(
        "--question",
        required=default_question is None,
        default=default_question,
        help="Yes/no question to ask about images",
    )
    parser.add_argument(
        "--server-url",
        default=os.getenv("SERVER_URL"),
        help="dora-openarm-classifier-server URL",
    )
    parser.add_argument(
        "--window",
        default=int(os.getenv("WINDOW", "5")),
        help="The number of images to use for classification",
        type=int,
    )
    parser.add_argument(
        "--threshold",
        default=float(os.getenv("THRESHOLD", "0.5")),
        help="P_yes threshold for SUCCESS (default: 0.5)",
        type=float,
    )
    parser.add_argument(
        "--gripper-threshold",
        type=float,
        default=float(os.getenv("GRIPPER_THRESHOLD", "0.2")),
        help="Above which a gripper counts as open (default 0.2)",
    )
    parser.add_argument(
        "--classify-hz",
        type=float,
        default=float(os.getenv("CLASSIFY_HZ", "0.0")),
        # Real cameras run at ~30fps but Qwen is ~2Hz, so cap it and
        # drop frames that arrive too soon.
        help="Max classify rate; drop frames arriving faster (0 = no limit)",
    )
    args = parser.parse_args()
    min_interval = 1.0 / args.classify_hz if args.classify_hz > 0 else 0.0
    last_classify = 0.0

    if args.server_url:
        print(f"Connecting to server: {args.server_url}", flush=True)
        try:
            httpx.get(f"{args.server_url}/health", timeout=5)
        except Exception as error:
            print(f"Cannot connect to server: {args.server_url}: {error}", flush=True)
            print(
                "Make sure server is running: dora-openarm-classifier-server",
                flush=True,
            )
            raise
    else:
        print("Loading Qwen3-VL ...", flush=True)
        scorer = TOPRewardScorer()

    print("Ready:", flush=True)
    print(f"  window: {args.window}", flush=True)
    print(f"  threshold: {args.threshold:.0%}", flush=True)
    print(f"  question: {args.question}", flush=True)

    scores = deque(maxlen=args.window)
    last_right_pos = None  # latest right arm position (8-dim)
    last_left_pos = None  # latest left arm position (8-dim)
    success_latched = False  # sticky once vision + grippers agree
    node = dora.Node()
    n = 0
    for event in node:
        if event["type"] != "INPUT":
            continue

        if event["id"] == "position":
            value = event["value"]
            if isinstance(value, pa.StructArray):
                value = value.field("qpos")
            last_pos = value.to_numpy(zero_copy_only=False)
            last_right_pos = last_pos[:8]
            last_left_pos = last_pos[8:]
            continue

        if event["id"] == "position_right":
            value = event["value"]
            if isinstance(value, pa.StructArray):
                value = value.field("qpos")
            last_right_pos = value.to_numpy(zero_copy_only=False)
            continue

        if event["id"] == "position_left":
            value = event["value"]
            if isinstance(value, pa.StructArray):
                value = value.field("qpos")
            last_left_pos = value.to_numpy(zero_copy_only=False)
            continue

        if event["id"] == "image":
            # rate-limit: drop frames that arrive faster than INFER_HZ
            now = time.monotonic()
            if min_interval and (now - last_classify) < min_interval:
                continue
            last_classify = now

            image_bytes = event["value"].to_numpy(zero_copy_only=False).tobytes()
            if args.server_url:
                score = _score_remote(args, image_bytes)
                if score is None:
                    continue
            else:
                image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                _, _, score = scorer.score(image, args.question)
            scores.append(score)
            agg = float(np.median(scores))

            grip_info = ""
            if (
                last_right_pos is not None
                and len(last_right_pos) >= 8
                and last_left_pos is not None
                and len(last_left_pos) >= 8
            ):
                gr, gl = float(last_right_pos[7]), float(last_left_pos[7])
                grippers_open = (
                    abs(gr) > args.gripper_threshold
                    and abs(gl) > args.gripper_threshold
                )
                if not success_latched and grippers_open and agg >= args.threshold:
                    success_latched = True
                verdict = "SUCCESS" if success_latched else "FAIL"
                grip_info = f"  grip(r={gr:+.2f} l={gl:+.2f} {'OPEN' if grippers_open else 'closed'})"
            else:
                # no arm data -> vision-only fallback
                verdict = "SUCCESS" if agg >= args.threshold else "FAIL"

            n += 1
            print(
                f"frame {n:03d}  P_yes={score:5.1%}  "
                f"window_median={agg:5.1%}{grip_info}  -> {verdict}",
                flush=True,
            )
            node.send_output(
                "result",
                pa.array([agg], type=pa.float32()),
                metadata={"verdict": verdict, "frame": n},
            )


if __name__ == "__main__":
    main()
