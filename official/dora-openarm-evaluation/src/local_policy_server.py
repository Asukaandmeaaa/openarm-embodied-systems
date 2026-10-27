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


"""Local policy server for OpenArm inference with LeRobot 0.6.1.

Since LeRobot 0.4, normalization no longer happens inside the policy: it moved
into pre/post-processor pipelines that are stored alongside the weights on the
Hub. The observation therefore goes through `preprocessor` before the policy and
the predicted chunk goes through `postprocessor` afterwards.
"""

import json
import os
import socket
import sys
import time

import numpy as np
import pyarrow as pa
import torch
from PIL import Image

from lerobot.configs import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from observation_timing import validate_timing

PRETRAINED_PATH = os.getenv(
    "OPENARM_PRETRAINED_PATH",
    "enactic/act-openarm-2-cell-pick_up_cube_mujoco",
)
DEFAULT_SOCKET = os.getenv("SOCKET", "/dev/shm/policy-server.socket")

CAMERA_KEY_MAP = {
    "camera_ceiling": "observation.images.ceiling",
    "camera_head_left": "observation.images.head_left",
    "camera_head_right": "observation.images.head_right",
    "camera_wrist_left": "observation.images.wrist_left",
    "camera_wrist_right": "observation.images.wrist_right",
}

IMAGE_SIZES: dict[str, tuple[int, int]] = {}
DEBUG_OBSERVATIONS = os.getenv("OPENARM_DEBUG_OBSERVATIONS", "no").lower() in (
    "1",
    "true",
    "yes",
)
_observation_count = 0
_previous_image_samples: dict[str, np.ndarray] = {}

KNOWN_RESOLUTIONS = {
    600 * 960 * 3: (600, 960),
    720 * 1280 * 3: (720, 1280),
    480 * 640 * 3: (480, 640),
    1080 * 1920 * 3: (1080, 1920),
}

INTERVAL_NS = 33_333_333
CUTOFF_HZ = 15


def validate_request_timing(metadata, now_ns=None):
    """Reject policy requests that lack fresh source timing metadata."""
    if now_ns is None:
        now_ns = time.monotonic_ns()
    required_timing = os.getenv("REQUIRE_SOURCE_TIMING", "yes").lower() in (
        "1",
        "true",
        "yes",
    )
    return validate_timing(
        metadata,
        now_ns,
        max_source_age_ns=int(float(os.getenv("MAX_SOURCE_AGE_MS", "200")) * 1_000_000),
        max_source_skew_ns=int(float(os.getenv("MAX_SOURCE_SKEW_MS", "100")) * 1_000_000),
        max_observation_age_ns=int(
            float(os.getenv("MAX_OBSERVATION_AGE_MS", "500")) * 1_000_000
        ),
        required=required_timing,
    )


def detect_resolution(n_bytes):
    if n_bytes in KNOWN_RESOLUTIONS:
        return KNOWN_RESOLUTIONS[n_bytes]
    n_pixels = n_bytes // 3
    for ratio_h, ratio_w in [(3, 4), (9, 16), (3, 5), (2, 3)]:
        h = int((n_pixels * ratio_h / ratio_w) ** 0.5)
        w = n_pixels // h
        if h * w == n_pixels:
            return (h, w)
    raise ValueError(f"Cannot determine resolution for {n_bytes} bytes")


def prepare_image(raw_data, target_h, target_w):
    data = raw_data.values.to_numpy(zero_copy_only=False).astype(np.uint8)
    src_h, src_w = detect_resolution(len(data))
    img = data.reshape(src_h, src_w, 3)
    if src_h != target_h or src_w != target_w:
        img = np.array(Image.fromarray(img).resize((target_w, target_h)))
    return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0


def observation_to_batch(observation):
    # The preprocessor adds the batch dimension and moves everything to the
    # policy device, so the batch is built here as unbatched CPU tensors.
    global _observation_count
    batch = {}

    position = observation["position"].values.to_numpy(zero_copy_only=False).astype(np.float32)
    batch["observation.state"] = torch.from_numpy(position)

    for arrow_key, model_key in CAMERA_KEY_MAP.items():
        if model_key not in IMAGE_SIZES:
            continue
        target_h, target_w = IMAGE_SIZES[model_key]
        image = prepare_image(observation[arrow_key], target_h, target_w)
        batch[model_key] = image
        if DEBUG_OBSERVATIONS and (_observation_count < 3 or _observation_count % 10 == 0):
            # A sparse deterministic sample is enough to detect black/frozen
            # frames without copying full-resolution images for diagnostics.
            sample = image[:, ::32, ::32].numpy()
            previous = _previous_image_samples.get(model_key)
            delta = (
                float(np.mean(np.abs(sample - previous)))
                if previous is not None
                else None
            )
            print(
                f"[DBG OBS] n={_observation_count} {model_key} "
                f"min={float(image.min()):.4f} max={float(image.max()):.4f} "
                f"mean={float(image.mean()):.4f} std={float(image.std()):.4f} "
                f"sample_delta={delta}",
                flush=True,
            )
            _previous_image_samples[model_key] = sample.copy()

    if DEBUG_OBSERVATIONS and (_observation_count < 3 or _observation_count % 10 == 0):
        print(
            f"[DBG OBS] n={_observation_count} state="
            f"{np.round(position, 4).tolist()}",
            flush=True,
        )
    _observation_count += 1

    return batch


def infer(policy, preprocessor, postprocessor, observation):
    batch = preprocessor(observation_to_batch(observation))
    actions = policy.predict_action_chunk(batch)

    # The postprocessor expects (batch, action_dim), so the chunk is folded into
    # the batch dimension for a single call and unfolded afterwards.
    n_batch, chunk_size, action_dim = actions.shape
    actions = postprocessor(actions.reshape(n_batch * chunk_size, action_dim))
    actions = actions.reshape(n_batch, chunk_size, action_dim)

    positions = actions.squeeze(0).cpu().numpy().tolist()
    return {
        "interval": INTERVAL_NS,
        "cutoff_hz": CUTOFF_HZ,
        "positions": positions,
    }


def warmup_policy(policy, preprocessor, postprocessor):
    """Initialize inference kernels before a fresh observation has a deadline."""
    sample = {key: torch.zeros(tuple(feature.shape), dtype=torch.float32)
              for key, feature in policy.config.input_features.items()}
    started = time.monotonic()
    with torch.inference_mode():
        for _ in range(3):
            prediction = policy.predict_action_chunk(preprocessor(sample))
            n_batch, n_steps, n_dim = prediction.shape
            postprocessor(prediction.reshape(n_batch*n_steps, n_dim))
    if torch.cuda.is_available() and str(policy.config.device).startswith("cuda"):
        torch.cuda.synchronize()
    policy.reset()
    preprocessor.reset()
    postprocessor.reset()
    print(f"Policy warmup finished in {time.monotonic()-started:.3f}s; state reset", flush=True)


def main():
    socket_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SOCKET
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Loading policy from {PRETRAINED_PATH} on {device}...")
    policy_config = PreTrainedConfig.from_pretrained(PRETRAINED_PATH)
    policy_config.pretrained_path = PRETRAINED_PATH
    policy_config.device = device
    policy = get_policy_class(policy_config.type).from_pretrained(
        config=policy_config,
        pretrained_name_or_path=policy_config.pretrained_path,
    )
    policy.reset()

    # The saved preprocessor pins the training device (cuda); only its device
    # step is overridden. The postprocessor already ends on cpu.
    preprocessor, postprocessor = make_pre_post_processors(
        policy_config,
        pretrained_path=PRETRAINED_PATH,
        preprocessor_overrides={"device_processor": {"device": device}},
    )

    for model_key in CAMERA_KEY_MAP.values():
        if model_key not in policy.config.input_features:
            continue
        feature = policy.config.input_features[model_key]
        c, h, w = feature.shape
        IMAGE_SIZES[model_key] = (h, w)
    if not IMAGE_SIZES:
        raise ValueError("checkpoint declares no supported OpenArm camera input")
    print(f"Policy loaded. Expected image sizes: {IMAGE_SIZES}")
    warmup_policy(policy, preprocessor, postprocessor)

    if os.path.exists(socket_path):
        os.remove(socket_path)

    print(f"Listening on {socket_path}")
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.bind(socket_path)
        sock.listen()
        try:
            with sock.accept()[0] as conn:
                print("Connected")
                with conn.makefile("rw") as io:
                    for line in io:
                        request = json.loads(line)
                        if request.get("reset"):
                            # A new episode started: drop the action queue and
                            # any state cached in the processor pipelines.
                            policy.reset()
                            preprocessor.reset()
                            postprocessor.reset()
                        # Keep the socket alive after a delayed request.  The
                        # Dora client turns this explicit rejection into a
                        # latched timing fault, so an older action cannot be
                        # silently reused by the arm boundary.
                        try:
                            validate_request_timing(
                                request.get("metadata", {}), time.monotonic_ns()
                            )
                        except ValueError as error:
                            io.write(
                                json.dumps(
                                    {
                                        "error": "observation_timing",
                                        "error_detail": str(error),
                                        "positions": [],
                                    }
                                )
                                + "\n"
                            )
                            io.flush()
                            continue
                        with pa.OSFile(request["data_path"], "rb") as f:
                            with pa.ipc.open_file(f) as reader:
                                obs = reader.get_batch(0).to_struct_array()[0]
                        actions = infer(policy, preprocessor, postprocessor, obs)
                        io.write(json.dumps(actions) + "\n")
                        io.flush()
        finally:
            if os.path.exists(socket_path):
                os.remove(socket_path)


if __name__ == "__main__":
    main()
