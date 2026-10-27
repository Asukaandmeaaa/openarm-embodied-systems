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

"""Regression tests for parallel observer image decoding."""

from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pyarrow as pa

from dora_openarm_observer.main import _build_output


def _camera_event(value):
    image = np.full((2, 3, 3), value, dtype=np.uint8)
    encoded, jpeg = cv2.imencode(".jpg", image)
    assert encoded
    return {"value": pa.array(jpeg, type=pa.uint8())}


def test_parallel_decode_preserves_mainline_output_format():
    """Parallel decoding keeps the existing Arrow schema and qpos values."""
    observation = {
        "arm_right": {"value": pa.array([1.0, 2.0], type=pa.float32())},
        "arm_left": {"value": pa.array([3.0, 4.0], type=pa.float32())},
        "camera_wrist_right": _camera_event(10),
        "camera_wrist_left": _camera_event(20),
        "camera_head_left": _camera_event(30),
        "camera_head_right": _camera_event(40),
        "camera_ceiling": _camera_event(50),
        "id": 7,
    }
    metadata = {}

    with ThreadPoolExecutor(max_workers=3) as decode_pool:
        output = _build_output(observation, None, "pick", metadata, decode_pool)

    assert output.type.names == [
        "position",
        "camera_wrist_right",
        "camera_wrist_left",
        "camera_head_left",
        "camera_head_right",
        "camera_ceiling",
        "phase_classifier_result",
        "task_prompt",
        "id",
    ]
    assert output.field("position").to_pylist() == [[1.0, 2.0, 3.0, 4.0]]
    assert output.field("position").type == pa.list_(pa.float32())
    assert output.field("camera_ceiling").type == pa.list_(pa.uint8())
    assert output.field("task_prompt").to_pylist() == ["pick"]
    assert output.field("id").to_pylist() == [7]
    assert metadata["camera_ceiling.encoding"] == "rgb8"
    assert metadata["camera_ceiling.height"] == 2
    assert metadata["camera_ceiling.width"] == 3
