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

import pytest

import yaml

from dora_openarm_dataset_recorder.main import (
    DatasetWriter,
    Episode,
    FrequencyDetector,
    _should_record,
)

CONFIGS = [
    {
        "id": "ui",
        "name": None,
        "description": None,
        "path": "dora-openarm-data-collection-ui",
        "env": {"METADATA_FILE": "metadata.yaml"},
        "outputs": ["command"],
        "inputs": {"tick": "dora/timer/secs/1"},
        "build": "pip install -e ..",
        "restart_policy": "never",
        "_unstable_deploy": None,
    },
    {
        "id": "quittable-tick-leader",
        "name": None,
        "description": None,
        "path": "dora-openarm-quitter",
        "env": None,
        "outputs": ["tick"],
        "inputs": {"command": "ui/command", "tick": "dora/timer/millis/4"},
        "build": "pip install dora-openarm-quitter",
        "restart_policy": "never",
        "_unstable_deploy": None,
    },
    {
        "id": "quittable-tick-camera",
        "name": None,
        "description": None,
        "path": "dora-openarm-quitter",
        "env": None,
        "outputs": ["tick"],
        "inputs": {"command": "ui/command", "tick": "dora/timer/millis/33"},
        "build": "pip install dora-openarm-quitter",
        "restart_policy": "never",
        "_unstable_deploy": None,
    },
    {
        "id": "leader",
        "name": None,
        "description": None,
        "path": "dora-openarm-dummy-ker",
        "env": None,
        "outputs": [
            "joystick_button",
            "joystick_x",
            "joystick_y",
            "left_follower_position",
            "left_position",
            "right_follower_position",
            "right_position",
        ],
        "inputs": {"tick": "quittable-tick-leader/tick"},
        "build": "pip install dora-openarm-dummy-ker",
        "restart_policy": "never",
        "_unstable_deploy": None,
    },
    {
        "id": "follower-right",
        "name": None,
        "description": None,
        "path": "dora-openarm-dummy",
        "env": None,
        "outputs": ["position", "status"],
        "inputs": {
            "move_position": "leader/right_follower_position",
            "request_position": "leader/right_follower_position",
        },
        "build": "pip install dora-openarm-dummy",
        "restart_policy": "never",
        "_unstable_deploy": None,
    },
    {
        "id": "camera-wrist-right",
        "name": None,
        "description": None,
        "path": "dora-openarm-dummy-camera",
        "env": {
            "ENCODING": "jpeg",
            "IMAGE_HEIGHT": 600,
            "IMAGE_WIDTH": 960,
            "JPEG_QUALITY": 90,
        },
        "outputs": ["image"],
        "inputs": {"tick": "quittable-tick-camera/tick"},
        "build": "pip install dora-openarm-dummy-camera",
        "restart_policy": "never",
        "_unstable_deploy": None,
    },
    {
        "id": "follower-left",
        "name": None,
        "description": None,
        "path": "dora-openarm-dummy",
        "env": None,
        "outputs": ["position", "status"],
        "inputs": {
            "move_position": "leader/left_follower_position",
            "request_position": "leader/left_follower_position",
        },
        "build": "pip install dora-openarm-dummy",
        "restart_policy": "never",
        "_unstable_deploy": None,
    },
    {
        "id": "camera-wrist-left",
        "name": None,
        "description": None,
        "path": "dora-openarm-dummy-camera",
        "env": {
            "ENCODING": "jpeg",
            "IMAGE_HEIGHT": 600,
            "IMAGE_WIDTH": 960,
            "JPEG_QUALITY": 90,
        },
        "outputs": ["image"],
        "inputs": {"tick": "quittable-tick-camera/tick"},
        "build": "pip install dora-openarm-dummy-camera",
        "restart_policy": "never",
        "_unstable_deploy": None,
    },
    {
        "id": "camera-head",
        "name": None,
        "description": None,
        "path": "dora-openarm-dummy-camera",
        "env": {
            "ENCODING": "jpeg",
            "IMAGE_HEIGHT": 600,
            "IMAGE_WIDTH": 960,
            "JPEG_QUALITY": 90,
        },
        "outputs": ["image"],
        "inputs": {"tick": "quittable-tick-camera/tick"},
        "build": "pip install dora-openarm-dummy-camera",
        "restart_policy": "never",
        "_unstable_deploy": None,
    },
    {
        "id": "camera-ceiling",
        "name": None,
        "description": None,
        "path": "dora-openarm-dummy-camera",
        "env": {
            "ENCODING": "jpeg",
            "IMAGE_HEIGHT": 600,
            "IMAGE_WIDTH": 960,
            "JPEG_QUALITY": 90,
        },
        "outputs": ["image"],
        "inputs": {"tick": "quittable-tick-camera/tick"},
        "build": "pip install dora-openarm-dummy-camera",
        "restart_policy": "never",
        "_unstable_deploy": None,
    },
    {
        "id": "recorder",
        "name": None,
        "description": None,
        "path": "dora-openarm-dataset-recorder",
        "env": {"METADATA_FILE": "metadata.yaml"},
        "outputs": [],
        "inputs": {
            "arm_left_action": "leader/left_follower_position",
            "arm_left_observation": "follower-left/position",
            "arm_right_action": "leader/right_follower_position",
            "arm_right_observation": "follower-right/position",
            "camera_ceiling": "camera-ceiling/image",
            "camera_head": "camera-head/image",
            "camera_wrist_left": "camera-wrist-left/image",
            "camera_wrist_right": "camera-wrist-right/image",
            "command": "ui/command",
        },
        "build": "pip install dora-openarm-dataset-recorder",
        "restart_policy": "never",
        "_unstable_deploy": None,
    },
]


def test_arm_left_action():
    detector = FrequencyDetector(CONFIGS)
    assert detector.detect("leader/left_follower_position") == pytest.approx(250.0)


def test_arm_left_observation():
    detector = FrequencyDetector(CONFIGS)
    assert detector.detect("follower-left/position") == pytest.approx(250.0)


def test_camera_ceiling():
    detector = FrequencyDetector(CONFIGS)
    assert detector.detect("camera-ceiling/image") == pytest.approx(1_000.0 / 33)


def test_record_hz_limits_each_stream_independently():
    episode = Episode()
    assert _should_record(episode, "arm_right_action", 0, 30.0)
    assert not _should_record(episode, "arm_right_action", 10_000_000, 30.0)
    assert _should_record(episode, "arm_left_action", 10_000_000, 30.0)
    assert _should_record(episode, "arm_right_action", 34_000_000, 30.0)


def test_episode_provenance_is_written_to_metadata(tmp_path):
    writer = DatasetWriter(tmp_path, "pilot", {"tasks": [{"prompt": "pick"}]})
    episode = Episode(number=3, success=True)
    episode.metadata = {
        "seed": 20260902,
        "reset_index": 3,
        "objects": {"orange_cube": {"qpos": [0.45, 0.0, 1.05, 1, 0, 0, 0]}},
    }
    writer.finish_episode(episode)
    metadata = yaml.safe_load((tmp_path / "pilot" / "metadata.yaml").read_text())
    assert metadata["episodes"][0]["seed"] == 20260902
    assert metadata["episodes"][0]["objects"]["orange_cube"]["qpos"][2] == 1.05
