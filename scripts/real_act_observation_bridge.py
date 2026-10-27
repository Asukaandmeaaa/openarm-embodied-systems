#!/usr/bin/env python3
"""Save fresh ROS observations to shared memory for read-only ACT inference."""

import argparse
import json
import math
import os
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, JointState


ORDER = [
    name
    for side in ("right", "left")
    for name in ([f"openarm_{side}_joint{i}" for i in range(1, 8)] + [f"openarm_{side}_finger_joint1"])
]


class Bridge(Node):
    def __init__(self, output: Path, rate: float, max_age: float, history_path: Path, keep_images: bool):
        super().__init__("real_act_observation_bridge")
        self.output = output
        self.output.mkdir(parents=True, exist_ok=True)
        self.max_age = max_age
        self.history_path = history_path
        self.keep_images = keep_images
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        self.latest = {}
        self.sequence = 0
        self.create_subscription(JointState, "/joint_states", self.on_state, qos_profile_sensor_data)
        self.create_subscription(
            CompressedImage,
            "/camera/global_camera/color/image_raw/compressed",
            lambda m: self.on_image("ceiling", m),
            qos_profile_sensor_data,
        )
        self.create_subscription(
            CompressedImage,
            "/camera/wrist_right/image_raw/compressed",
            lambda m: self.on_image("wrist_right", m),
            qos_profile_sensor_data,
        )
        self.create_timer(1.0 / rate, self.save)

    def on_state(self, msg):
        positions = dict(zip(msg.name, msg.position))
        if all(name in positions for name in ORDER):
            state = [float(positions[name]) for name in ORDER]
            if all(math.isfinite(value) for value in state):
                self.latest["state"] = (time.monotonic(), state)

    def on_image(self, key, msg):
        if msg.data and "jpeg" in msg.format.lower():
            self.latest[key] = (time.monotonic(), bytes(msg.data))

    def save(self):
        now = time.monotonic()
        keys = ("state", "ceiling", "wrist_right")
        if any(key not in self.latest for key in keys):
            return
        ages = {key: now - self.latest[key][0] for key in keys}
        if any(age < 0 or age > self.max_age for age in ages.values()):
            return
        self.sequence += 1
        number = self.sequence
        image_paths = {}
        for key in ("ceiling", "wrist_right"):
            name = f"{number:08d}_{key}.jpg"
            path = self.output / name
            path.write_bytes(self.latest[key][1])
            image_paths[key] = name
        payload = {
            "sequence": number,
            "monotonic_s": now,
            "ages_s": ages,
            "joint_order": ORDER,
            "state": self.latest["state"][1],
            "images": image_paths,
        }
        temp = self.output / "latest.json.tmp"
        temp.write_text(json.dumps(payload))
        os.replace(temp, self.output / "latest.json")
        with self.history_path.open("a", encoding="utf-8") as history:
            history.write(json.dumps(payload, separators=(",", ":")) + "\n")
        old = number - 3
        if old > 0 and not self.keep_images:
            for key in ("ceiling", "wrist_right"):
                (self.output / f"{old:08d}_{key}.jpg").unlink(missing_ok=True)
        if number % 25 == 0:
            self.get_logger().info(f"snapshots={number}, ages_ms={{{', '.join(f'{k}: {v*1000:.0f}' for k,v in ages.items())}}}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("/dev/shm/act_live"))
    parser.add_argument("--rate", type=float, default=5.0)
    parser.add_argument("--max-age", type=float, default=0.25)
    parser.add_argument(
        "--history-path",
        type=Path,
        default=None,
        help="append every accepted observation manifest to this JSONL path",
    )
    parser.add_argument(
        "--keep-images",
        action="store_true",
        help="retain all observation JPEGs instead of the default 3-snapshot ring",
    )
    args = parser.parse_args()
    if args.rate <= 0 or args.max_age <= 0:
        parser.error("rate and max-age must be positive")
    rclpy.init()
    history_path = args.history_path or (args.output / "observation_history.jsonl")
    node = Bridge(args.output, args.rate, args.max_age, history_path, args.keep_images)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
