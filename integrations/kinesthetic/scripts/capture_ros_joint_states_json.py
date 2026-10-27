#!/usr/bin/env python3
"""Capture one fresh /joint_states sample as a name-to-position JSON mapping.

This utility only subscribes to sensor_msgs/JointState.  It creates no service or action
client and cannot command either arm.  The output format is accepted by
``audit_can_ros_joint_snapshot.py``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState


class JointStateCapture(Node):
    def __init__(self, topic: str) -> None:
        super().__init__("openarm_joint_state_capture")
        self.mapping: dict[str, float] | None = None
        self.create_subscription(
            JointState, topic, self._on_joint_state, qos_profile_sensor_data
        )

    def _on_joint_state(self, message: JointState) -> None:
        if len(message.name) != len(message.position) or not message.name:
            return
        if len(set(message.name)) != len(message.name):
            return
        values = [float(value) for value in message.position]
        if not all(math.isfinite(value) for value in values):
            return
        self.mapping = dict(zip(message.name, values))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topic", default="/joint_states")
    parser.add_argument("--timeout-s", type=float, default=10.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not math.isfinite(args.timeout_s) or args.timeout_s <= 0.0:
        raise ValueError("--timeout-s must be positive and finite")

    rclpy.init()
    node = JointStateCapture(args.topic)
    try:
        deadline = time.monotonic() + args.timeout_s
        while rclpy.ok() and node.mapping is None and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
        if node.mapping is None:
            raise RuntimeError(f"fresh {args.topic} sample unavailable/timeout")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(node.mapping, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(node.mapping, ensure_ascii=False, indent=2))
        return 0
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
