"""Compact read-only status console for attended kinesthetic bringup."""

from __future__ import annotations

import json

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, String


def combined_status(
    gamepad: dict | None,
    bridge: dict | None,
    hardware_passive: bool | None,
) -> dict:
    gamepad = gamepad or {}
    bridge = bridge or {}
    return {
        "gamepad_connected": gamepad.get("connected"),
        "deadman_pressed": gamepad.get("deadman_pressed"),
        "control_mode": bridge.get("control_mode", gamepad.get("control_mode")),
        "operator_command_fresh": bridge.get("command_fresh"),
        "operator_armed": bridge.get("operator_armed"),
        "joint_state_fresh": bridge.get("joint_state_fresh"),
        "passive_enabled": bridge.get("passive_enabled"),
        "hardware_passive": hardware_passive,
        "interlock_latched": bridge.get("interlock_latched"),
        "stop_reason": bridge.get("stop_reason"),
    }


class TeachingStatusNode(Node):
    def __init__(self) -> None:
        super().__init__("teaching_status_console")
        self.declare_parameter(
            "gamepad_status_topic", "/openarm_kinesthetic/gamepad_status"
        )
        self.declare_parameter(
            "bridge_status_topic", "/openarm_kinesthetic/hardware_status"
        )
        self.declare_parameter(
            "hardware_passive_topic", "/openarm_right_hardware/passive_state"
        )
        self.declare_parameter("summary_rate", 2.0)
        rate = float(self.get_parameter("summary_rate").value)
        if not 0.5 <= rate <= 10.0:
            raise ValueError("summary_rate must be in [0.5, 10.0] Hz")

        reliable = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        latched = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.gamepad: dict | None = None
        self.bridge: dict | None = None
        self.hardware_passive: bool | None = None
        self.last_output = ""
        self.create_subscription(
            String,
            str(self.get_parameter("gamepad_status_topic").value),
            self._gamepad,
            reliable,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("bridge_status_topic").value),
            self._bridge,
            reliable,
        )
        self.create_subscription(
            Bool,
            str(self.get_parameter("hardware_passive_topic").value),
            self._passive,
            latched,
        )
        self.create_timer(1.0 / rate, self._publish_summary)

    @staticmethod
    def _decode(message: String) -> dict | None:
        try:
            value = json.loads(message.data)
        except (json.JSONDecodeError, TypeError):
            return None
        return value if isinstance(value, dict) else None

    def _gamepad(self, message: String) -> None:
        self.gamepad = self._decode(message)

    def _bridge(self, message: String) -> None:
        self.bridge = self._decode(message)

    def _passive(self, message: Bool) -> None:
        self.hardware_passive = bool(message.data)

    def _publish_summary(self) -> None:
        payload = combined_status(
            self.gamepad,
            self.bridge,
            self.hardware_passive,
        )
        output = json.dumps(payload, separators=(",", ":"), sort_keys=True)
        if output == self.last_output:
            return
        self.last_output = output
        if payload["hardware_passive"] is True:
            self.get_logger().warn(f"TEACHING_STATUS {output}")
        else:
            self.get_logger().info(f"TEACHING_STATUS {output}")


def main() -> int:
    rclpy.init()
    node = TeachingStatusNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
