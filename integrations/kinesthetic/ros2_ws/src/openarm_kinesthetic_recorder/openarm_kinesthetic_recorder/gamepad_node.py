"""Local Linux G3V2 input for guarded Cartesian jog and physical dragging."""

from __future__ import annotations

import json
import time
import uuid

import numpy as np
import rclpy
from geometry_msgs.msg import TwistStamped
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

from openarm_kinesthetic.config import load_yaml
from openarm_kinesthetic.real import OperatorPacket
from openarm_kinesthetic.teaching.gamepad import (
    BTN_TL,
    GamepadMapper,
    LinuxJoystick,
    normalize_axis_value,
)


def continuous_marked_gripper_policy(
    mode: str, recording_active: bool, drive_gripper: bool
) -> tuple[bool, bool]:
    """Return (publish semantic marker, drive physical gripper) for A/Y."""
    marked_active = mode == "continuous_marked" and recording_active
    keyframe_semantic = mode == "keyframe" and recording_active
    drive = not keyframe_semantic and (not marked_active or drive_gripper)
    return marked_active, drive


def continuous_marked_lb_enabled(
    mode: str, recording_active: bool, enable_keypoints: bool
) -> bool:
    return mode == "continuous_marked" and recording_active and enable_keypoints


class RealGamepadNode(Node):
    def __init__(self) -> None:
        super().__init__("openarm_real_gamepad")
        self.declare_parameter("device", "auto")
        self.declare_parameter("publish_rate", 20.0)
        self.declare_parameter(
            "command_topic", "/openarm_kinesthetic/operator_command"
        )
        self.declare_parameter(
            "status_topic", "/openarm_kinesthetic/gamepad_status"
        )
        self.declare_parameter(
            "cartesian_twist_topic", "/openarm_kinesthetic/cartesian_twist"
        )
        self.declare_parameter(
            "motion_record_toggle_topic",
            "/openarm_kinesthetic/motion_record_toggle",
        )
        self.declare_parameter("cartesian_command_frame", "world")
        self.declare_parameter("diagnostic_only", False)
        self.declare_parameter("motion_recording_mode", "continuous_marked")
        self.declare_parameter(
            "motion_marker_event_topic", "/openarm_kinesthetic/motion_marker_event"
        )
        self.declare_parameter("continuous_marked_drive_gripper", False)
        self.declare_parameter("continuous_marked_enable_lb_keypoints", False)
        rate = float(self.get_parameter("publish_rate").value)
        if rate < 10.0 or rate > 100.0:
            raise ValueError("publish_rate must be in [10, 100] Hz")

        self.config = load_yaml("gamepad_control.yaml")
        self.mapper = GamepadMapper(self.config)
        self.joystick: LinuxJoystick | None = None
        self.device_request = str(self.get_parameter("device").value)
        self.session_id = uuid.uuid4().hex
        self.sequence = 0
        self.open_sequence = 0
        self.close_sequence = 0
        self.record_sequence = 0
        self.marker_sequence = 0
        self.hand_drag_toggle_opens_next = True
        self.last_connect_attempt = 0.0
        self.last_connect_error = ""
        self.last_missing_log = 0.0
        self.last_status_publish = 0.0
        self.last_summary: tuple[bool, str, bool] | None = None
        self.cartesian_command_frame = str(
            self.get_parameter("cartesian_command_frame").value
        )
        self.diagnostic_only = bool(self.get_parameter("diagnostic_only").value)
        self.motion_recording_mode = str(
            self.get_parameter("motion_recording_mode").value
        )
        if self.motion_recording_mode not in ("continuous", "keyframe", "continuous_marked"):
            raise ValueError(
                "motion_recording_mode must be continuous, keyframe, or continuous_marked"
            )
        self.continuous_marked_drive_gripper = bool(
            self.get_parameter("continuous_marked_drive_gripper").value
        )
        self.continuous_marked_enable_lb_keypoints = bool(
            self.get_parameter("continuous_marked_enable_lb_keypoints").value
        )
        self.keyframe_recording_active = False

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        topic = str(self.get_parameter("command_topic").value)
        self.publisher = self.create_publisher(String, topic, qos)
        self.raw_event_publisher = self.create_publisher(
            String, "/openarm_kinesthetic/gamepad_events", qos
        )
        self.status_publisher = self.create_publisher(
            String,
            str(self.get_parameter("status_topic").value),
            qos,
        )
        self.twist_publisher = self.create_publisher(
            TwistStamped,
            str(self.get_parameter("cartesian_twist_topic").value),
            qos,
        )
        self.record_toggle_publisher = self.create_publisher(
            String,
            str(self.get_parameter("motion_record_toggle_topic").value),
            qos,
        )
        self.create_subscription(
            String,
            "/openarm_kinesthetic/motion_record_status",
            self.on_motion_record_status,
            qos,
        )
        self.keyframe_event_publisher = self.create_publisher(
            String, "/openarm_kinesthetic/keyframe_event", qos
        )
        self.motion_marker_publisher = self.create_publisher(
            String, str(self.get_parameter("motion_marker_event_topic").value), qos
        )
        self.timer = self.create_timer(1.0 / rate, self.on_timer)
        self.get_logger().warn(
            "REAL dual-mode input: LT=guarded Cartesian jog, "
            "LT+LB=physical drag, A=open, Y=close, L3=drag gripper toggle, "
            "X=start/stop motion recording, RT/RB=software stop. "
            f"recording_mode={self.motion_recording_mode}. "
            "Physical E-stop remains required."
        )

    def on_motion_record_status(self, message: String) -> None:
        try:
            state = str(json.loads(message.data).get("state", ""))
        except (AttributeError, json.JSONDecodeError):
            return
        active = state == "RECORDING"
        if active != self.keyframe_recording_active:
            self.keyframe_recording_active = active
            mode = getattr(self, "motion_recording_mode", "keyframe")
            if mode == "keyframe":
                self.get_logger().warn(
                    "KEYFRAME_SEMANTIC_GRIPPER=%s; A/Y %s real gripper"
                    % ("ACTIVE" if active else "INACTIVE", "do not drive" if active else "control")
                )
            elif mode == "continuous_marked":
                drives = getattr(self, "continuous_marked_drive_gripper", False)
                keypoints = getattr(
                    self, "continuous_marked_enable_lb_keypoints", False
                )
                self.get_logger().warn(
                    "CONTINUOUS_MARKED=%s; A/Y=%s; standalone_LB_keypoints=%s; "
                    "L3 remains explicit physical gripper control during LT+LB"
                    % (
                        "ACTIVE" if active else "INACTIVE",
                        "physical+marker" if drives else "semantic_marker_only",
                        "enabled" if keypoints else "disabled",
                    )
                )

    def publish_motion_marker(
        self, marker_type: str, source_button: str, now: float,
        target_position_rad: float | None = None,
        result: str = "requested",
    ) -> None:
        self.marker_sequence += 1
        payload = {
            "schema_version": 1,
            "session_id": self.session_id,
            "sequence": self.marker_sequence,
            "type": marker_type,
            "source_button": source_button,
            "sent_monotonic_s": now,
            "source_timestamp_s": time.time(),
        }
        if target_position_rad is not None:
            payload["target_position_rad"] = float(target_position_rad)
            payload["result"] = result
        message = String()
        message.data = json.dumps(payload, separators=(",", ":"), sort_keys=True)
        self.motion_marker_publisher.publish(message)
        self.get_logger().info(
            "MOTION_MARKER_EVENT type=%s source=%s sequence=%d result=%s"
            % (marker_type, source_button, self.marker_sequence, payload.get("result", "n/a"))
        )

    def try_connect(self, now: float) -> None:
        if self.joystick is not None and self.joystick.connected:
            return
        if now - self.last_connect_attempt < 1.0:
            return
        self.last_connect_attempt = now
        if self.joystick is not None:
            self.joystick.close()
            self.joystick = None
        try:
            self.joystick = LinuxJoystick.open_configured(
                self.device_request, self.config
            )
        except (FileNotFoundError, PermissionError, OSError) as error:
            self.last_connect_error = str(error)
            self.joystick = None
        if self.joystick is not None:
            self.last_connect_error = ""
            axes_from_kernel = bool(
                hasattr(self.mapper, "configure_axis_codes")
                and self.mapper.configure_axis_codes(
                    getattr(self.joystick, "axis_codes", [])
                )
            )
            kernel_lb = self.joystick.button_index_for_code(BTN_TL)
            if kernel_lb is not None:
                self.mapper.button["hand_drag_lb"] = kernel_lb
            self.get_logger().info(
                f"gamepad connected: {self.joystick.path} name={self.joystick.name!r} "
                f"LB_button={self.mapper.button['hand_drag_lb']} "
                f"source={'kernel_BTN_TL' if kernel_lb is not None else 'yaml_fallback'} "
                f"axes_source={'kernel_JSIOCGAXMAP' if axes_from_kernel else 'yaml_fallback'}"
            )
            if getattr(self.mapper, "stop_latched", False):
                self.get_logger().warn(
                    "gamepad reconnected with software stop latched: release "
                    "LT/RT/RB completely, then press LT once to re-arm"
                )
        elif now - self.last_missing_log >= 5.0:
            self.last_missing_log = now
            if not self.last_connect_error:
                self.last_connect_error = "no matching /dev/input/js* device"
            detail = self.last_connect_error
            self.get_logger().warn(
                "gamepad unavailable (%s); bridge remains rigid. Check device path "
                "and membership in the Linux 'input' group." % detail
            )

    def on_timer(self) -> None:
        now = time.monotonic()
        self.try_connect(now)
        events = self.joystick.poll() if self.joystick is not None else []
        connected = bool(self.joystick is not None and self.joystick.connected)

        if events:
            for event in events:
                if event.kind == "axis" and not event.initial:
                    code, code_name = self.joystick.axis_code_for_index(event.index)
                    normalized = normalize_axis_value(event.value)
                    self.get_logger().info(
                        f"gamepad raw axis[{event.index}] code={code_name}({code}) "
                        f"value={event.value} normalized={normalized:+.4f}"
                    )
                elif event.kind == "button" and not event.initial:
                    self.get_logger().info(
                        f"gamepad raw button[{event.index}]={event.value}"
                    )
            message = String()
            message.data = json.dumps(
                {
                    "session_id": self.session_id,
                    "events": [
                        {
                            **event.to_dict(),
                            "axis_code": (
                                self.joystick.axis_code_for_index(event.index)[0]
                                if event.kind == "axis"
                                else None
                            ),
                            "axis_code_name": (
                                self.joystick.axis_code_for_index(event.index)[1]
                                if event.kind == "axis"
                                else ""
                            ),
                            "normalized": (
                                normalize_axis_value(event.value)
                                if event.kind == "axis"
                                else None
                            ),
                        }
                        for event in events
                    ],
                },
                separators=(",", ":"),
            )
            self.raw_event_publisher.publish(message)

        if connected:
            command = self.mapper.update(
                self.joystick.axes, self.joystick.buttons, connected=True
            )
        else:
            # Shape-compatible zero arrays are owned by the disconnected object
            # when available.  A first-start disconnect uses conservative sizes.
            axes = (
                self.joystick.axes
                if self.joystick is not None
                else np.zeros(8, dtype=np.int16)
            )
            buttons = (
                self.joystick.buttons
                if self.joystick is not None
                else np.zeros(15, dtype=np.uint8)
            )
            command = self.mapper.update(axes, buttons, connected=False)

        physical_drag_intent = bool(command.active and command.drag_modifier_pressed)
        semantic_gripper = (
            self.motion_recording_mode == "keyframe"
            and self.keyframe_recording_active
        )
        marked_drives_gripper = getattr(
            self, "continuous_marked_drive_gripper", False
        )
        marked_active, ay_drives_gripper = continuous_marked_gripper_policy(
            self.motion_recording_mode,
            self.keyframe_recording_active,
            marked_drives_gripper,
        )
        if command.open_pressed:
            if marked_active:
                self.publish_motion_marker(
                    "gripper_open", "A", now, -0.35,
                    "requested" if marked_drives_gripper else "semantic_only",
                )
            if ay_drives_gripper:
                self.open_sequence += 1
                self.hand_drag_toggle_opens_next = False
                self.get_logger().info(
                    f"gamepad A: request right gripper OPEN sequence={self.open_sequence}"
                )
            elif marked_active:
                self.get_logger().info(
                    "gamepad A: semantic gripper_open marker only; physical gripper unchanged"
                )
        if command.close_pressed:
            if marked_active:
                self.publish_motion_marker(
                    "gripper_close", "Y", now, -0.10,
                    "requested" if marked_drives_gripper else "semantic_only",
                )
            if ay_drives_gripper:
                self.close_sequence += 1
                self.hand_drag_toggle_opens_next = True
                self.get_logger().info(
                    f"gamepad Y: request right gripper CLOSE sequence={self.close_sequence}"
                )
            elif marked_active:
                self.get_logger().info(
                    "gamepad Y: semantic gripper_close marker only; physical gripper unchanged"
                )
        # A/Y remain semantic-only during keyframe recording.  L3 is the
        # unambiguous physical gripper control while LT+LB drag is active, so
        # keep it available both during and outside a recording.
        if physical_drag_intent and command.hand_drag_gripper_toggle_pressed:
            if self.hand_drag_toggle_opens_next:
                self.open_sequence += 1
                label = "OPEN"
                marker_type = "gripper_open"
                marker_target = -0.35
            else:
                self.close_sequence += 1
                label = "CLOSE"
                marker_type = "gripper_close"
                marker_target = -0.10
            self.hand_drag_toggle_opens_next = not self.hand_drag_toggle_opens_next
            self.get_logger().info(
                f"gamepad L3 during LT+LB: request right gripper {label} "
                f"open_sequence={self.open_sequence} close_sequence={self.close_sequence}"
            )
            if self.motion_recording_mode == "continuous_marked" and self.keyframe_recording_active:
                self.publish_motion_marker(marker_type, "L3", now, marker_target)
        if command.record_toggle_pressed:
            self.record_sequence += 1
            toggle = String()
            toggle.data = json.dumps(
                {
                    "schema_version": 1,
                    "session_id": self.session_id,
                    "sequence": self.record_sequence,
                    "sent_monotonic_s": now,
                },
                separators=(",", ":"),
                sort_keys=True,
            )
            self.record_toggle_publisher.publish(toggle)
            self.get_logger().info(
                f"gamepad X: toggle motion recording sequence={self.record_sequence}"
            )

        if semantic_gripper:
            keyframe_event = None
            if command.waypoint_capture_pressed:
                keyframe_event = "capture"
            elif command.close_pressed:
                keyframe_event = "close"
            elif command.open_pressed:
                keyframe_event = "open"
            if keyframe_event is not None:
                event = String()
                event.data = json.dumps(
                    {
                        "schema_version": 1,
                        "session_id": self.session_id,
                        "sequence": self.sequence,
                        "event": keyframe_event,
                        "sent_monotonic_s": now,
                    },
                    separators=(",", ":"),
                    sort_keys=True,
                )
                self.keyframe_event_publisher.publish(event)
                self.get_logger().info(
                    f"KEYFRAME_EVENT event={keyframe_event} sequence={self.sequence}"
                )
        elif (
            continuous_marked_lb_enabled(
                self.motion_recording_mode,
                self.keyframe_recording_active,
                getattr(self, "continuous_marked_enable_lb_keypoints", False),
            )
            and command.waypoint_capture_pressed
        ):
            self.publish_motion_marker("keypoint", "LB", now)

        if physical_drag_intent:
            selected_control_mode = "physical_kinesthetic"
        elif command.active:
            selected_control_mode = "cartesian_jog"
        else:
            selected_control_mode = "idle"
        control_mode = "idle" if self.diagnostic_only else selected_control_mode

        if selected_control_mode == "cartesian_jog":
            twist = TwistStamped()
            twist.header.stamp = self.get_clock().now().to_msg()
            twist.header.frame_id = self.cartesian_command_frame
            linear = command.linear_velocity_world_m_s
            angular = command.angular_velocity_local_rad_s
            twist.twist.linear.x, twist.twist.linear.y, twist.twist.linear.z = (
                float(linear[0]), float(linear[1]), float(linear[2])
            )
            twist.twist.angular.x, twist.twist.angular.y, twist.twist.angular.z = (
                float(angular[0]), float(angular[1]), float(angular[2])
            )
            self.twist_publisher.publish(twist)

        packet = OperatorPacket(
            schema_version=1,
            session_id=self.session_id,
            sequence=self.sequence,
            sent_monotonic_s=now,
            connected=connected and not self.diagnostic_only,
            deadman_pressed=command.deadman_pressed and not self.diagnostic_only,
            active=command.active and not self.diagnostic_only,
            stop_latched=command.stop_latched or self.diagnostic_only,
            open_sequence=self.open_sequence,
            close_sequence=self.close_sequence,
            arm_motion_source=control_mode,
        )
        self.sequence += 1
        output = String()
        output.data = packet.to_json()
        self.publisher.publish(output)

        summary = (connected, control_mode, command.stop_latched)
        if summary != self.last_summary:
            self.last_summary = summary
            self.get_logger().info(
                "connected=%s control_mode=%s stop=%s open_seq=%d close_seq=%d"
                % (*summary, self.open_sequence, self.close_sequence)
            )
        if now - self.last_status_publish >= 0.5:
            self.last_status_publish = now
            status = String()
            status.data = json.dumps(
                {
                    "source": "local_linux_joystick",
                    "requested_device": self.device_request,
                    "connected": connected,
                    "device_path": self.joystick.path if connected else "",
                    "device_name": self.joystick.name if connected else "",
                    "connect_error": self.last_connect_error,
                    "deadman_pressed": command.deadman_pressed,
                    "active": command.active,
                    "drag_modifier_pressed": command.drag_modifier_pressed,
                    "hand_drag_gripper_toggle_pressed": (
                        command.hand_drag_gripper_toggle_pressed
                    ),
                    "record_sequence": self.record_sequence,
                    "drag_modifier_button_index": self.mapper.button["hand_drag_lb"],
                    "pressed_button_indices": (
                        np.flatnonzero(self.joystick.buttons).astype(int).tolist()
                        if connected
                        else []
                    ),
                    "control_mode": control_mode,
                    "selected_control_mode": selected_control_mode,
                    "diagnostic_only": self.diagnostic_only,
                    "raw_axes": (
                        self.joystick.axes.astype(int).tolist() if connected else []
                    ),
                    "normalized_axes": (
                        [
                            self.mapper._axis_value(self.joystick.axes, index)
                            for index in range(len(self.joystick.axes))
                        ]
                        if connected
                        else []
                    ),
                    "axis_codes": (
                        [
                            self.joystick.axis_code_for_index(index)[1]
                            for index in range(len(self.joystick.axes))
                        ]
                        if connected
                        else []
                    ),
                    "cartesian_command_before_transform": [
                        *command.linear_velocity_world_m_s.astype(float).tolist(),
                        *command.angular_velocity_local_rad_s.astype(float).tolist(),
                    ],
                    "stop_latched": command.stop_latched,
                    "open_sequence": self.open_sequence,
                    "close_sequence": self.close_sequence,
                    "arm_motion_source": control_mode,
                },
                separators=(",", ":"),
                sort_keys=True,
            )
            self.status_publisher.publish(status)

    def publish_shutdown(self) -> None:
        packet = OperatorPacket(
            schema_version=1,
            session_id=self.session_id,
            sequence=self.sequence,
            sent_monotonic_s=time.monotonic(),
            connected=False,
            deadman_pressed=False,
            active=False,
            stop_latched=True,
            open_sequence=self.open_sequence,
            close_sequence=self.close_sequence,
            arm_motion_source="idle",
        )
        message = String()
        message.data = packet.to_json()
        for _ in range(3):
            self.publisher.publish(message)
            time.sleep(0.02)
        if self.joystick is not None:
            self.joystick.close()

    def close_device(self) -> None:
        if self.joystick is not None:
            self.joystick.close()


def main() -> int:
    rclpy.init()
    node = RealGamepadNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.publish_shutdown()
        else:
            node.close_device()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0
