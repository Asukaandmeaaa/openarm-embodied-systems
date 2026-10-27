"""Nano-side fail-rigid bridge for direct right-arm kinesthetic teaching."""

from __future__ import annotations

import json
import math
import threading
import time
from enum import Enum

import numpy as np
import rclpy
from controller_manager_msgs.srv import ListControllers, SwitchController
from geometry_msgs.msg import TwistStamped
from moveit_msgs.msg import RobotState
from moveit_msgs.srv import GetStateValidity
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, String
from std_srvs.srv import SetBool
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from openarm_kinesthetic.real import OperatorPacket, OperatorWatchdog
from openarm_human_retarget.kinematics import OpenArmRightKinematics


ARM_DOF = 7
GRIPPER_GOAL_TOLERANCE_RAD = 0.02
GRIPPER_COMMAND_TIMEOUT_S = 5.0
CARTESIAN_MINIMUM_PARALLEL_SCALE = 0.25
CARTESIAN_MAXIMUM_ORTHOGONAL_RATIO = 0.15
CARTESIAN_MAX_COMMAND_LEAD_RAD = 0.025


class PassiveState(str, Enum):
    RIGID = "RIGID"
    ENTERING_PASSIVE = "ENTERING_PASSIVE"
    PASSIVE = "PASSIVE"
    EXITING_PASSIVE = "EXITING_PASSIVE"
    RIGID_HOLD_CURRENT = "RIGID_HOLD_CURRENT"
    PASSIVE_CONTROLLER_RECOVERY_REQUIRED = "PASSIVE_CONTROLLER_RECOVERY_REQUIRED"
    PASSIVE_STATE_RECOVERY_REQUIRED = "PASSIVE_STATE_RECOVERY_REQUIRED"


def measured_position_is_recoverable(measured: np.ndarray) -> tuple[bool, str]:
    """Any finite encoder pose may become a rigid hold target after hand drag.

    URDF limits are planning limits, not a reliable hardware-stop detector.  A
    manually reached encoder pose must be held safely first; MoveIt remains
    responsible for accepting or rejecting any later point-to-point motion.
    """
    measured = np.asarray(measured, dtype=float)
    if measured.shape != (ARM_DOF,) or not np.all(np.isfinite(measured)):
        return False, "measured_position_invalid"
    return True, "none"


def cartesian_joint_velocity(
    kinematics,
    measured: np.ndarray,
    twist_world_linear_local_angular: np.ndarray,
    *,
    damping: float,
    maximum_joint_velocity: float,
) -> tuple[np.ndarray, float]:
    """Map the simulation gamepad frame contract to a bounded joint velocity."""
    measured = np.asarray(measured, dtype=float)
    twist = np.asarray(twist_world_linear_local_angular, dtype=float)
    if measured.shape != (ARM_DOF,) or twist.shape != (6,):
        raise ValueError("Cartesian mapping requires seven joints and a six-axis twist")
    if not np.all(np.isfinite(measured)) or not np.all(np.isfinite(twist)):
        raise ValueError("Cartesian mapping inputs must be finite")
    state = kinematics.forward(measured)
    angular_world = state.ee[:3, :3] @ twist[3:]
    spatial_velocity = np.concatenate((twist[:3], angular_world))
    jacobian = kinematics.jacobian(measured)
    condition = float(np.linalg.cond(jacobian))
    damping_squared = float(damping) * float(damping)
    qdot = jacobian.T @ np.linalg.solve(
        jacobian @ jacobian.T + damping_squared * np.eye(6),
        spatial_velocity,
    )
    # Scale the complete solution instead of clipping joints independently.
    # Per-joint clipping changes J*qdot's direction and was visible on hardware
    # as unintended Z/rotation during a pure X or Y command.
    peak = float(np.max(np.abs(qdot)))
    limit = float(maximum_joint_velocity)
    if peak > limit:
        qdot *= limit / peak
    return qdot, condition


def cartesian_direction_fidelity(
    requested_world: np.ndarray, achieved_world: np.ndarray
) -> tuple[float, float]:
    """Measure commanded progress and unintended task-space cross motion."""
    requested = np.asarray(requested_world, dtype=float)
    achieved = np.asarray(achieved_world, dtype=float)
    if requested.shape != (6,) or achieved.shape != (6,):
        raise ValueError("Cartesian fidelity requires two six-axis vectors")
    norm_squared = float(requested @ requested)
    if norm_squared <= 1.0e-12:
        return 0.0, 0.0
    parallel_scale = float(achieved @ requested) / norm_squared
    orthogonal = achieved - parallel_scale * requested
    orthogonal_ratio = float(np.linalg.norm(orthogonal)) / math.sqrt(norm_squared)
    return parallel_scale, orthogonal_ratio


def directional_joint_limit_step(
    commanded: np.ndarray,
    joint_velocity: np.ndarray,
    period: float,
    lower: np.ndarray,
    upper: np.ndarray,
    margin: float,
) -> np.ndarray:
    """Integrate one step without trapping a pose already inside a soft margin.

    A blanket ``lower + margin <= target <= upper - margin`` check makes every
    Cartesian direction unavailable when a manually placed arm starts inside a
    margin (motor zero is the common case for joint 4).  The margin is a
    directional guard: motion toward a nearby limit is suppressed, while motion
    away from it remains available.  The URDF hard limits are never crossed.
    """
    commanded = np.asarray(commanded, dtype=float)
    delta = np.asarray(joint_velocity, dtype=float) * float(period)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    if any(
        array.shape != (ARM_DOF,)
        for array in (commanded, delta, lower, upper)
    ):
        raise ValueError("joint-limit step requires seven-element vectors")
    if not all(
        np.all(np.isfinite(array))
        for array in (commanded, delta, lower, upper)
    ):
        raise ValueError("joint-limit step inputs must be finite")

    lower_guard = lower + float(margin)
    upper_guard = upper - float(margin)
    delta = delta.copy()
    delta[(commanded <= lower_guard) & (delta < 0.0)] = 0.0
    delta[(commanded >= upper_guard) & (delta > 0.0)] = 0.0
    return np.clip(commanded + delta, lower, upper)


def bounded_cartesian_command_seed(
    measured: np.ndarray,
    commanded: np.ndarray,
    maximum_lead_rad: float = CARTESIAN_MAX_COMMAND_LEAD_RAD,
) -> np.ndarray:
    """Keep the rolling jog target close to fresh encoder feedback.

    The rolling target is needed so frequent one-point JTC goals still make
    progress, but it must not run away while hardware is tracking a previous
    goal.  Excess lead caused delayed, coupled motion after the stick/D-pad was
    already released on the real arm.
    """
    measured = np.asarray(measured, dtype=float)
    commanded = np.asarray(commanded, dtype=float)
    if measured.shape != (ARM_DOF,) or commanded.shape != (ARM_DOF,):
        raise ValueError("Cartesian command seed requires seven joint values")
    if not np.all(np.isfinite(measured)) or not np.all(np.isfinite(commanded)):
        raise ValueError("Cartesian command seed must be finite")
    maximum_lead_rad = float(maximum_lead_rad)
    if not math.isfinite(maximum_lead_rad) or maximum_lead_rad <= 0.0:
        raise ValueError("maximum Cartesian command lead must be positive")
    return np.clip(
        commanded,
        measured - maximum_lead_rad,
        measured + maximum_lead_rad,
    )


class RightKinestheticBridge(Node):
    def __init__(self) -> None:
        super().__init__("right_kinesthetic_bridge")
        self.declare_parameter("publish_rate", 100.0)
        self.declare_parameter("operator_watchdog_s", 0.25)
        self.declare_parameter("joint_state_watchdog_s", 0.50)
        self.declare_parameter("maximum_measured_step_rad", 0.10)
        self.declare_parameter("rigid_sync_delay_s", 0.05)
        self.declare_parameter(
            "command_topic", "/openarm_kinesthetic/operator_command"
        )
        self.declare_parameter(
            "arm_controller", "right_joint_trajectory_controller"
        )
        self.declare_parameter("gripper_controller", "right_gripper_controller")
        self.declare_parameter(
            "hardware_service", "/openarm_right_hardware/set_passive"
        )
        self.declare_parameter(
            "hardware_heartbeat_topic",
            "/openarm_right_hardware/passive_heartbeat",
        )
        self.declare_parameter(
            "hardware_state_topic", "/openarm_right_hardware/passive_state"
        )
        self.declare_parameter("controller_manager", "/controller_manager")
        self.declare_parameter("controller_switch_timeout_s", 1.0)
        self.declare_parameter("rigid_target_tolerance_rad", 0.002)
        self.declare_parameter("diagnostic_only", False)
        self.declare_parameter(
            "cartesian_twist_topic", "/openarm_kinesthetic/cartesian_twist"
        )
        self.declare_parameter("cartesian_command_watchdog_s", 0.15)
        self.declare_parameter("cartesian_command_rate_hz", 10.0)
        self.declare_parameter("cartesian_max_joint_velocity_rad_s", 0.25)
        self.declare_parameter("cartesian_damping", 0.08)
        self.declare_parameter("cartesian_hard_singularity_condition", 80.0)
        self.declare_parameter("cartesian_joint_limit_margin_rad", 0.03)
        self.declare_parameter("cartesian_trajectory_duration_s", 0.08)
        self.declare_parameter("cartesian_state_validity_check", False)
        self.declare_parameter("state_validity_service", "/check_state_validity")
        self.declare_parameter("gripper_open_rad", -0.35)
        self.declare_parameter("gripper_close_rad", -0.10)
        self.declare_parameter("gripper_command_duration_s", 0.50)

        rate = float(self.get_parameter("publish_rate").value)
        if rate < 50.0 or rate > 200.0:
            raise ValueError("publish_rate must be in [50, 200] Hz")
        self.period = 1.0 / rate
        self.joint_watchdog = float(
            self.get_parameter("joint_state_watchdog_s").value
        )
        self.maximum_step = float(
            self.get_parameter("maximum_measured_step_rad").value
        )
        self.rigid_sync_delay = float(
            self.get_parameter("rigid_sync_delay_s").value
        )
        self.controller_switch_timeout = float(
            self.get_parameter("controller_switch_timeout_s").value
        )
        self.rigid_target_tolerance = float(
            self.get_parameter("rigid_target_tolerance_rad").value
        )
        self.diagnostic_only = bool(self.get_parameter("diagnostic_only").value)
        self.open_target = float(self.get_parameter("gripper_open_rad").value)
        self.close_target = float(self.get_parameter("gripper_close_rad").value)
        self.gripper_duration = float(
            self.get_parameter("gripper_command_duration_s").value
        )
        self.cartesian_watchdog = float(
            self.get_parameter("cartesian_command_watchdog_s").value
        )
        self.cartesian_period = 1.0 / float(
            self.get_parameter("cartesian_command_rate_hz").value
        )
        self.cartesian_max_joint_velocity = float(
            self.get_parameter("cartesian_max_joint_velocity_rad_s").value
        )
        self.cartesian_damping = float(
            self.get_parameter("cartesian_damping").value
        )
        self.cartesian_hard_condition = float(
            self.get_parameter("cartesian_hard_singularity_condition").value
        )
        self.cartesian_joint_margin = float(
            self.get_parameter("cartesian_joint_limit_margin_rad").value
        )
        self.cartesian_trajectory_duration = float(
            self.get_parameter("cartesian_trajectory_duration_s").value
        )
        self.cartesian_state_validity_check = bool(
            self.get_parameter("cartesian_state_validity_check").value
        )
        if not (-0.7854 <= self.open_target < self.close_target <= 0.0):
            raise ValueError("right gripper targets violate v2 pinch limits")

        self.operator = OperatorWatchdog(
            float(self.get_parameter("operator_watchdog_s").value)
        )
        self.arm_names = [f"openarm_right_joint{i}" for i in range(1, 8)]
        self.gripper_name = "openarm_right_finger_joint1"
        self.kinematics = OpenArmRightKinematics.from_current_xacro()
        arm_controller = str(self.get_parameter("arm_controller").value)
        gripper_controller = str(self.get_parameter("gripper_controller").value)
        # Passive-state confirmation must be dispatchable while an operator
        # callback is synchronously waiting for the hardware transition.
        self.service_group = ReentrantCallbackGroup()

        reliable = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        state_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.arm_pub = self.create_publisher(
            JointTrajectory, f"/{arm_controller}/joint_trajectory", reliable
        )
        self.gripper_pub = self.create_publisher(
            JointTrajectory, f"/{gripper_controller}/joint_trajectory", reliable
        )
        self.heartbeat_pub = self.create_publisher(
            Bool,
            str(self.get_parameter("hardware_heartbeat_topic").value),
            reliable,
        )
        self.status_pub = self.create_publisher(
            String, "/openarm_kinesthetic/hardware_status", reliable
        )
        self.create_subscription(
            String,
            str(self.get_parameter("command_topic").value),
            self.on_operator_command,
            reliable,
        )
        self.create_subscription(
            JointState, "/joint_states", self.on_joint_state, reliable
        )
        self.create_subscription(
            TwistStamped,
            str(self.get_parameter("cartesian_twist_topic").value),
            self.on_cartesian_twist,
            reliable,
        )
        self.create_subscription(
            Bool,
            str(self.get_parameter("hardware_state_topic").value),
            self.on_hardware_state,
            state_qos,
            callback_group=self.service_group,
        )

        self.hardware_client = self.create_client(
            SetBool,
            str(self.get_parameter("hardware_service").value),
            callback_group=self.service_group,
        )
        self.state_validity_client = self.create_client(
            GetStateValidity,
            str(self.get_parameter("state_validity_service").value),
            callback_group=self.service_group,
        )
        manager = str(self.get_parameter("controller_manager").value).rstrip("/")
        self.list_controllers_client = self.create_client(
            ListControllers,
            f"{manager}/list_controllers",
            callback_group=self.service_group,
        )
        self.switch_controller_client = self.create_client(
            SwitchController,
            f"{manager}/switch_controller",
            callback_group=self.service_group,
        )
        self.transition_lock = threading.Lock()
        self.state_lock = threading.Lock()
        self.cartesian_lock = threading.Lock()
        self.measured: list[float] | None = None
        self.measured_velocity: list[float] | None = None
        self.measured_gripper: float | None = None
        self.measured_at: float | None = None
        self.complete_joint_state: dict[str, float] = {}
        self.cartesian_twist: np.ndarray | None = None
        self.cartesian_twist_at: float | None = None
        self.last_cartesian_step_at = 0.0
        # The real controller must advance one continuous command state.  Using
        # the latest encoder sample as every trajectory's start made successive
        # JTC goals collapse back toward the same point whenever hardware lagged.
        self.cartesian_commanded: np.ndarray | None = None
        self.gripper_target: float | None = None
        self.gripper_target_label = ""
        self.gripper_requested_at: float | None = None
        self.gripper_last_publish_at: float | None = None
        self.last_published: list[float] | None = None
        self.passive_enabled = False
        self.hardware_passive = False
        self.transition_target: bool | None = None
        self.passive_state = PassiveState.RIGID
        self.last_transition_attempt = 0.0
        self.control_mode = "idle"
        # The motion recorder owns the controller during its automatic HOME
        # tail.  Keep the bridge alive for status, but forbid new manual goals.
        self.automatic_home_handoff = False
        # Leaving passive drag is a controller/hardware transition, not a jog
        # command.  Require one complete LT release before Cartesian jog can be
        # armed again, so LB-release while LT remains held cannot race the
        # rigid restore handshake.
        self.cartesian_rearm_required = False
        self.interlock_latched = False
        self.last_status_publish = 0.0
        self.last_stop_reason = "startup_rigid"
        self.last_cartesian_diagnostic: dict[str, object] = {}
        self.last_cartesian_diagnostic_log = 0.0
        self.timer = self.create_timer(
            self.period, self.on_timer, callback_group=self.service_group
        )
        self.create_service(
            SetBool,
            "~/automatic_home_handoff",
            self.on_automatic_home_handoff,
            callback_group=self.service_group,
        )
        self.get_logger().warn(
            "RIGHT dual-mode bridge starts RIGID: LT=guarded Cartesian "
            "control, LT+LB=physical drag; release LT before first activation."
        )

    def on_joint_state(self, message: JointState) -> None:
        index = {name: i for i, name in enumerate(message.name)}
        if any(name not in index for name in self.arm_names):
            return
        measured = [float(message.position[index[name]]) for name in self.arm_names]
        if not all(math.isfinite(value) for value in measured):
            self.interlock_latched = True
            self.last_stop_reason = "nonfinite_joint_state"
            return
        with self.state_lock:
            self.measured = measured
            if len(message.velocity) >= len(message.name):
                velocity = [
                    float(message.velocity[index[name]]) for name in self.arm_names
                ]
                self.measured_velocity = (
                    velocity if all(math.isfinite(value) for value in velocity) else None
                )
            self.measured_at = time.monotonic()
            for name, raw_value in zip(message.name, message.position):
                state_value = float(raw_value)
                if math.isfinite(state_value):
                    self.complete_joint_state[str(name)] = state_value
            if self.gripper_name in index:
                value = float(message.position[index[self.gripper_name]])
                if math.isfinite(value):
                    self.measured_gripper = value

    def on_cartesian_twist(self, message: TwistStamped) -> None:
        if message.header.frame_id != "world":
            self.interlock_latched = True
            self.last_stop_reason = "cartesian_command_frame_must_be_world"
            return
        values = np.asarray(
            [
                message.twist.linear.x,
                message.twist.linear.y,
                message.twist.linear.z,
                message.twist.angular.x,
                message.twist.angular.y,
                message.twist.angular.z,
            ],
            dtype=float,
        )
        if not np.all(np.isfinite(values)):
            self.interlock_latched = True
            self.last_stop_reason = "nonfinite_cartesian_command"
            return
        self.cartesian_twist = values
        self.cartesian_twist_at = time.monotonic()
        if self.diagnostic_only:
            self.update_cartesian_diagnostic(values)

    def update_cartesian_diagnostic(self, twist: np.ndarray) -> None:
        measured = self.measured_snapshot()
        if measured is None:
            self.last_cartesian_diagnostic = {
                "command_before_transform": [float(value) for value in twist],
                "diagnostic_error": "measured_joint_state_unavailable",
            }
            return
        kinematic_state = self.kinematics.forward(measured)
        qdot, condition = cartesian_joint_velocity(
            self.kinematics,
            measured,
            twist,
            damping=self.cartesian_damping,
            maximum_joint_velocity=self.cartesian_max_joint_velocity,
        )
        predicted_twist_world = self.kinematics.jacobian(measured) @ qdot
        with self.state_lock:
            measured_velocity = (
                None
                if self.measured_velocity is None
                else np.asarray(self.measured_velocity, dtype=float)
            )
        observed_twist_world = (
            None
            if measured_velocity is None
            else self.kinematics.jacobian(measured) @ measured_velocity
        )
        self.last_cartesian_diagnostic = {
            "command_before_transform": [float(value) for value in twist],
            "angular_after_frame_transform": [
                float(value)
                for value in (kinematic_state.ee[:3, :3] @ twist[3:])
            ],
            "ik_requested_ee_delta": [
                float(value * self.cartesian_period) for value in twist
            ],
            "final_joint_delta": [
                float(value * self.cartesian_period) for value in qdot
            ],
            "predicted_ee_twist_world": [
                float(value) for value in predicted_twist_world
            ],
            "observed_joint_velocity": (
                None
                if measured_velocity is None
                else [float(value) for value in measured_velocity]
            ),
            "observed_ee_twist_world": (
                None
                if observed_twist_world is None
                else [float(value) for value in observed_twist_world]
            ),
            "jacobian_condition": condition,
            "diagnostic_only": self.diagnostic_only,
        }
        now = time.monotonic()
        if float(np.linalg.norm(twist)) > 1.0e-6 and (
            now - self.last_cartesian_diagnostic_log >= 0.25
        ):
            self.last_cartesian_diagnostic_log = now
            self.get_logger().info(
                "CARTESIAN_DIAG requested=%s predicted_world=%s joint_delta=%s "
                "observed_qdot=%s observed_world=%s"
                % (
                    np.round(twist, 4).tolist(),
                    np.round(predicted_twist_world, 4).tolist(),
                    np.round(qdot * self.cartesian_period, 5).tolist(),
                    (
                        "unavailable"
                        if measured_velocity is None
                        else np.round(measured_velocity, 4).tolist()
                    ),
                    (
                        "unavailable"
                        if observed_twist_world is None
                        else np.round(observed_twist_world, 4).tolist()
                    ),
                )
            )

    def on_hardware_state(self, message: Bool) -> None:
        was_passive = self.hardware_passive
        self.hardware_passive = bool(message.data)
        if (
            self.hardware_passive
            and not self.passive_enabled
            and self.transition_target is not True
        ):
            self.interlock_latched = True
            self.last_stop_reason = "unexpected_hardware_passive_state"
        if (
            was_passive
            and self.passive_enabled
            and not self.hardware_passive
            and self.transition_target is not False
        ):
            self.passive_enabled = False
            self.interlock_latched = True
            self.last_stop_reason = "hardware_passive_lost"
            self.get_logger().error(
                "hardware passive state was lost; release LT before retry"
            )

    def on_operator_command(self, message: String) -> None:
        now = time.monotonic()
        try:
            packet = OperatorPacket.from_json(message.data)
            decision = self.operator.ingest(packet)
        except (ValueError, TypeError, json.JSONDecodeError) as error:
            self.interlock_latched = True
            self.last_stop_reason = f"invalid_operator_packet:{error}"
            self.reconcile(False)
            return

        # Keep observing live buttons during automatic HOME so releasing the
        # bridge cannot silently reactivate a still-held LT/LB. No manual goal
        # or passive transition is allowed while the handoff latch is set.
        if self.automatic_home_handoff:
            return

        if not decision.packet_accepted:
            return

        if not decision.request_passive and not packet.deadman_pressed:
            self.interlock_latched = False
        if decision.request_open:
            self.request_gripper(self.open_target, "OPEN", now)
        if decision.request_close:
            self.request_gripper(self.close_target, "CLOSE", now)
        self.last_stop_reason = decision.stop_reason or "none"
        requested_mode = (
            decision.control_mode if not self.interlock_latched else "idle"
        )
        self.apply_control_mode(requested_mode)

    def joint_state_is_fresh(self, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        with self.state_lock:
            return (
                self.measured is not None
                and self.measured_at is not None
                and now - self.measured_at <= self.joint_watchdog
            )

    def publish_arm_current(self) -> bool:
        with self.state_lock:
            target = list(self.measured) if self.measured is not None else None
        if target is None:
            return False
        if self.last_published is not None:
            jump = max(abs(a - b) for a, b in zip(target, self.last_published))
            if jump > self.maximum_step:
                self.interlock_latched = True
                self.last_stop_reason = f"measured_step_rejected:{jump:.4f}"
                return False
        trajectory = JointTrajectory()
        trajectory.joint_names = list(self.arm_names)
        point = JointTrajectoryPoint()
        point.positions = target
        point.velocities = [0.0] * ARM_DOF
        point.time_from_start.nanosec = int(self.period * 1e9)
        trajectory.points = [point]
        self.arm_pub.publish(trajectory)
        self.last_published = target
        return True

    def request_gripper(self, target: float, label: str, now: float | None = None) -> None:
        if self.automatic_home_handoff:
            return
        now = time.monotonic() if now is None else now
        self.gripper_target = float(target)
        self.gripper_target_label = str(label)
        self.gripper_requested_at = now
        self.gripper_last_publish_at = None
        measured = self.measured_gripper
        self.get_logger().info(
            "right gripper %s requested: target=%.3f measured=%s"
            % (
                label,
                target,
                "unavailable" if measured is None else f"{measured:.3f}",
            )
        )
        self.service_gripper_command(now)

    def publish_gripper(self, target: float, label: str) -> bool:
        if self.gripper_pub.get_subscription_count() < 1:
            self.get_logger().error(f"{label} rejected: gripper controller missing")
            return False
        trajectory = JointTrajectory()
        trajectory.joint_names = [self.gripper_name]
        point = JointTrajectoryPoint()
        point.positions = [float(target)]
        duration_ns = int(self.gripper_duration * 1e9)
        point.time_from_start.sec = duration_ns // 1_000_000_000
        point.time_from_start.nanosec = duration_ns % 1_000_000_000
        trajectory.points = [point]
        self.gripper_pub.publish(trajectory)
        return True

    def service_gripper_command(self, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        target = self.gripper_target
        requested_at = self.gripper_requested_at
        if target is None or requested_at is None:
            return
        measured = self.measured_gripper
        if measured is not None and abs(measured - target) <= GRIPPER_GOAL_TOLERANCE_RAD:
            self.get_logger().info(
                f"right gripper {self.gripper_target_label} reached: "
                f"target={target:.3f} measured={measured:.3f}"
            )
            self.gripper_target = None
            self.gripper_requested_at = None
            self.gripper_last_publish_at = None
            return
        if now - requested_at > GRIPPER_COMMAND_TIMEOUT_S:
            self.get_logger().error(
                "right gripper %s timed out: target=%.3f measured=%s subscriber_count=%d"
                % (
                    self.gripper_target_label,
                    target,
                    "unavailable" if measured is None else f"{measured:.3f}",
                    self.gripper_pub.get_subscription_count(),
                )
            )
            self.gripper_target = None
            self.gripper_requested_at = None
            self.gripper_last_publish_at = None
            return
        # An operator edge produces exactly one trajectory.  Republishing the
        # same spline used to reset the controller every 600 ms, which caused
        # real-hardware stalls and hid contact-detector TIMEOUTs.
        if self.gripper_last_publish_at is None:
            self.publish_gripper(target, self.gripper_target_label)
            self.gripper_last_publish_at = now

    def apply_control_mode(self, requested_mode: str) -> None:
        if self.automatic_home_handoff and requested_mode != "idle":
            return
        if requested_mode not in ("idle", "cartesian_jog", "physical_kinesthetic"):
            self.interlock_latched = True
            self.last_stop_reason = f"unsupported_control_mode:{requested_mode}"
            requested_mode = "idle"
        if (
            self.control_mode == "physical_kinesthetic"
            and requested_mode == "cartesian_jog"
        ):
            self.cartesian_rearm_required = True
            self.last_stop_reason = "release_lt_after_physical_drag"
            requested_mode = "idle"
            self.get_logger().warn(
                "PHYSICAL DRAG ENDED: release LT fully before Cartesian jog"
            )
        elif self.cartesian_rearm_required:
            if requested_mode == "idle":
                self.cartesian_rearm_required = False
            elif requested_mode == "cartesian_jog":
                self.last_stop_reason = "release_lt_after_physical_drag"
                requested_mode = "idle"
        if requested_mode == self.control_mode:
            return
        if self.control_mode == "physical_kinesthetic":
            self.get_logger().warn(
                f"RIGHT PASSIVE EXIT REQUEST: next={requested_mode} "
                f"operator_reason={self.operator.stop_reason} "
                f"bridge_reason={self.last_stop_reason} interlock={self.interlock_latched}"
            )
        if requested_mode == "cartesian_jog":
            self.reconcile(False)
            if self.hardware_passive or self.passive_enabled:
                self.last_stop_reason = "cartesian_control_blocked_while_passive"
                return
            self.resync_cartesian_command()
            self.control_mode = requested_mode
            self.get_logger().warn(
                "RIGHT CARTESIAN CONTROL ENABLED: LT commands sticks/D-pad"
            )
            return

        self.cartesian_commanded = None
        self.reconcile(requested_mode == "physical_kinesthetic")
        if requested_mode == "physical_kinesthetic" and not self.passive_enabled:
            return
        self.control_mode = requested_mode

    def resync_cartesian_command(self) -> None:
        """Discard an old jog target and seed the next command from encoders."""
        with self.state_lock:
            self.cartesian_commanded = (
                None
                if self.measured is None
                else np.asarray(self.measured, dtype=float).copy()
            )

    def cartesian_target_is_valid(self, target: np.ndarray) -> bool:
        if not self.state_validity_client.wait_for_service(timeout_sec=0.05):
            self.last_stop_reason = "state_validity_service_unavailable"
            return False
        with self.state_lock:
            state = dict(self.complete_joint_state)
        state.update(zip(self.arm_names, (float(value) for value in target)))
        request = GetStateValidity.Request()
        request.group_name = "right_arm"
        request.robot_state = RobotState()
        request.robot_state.is_diff = True
        request.robot_state.joint_state.name = list(state)
        request.robot_state.joint_state.position = list(state.values())
        try:
            response = self.state_validity_client.call(request, timeout_sec=0.10)
        except Exception as error:
            self.last_stop_reason = f"state_validity_call_failed:{error}"
            return False
        if response is None:
            self.last_stop_reason = "state_validity_call_timed_out"
            return False
        if not response.valid:
            self.last_stop_reason = "cartesian_target_collision_or_constraint"
            return False
        return True

    def service_cartesian_command(self, now: float) -> None:
        if self.automatic_home_handoff:
            return
        if not self.cartesian_lock.acquire(blocking=False):
            return
        try:
            self._service_cartesian_command_locked(now)
        finally:
            self.cartesian_lock.release()

    def _service_cartesian_command_locked(self, now: float) -> None:
        if self.automatic_home_handoff or self.control_mode != "cartesian_jog":
            return
        if now - self.last_cartesian_step_at < self.cartesian_period:
            return
        self.last_cartesian_step_at = now
        if (
            self.cartesian_twist is None
            or self.cartesian_twist_at is None
            or now - self.cartesian_twist_at > self.cartesian_watchdog
        ):
            self.last_stop_reason = "cartesian_command_watchdog_timeout"
            self.resync_cartesian_command()
            return
        if not self.joint_state_is_fresh(now):
            self.last_stop_reason = "joint_state_watchdog_timeout"
            self.cartesian_commanded = None
            return
        if self.hardware_passive or self.passive_enabled:
            self.last_stop_reason = "cartesian_control_blocked_while_passive"
            self.resync_cartesian_command()
            return
        twist = self.cartesian_twist.copy()
        if float(np.linalg.norm(twist)) <= 1.0e-6:
            # Centering the controls is a hold/reseed boundary.  This prevents a
            # rejected or lagging target from being replayed on the next motion.
            self.resync_cartesian_command()
            return
        with self.state_lock:
            measured = np.asarray(self.measured, dtype=float)
        if self.cartesian_commanded is None:
            self.cartesian_commanded = measured.copy()
        commanded = bounded_cartesian_command_seed(
            measured,
            self.cartesian_commanded,
        )
        qdot, condition = cartesian_joint_velocity(
            self.kinematics,
            commanded,
            twist,
            damping=self.cartesian_damping,
            maximum_joint_velocity=self.cartesian_max_joint_velocity,
        )
        target = directional_joint_limit_step(
            commanded,
            qdot,
            self.cartesian_period,
            self.kinematics.lower_limits,
            self.kinematics.upper_limits,
            self.cartesian_joint_margin,
        )
        target = np.clip(
            target,
            measured - CARTESIAN_MAX_COMMAND_LEAD_RAD,
            measured + CARTESIAN_MAX_COMMAND_LEAD_RAD,
        )
        achieved = self.kinematics.jacobian(commanded) @ (
            (target - commanded) / self.cartesian_period
        )
        requested_world = twist.copy()
        requested_world[3:] = (
            self.kinematics.forward(commanded).ee[:3, :3] @ twist[3:]
        )
        parallel_scale, orthogonal_ratio = cartesian_direction_fidelity(
            requested_world, achieved
        )
        # A global Jacobian condition number rejects even reachable X/Y motion
        # at motor zero because some *other* Cartesian axis is singular.  Judge
        # the requested direction itself instead.
        if (
            not np.all(np.isfinite(achieved))
            or parallel_scale < CARTESIAN_MINIMUM_PARALLEL_SCALE
            or orthogonal_ratio > CARTESIAN_MAXIMUM_ORTHOGONAL_RATIO
        ):
            condition_text = (
                "inf" if not math.isfinite(condition) else f"{condition:.3f}"
            )
            self.last_stop_reason = (
                "cartesian_requested_direction_unreliable:"
                f"parallel={parallel_scale:.3f}:orthogonal={orthogonal_ratio:.3f}:"
                f"condition={condition_text}"
            )
            self.resync_cartesian_command()
            return
        self.update_cartesian_diagnostic(twist)
        if (
            self.cartesian_state_validity_check
            and not self.cartesian_target_is_valid(target)
        ):
            self.resync_cartesian_command()
            return
        trajectory = JointTrajectory()
        trajectory.joint_names = list(self.arm_names)
        point = JointTrajectoryPoint()
        point.positions = [float(value) for value in target]
        # The deployed JTC requires the last point velocity to be exactly zero.
        # Continuity comes from bounded 20 Hz position increments.
        point.velocities = [0.0] * ARM_DOF
        # Finish before the next 20 Hz command.  The previous 60 ms points were
        # necessarily preempted every 50 ms and repeatedly restarted the JTC.
        duration_s = min(
            self.cartesian_trajectory_duration,
            0.8 * self.cartesian_period,
        )
        duration_ns = int(duration_s * 1.0e9)
        point.time_from_start.sec = duration_ns // 1_000_000_000
        point.time_from_start.nanosec = duration_ns % 1_000_000_000
        trajectory.points = [point]
        self.arm_pub.publish(trajectory)
        self.cartesian_commanded = target.copy()
        self.last_published = list(target)
        self.last_stop_reason = "none"

    def call_hardware(self, enabled: bool) -> tuple[bool, str]:
        if not self.hardware_client.wait_for_service(timeout_sec=1.0):
            return False, "hardware passive service unavailable"
        request = SetBool.Request()
        request.data = bool(enabled)
        try:
            # This method is also used during teardown, after executor.spin() has
            # returned.  A timeout is mandatory there: without an executor thread a
            # service response cannot be dispatched and call(None) waits forever.
            response = self.hardware_client.call(request, timeout_sec=1.0)
        except Exception as error:  # service failure must never report success
            return False, f"hardware service call failed: {error}"
        if response is None:
            return False, "hardware passive service timed out"
        return bool(response.success), str(response.message)

    def controller_state(self) -> tuple[str | None, str]:
        if not self.list_controllers_client.wait_for_service(timeout_sec=0.25):
            return None, "controller_manager_list_unavailable"
        try:
            response = self.list_controllers_client.call(
                ListControllers.Request(), timeout_sec=0.50
            )
        except Exception as error:
            return None, f"controller_manager_list_failed:{error}"
        if response is None:
            return None, "controller_manager_list_timed_out"
        expected_names = (str(self.get_parameter("arm_controller").value),)
        states = {
            str(controller.name): str(controller.state)
            for controller in response.controller
            if controller.name in expected_names
        }
        missing = [name for name in expected_names if name not in states]
        if missing:
            return None, "controllers_missing:" + ",".join(missing)
        if len(set(states.values())) != 1:
            return "mixed", "controller_states:" + ",".join(
                f"{name}={states[name]}" for name in expected_names
            )
        return states[expected_names[0]], "none"

    def switch_arm_controller(self, activate: bool) -> tuple[bool, str]:
        if not self.switch_controller_client.wait_for_service(timeout_sec=0.25):
            return False, "controller_manager_switch_unavailable"
        # J1-J7 become passive, while the gripper controller remains active so
        # LT+LB+L3 can operate J8 during physical teaching.
        controllers = [str(self.get_parameter("arm_controller").value)]
        request = SwitchController.Request()
        request.activate_controllers = controllers if activate else []
        request.deactivate_controllers = [] if activate else controllers
        request.strictness = 2  # STRICT
        request.activate_asap = True
        timeout_ns = int(self.controller_switch_timeout * 1.0e9)
        request.timeout.sec = timeout_ns // 1_000_000_000
        request.timeout.nanosec = timeout_ns % 1_000_000_000
        try:
            response = self.switch_controller_client.call(
                request, timeout_sec=self.controller_switch_timeout + 0.25
            )
        except Exception as error:
            return False, f"controller_switch_failed:{error}"
        if response is None:
            return False, "controller_switch_timed_out"
        if not response.ok:
            return False, f"controller_switch_rejected:{response.message}"
        expected = "active" if activate else "inactive"
        state, detail = self.controller_state()
        if state != expected:
            return False, f"controller_state_{state or 'missing'}_expected_{expected}:{detail}"
        return True, "none"

    def wait_hardware_state(self, expected: bool, timeout_s: float = 0.50) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.hardware_passive == expected:
                return True
            time.sleep(0.005)
        return self.hardware_passive == expected

    def measured_snapshot(self) -> np.ndarray | None:
        with self.state_lock:
            if self.measured is None:
                return None
            return np.asarray(self.measured, dtype=float)

    def keep_passive_fail_safe(self, state: PassiveState, reason: str) -> None:
        self.passive_state = state
        self.passive_enabled = True
        self.last_stop_reason = reason
        heartbeat = Bool()
        heartbeat.data = True
        self.heartbeat_pub.publish(heartbeat)
        self.get_logger().error(
            f"RIGHT remains PASSIVE (fail-safe): state={state.value} reason={reason}"
        )

    def reconcile(self, desired_passive: bool) -> None:
        if desired_passive and self.automatic_home_handoff:
            return
        if desired_passive and self.passive_enabled:
            return
        if not desired_passive and not self.passive_enabled and not self.hardware_passive:
            return
        if not self.transition_lock.acquire(blocking=False):
            return
        self.transition_target = desired_passive
        try:
            if desired_passive:
                self.passive_state = PassiveState.ENTERING_PASSIVE
                if not self.joint_state_is_fresh():
                    self.last_stop_reason = "fresh_joint_state_unavailable"
                    self.passive_state = PassiveState.RIGID
                    self.get_logger().error(
                        "RIGHT PASSIVE REJECTED: fresh_joint_state_unavailable"
                    )
                    return
                state, detail = self.controller_state()
                if state != "active":
                    if state == "inactive":
                        # A previous passive transition may have stopped JTC
                        # before the hardware acknowledgement was lost. Hold
                        # this state until the operator releases LT/LB; the
                        # normal measured-pose restore path then reactivates it.
                        self.keep_passive_fail_safe(
                            PassiveState.PASSIVE_CONTROLLER_RECOVERY_REQUIRED,
                            "arm_controller_inactive_before_passive_enable",
                        )
                        return
                    self.last_stop_reason = (
                        f"arm_controller_not_active:{state or 'missing'}:{detail}"
                    )
                    self.passive_state = PassiveState.RIGID
                    self.get_logger().error(
                        f"RIGHT PASSIVE REJECTED: {self.last_stop_reason}"
                    )
                    return
                success, detail = self.switch_arm_controller(False)
                if not success:
                    self.last_stop_reason = detail
                    self.passive_state = PassiveState.RIGID
                    self.get_logger().error(
                        f"RIGHT PASSIVE REJECTED: {detail}"
                    )
                    return
                success, detail = self.call_hardware(True)
                confirmed = success and self.wait_hardware_state(True)
                if not confirmed:
                    # Hardware never became passive: restore the controller;
                    # rigid hardware is still holding its prior target.
                    restored, restore_detail = self.switch_arm_controller(True)
                    if not restored:
                        self.keep_passive_fail_safe(
                            PassiveState.PASSIVE_CONTROLLER_RECOVERY_REQUIRED,
                            "passive_enable_failed_controller_restore_failed:"
                            + restore_detail,
                        )
                        return
                    self.passive_enabled = False
                    self.passive_state = PassiveState.RIGID
                    failure = (
                        detail if not success else "hardware_passive_enable_not_confirmed"
                    )
                    self.last_stop_reason = failure
                    self.get_logger().error(failure)
                    return
                self.passive_enabled = True
                self.passive_state = PassiveState.PASSIVE
                self.last_published = None
                self.get_logger().warn(
                    "RIGHT PASSIVE ENABLED: JTC inactive; human physical dragging allowed while LT+LB are held"
                )
            else:
                self.passive_state = PassiveState.EXITING_PASSIVE
                if not self.joint_state_is_fresh():
                    self.keep_passive_fail_safe(
                        PassiveState.PASSIVE_STATE_RECOVERY_REQUIRED,
                        "fresh_joint_state_unavailable",
                    )
                    return
                measured = self.measured_snapshot()
                recoverable, detail = measured_position_is_recoverable(measured)
                if not recoverable:
                    self.keep_passive_fail_safe(
                        PassiveState.PASSIVE_STATE_RECOVERY_REQUIRED, detail
                    )
                    return
                state, state_detail = self.controller_state()
                if state != "inactive":
                    self.keep_passive_fail_safe(
                        PassiveState.PASSIVE_CONTROLLER_RECOVERY_REQUIRED,
                        f"arm_controller_{state or 'missing'}:{state_detail}",
                    )
                    return
                success, detail = self.switch_arm_controller(True)
                if not success:
                    self.keep_passive_fail_safe(
                        PassiveState.PASSIVE_CONTROLLER_RECOVERY_REQUIRED, detail
                    )
                    return
                # Seed JTC from a fresh measured target while hardware kp is
                # still zero. OpenArmHW independently performs measured ->
                # command synchronization on its first rigid write cycle.
                self.last_published = None
                if not self.publish_arm_current():
                    self.switch_arm_controller(False)
                    self.keep_passive_fail_safe(
                        PassiveState.PASSIVE_STATE_RECOVERY_REQUIRED,
                        "measured_hold_publish_failed",
                    )
                    return
                time.sleep(max(0.0, self.rigid_sync_delay))
                if not self.joint_state_is_fresh():
                    self.switch_arm_controller(False)
                    self.keep_passive_fail_safe(
                        PassiveState.PASSIVE_STATE_RECOVERY_REQUIRED,
                        "joint_state_stale_during_rigid_sync",
                    )
                    return
                measured = self.measured_snapshot()
                self.last_published = None
                if not self.publish_arm_current():
                    self.switch_arm_controller(False)
                    self.keep_passive_fail_safe(
                        PassiveState.PASSIVE_STATE_RECOVERY_REQUIRED,
                        "final_measured_hold_publish_failed",
                    )
                    return
                command_error = float(
                    np.max(np.abs(np.asarray(self.last_published) - measured))
                )
                if command_error > self.rigid_target_tolerance:
                    self.switch_arm_controller(False)
                    self.keep_passive_fail_safe(
                        PassiveState.PASSIVE_STATE_RECOVERY_REQUIRED,
                        f"rigid_target_sync_error:{command_error:.6f}",
                    )
                    return
                success, detail = self.call_hardware(False)
                confirmed = success and self.wait_hardware_state(False)
                if not confirmed:
                    self.switch_arm_controller(False)
                    self.keep_passive_fail_safe(
                        PassiveState.PASSIVE_STATE_RECOVERY_REQUIRED,
                        "rigid_hardware_transition_failed:"
                        + (detail if not success else "state_not_confirmed"),
                    )
                    return
                heartbeat = Bool()
                heartbeat.data = False
                self.heartbeat_pub.publish(heartbeat)
                self.passive_enabled = False
                self.passive_state = PassiveState.RIGID_HOLD_CURRENT
                self.last_stop_reason = "none"
                self.get_logger().warn(
                    "RIGHT RIGID_HOLD_CURRENT: first target synchronized from fresh measured J1-J7"
                )
        finally:
            self.transition_target = None
            self.transition_lock.release()

    def on_timer(self) -> None:
        now = time.monotonic()
        if not self.automatic_home_handoff:
            self.service_gripper_command(now)
        decision = self.operator.decision()
        requested_mode = (
            "idle"
            if self.automatic_home_handoff or self.interlock_latched
            else decision.control_mode
        )
        self.apply_control_mode(requested_mode)
        if not self.automatic_home_handoff:
            self.service_cartesian_command(now)
        desired = requested_mode == "physical_kinesthetic"
        if self.passive_enabled or self.hardware_passive:
            heartbeat = Bool()
            heartbeat.data = True
            self.heartbeat_pub.publish(heartbeat)
            if not desired or not self.joint_state_is_fresh(now):
                if not self.joint_state_is_fresh(now):
                    self.last_stop_reason = "joint_state_watchdog_timeout"
                    self.interlock_latched = True
                if now - self.last_transition_attempt >= 0.25:
                    self.last_transition_attempt = now
                    self.reconcile(False)
        elif desired:
            self.reconcile(True)

        if now - self.last_status_publish >= 0.10:
            self.last_status_publish = now
            message = String()
            message.data = json.dumps(
                {
                    "command_fresh": decision.command_fresh,
                    "operator_armed": decision.armed,
                    "passive_enabled": self.passive_enabled,
                    "passive_state": self.passive_state.value,
                    "hardware_passive": self.hardware_passive,
                    "joint_state_fresh": self.joint_state_is_fresh(now),
                    "interlock_latched": self.interlock_latched,
                    "stop_reason": self.last_stop_reason,
                    "control_mode": self.control_mode,
                    "cartesian_rearm_required": self.cartesian_rearm_required,
                    "gripper_measured_rad": self.measured_gripper,
                    "gripper_target_rad": self.gripper_target,
                    "gripper_controller_subscribers": self.gripper_pub.get_subscription_count(),
                    "arm_motion_source": self.control_mode,
                    "automatic_home_handoff": self.automatic_home_handoff,
                    "cartesian_diagnostic": self.last_cartesian_diagnostic,
                },
                separators=(",", ":"),
                sort_keys=True,
            )
            self.status_pub.publish(message)

    def on_automatic_home_handoff(self, request, response):
        # Serialize only with a passive/rigid transition. Never hold a lock
        # shared with the 100 Hz timer or joystick callback across ROS calls.
        if not self.transition_lock.acquire(blocking=False):
            response.success = False
            response.message = "release LT/LB and wait for rigid transition to finish"
            return response
        try:
            operator = self.operator.decision()
            if (
                not operator.command_fresh
                or operator.control_mode != "idle"
                or operator.stop_reason is not None
                or self.passive_enabled
                or self.hardware_passive
                or self.interlock_latched
            ):
                response.success = False
                response.message = "release LT/LB and wait for fresh rigid idle with no interlock"
                return response
            state, detail = self.controller_state()
            if state != "active" or not self.joint_state_is_fresh():
                response.success = False
                response.message = (
                    "release LT/LB and wait for active controller and fresh joints: "
                    f"controller={state} detail={detail}"
                )
                return response
            if request.data:
                if self.control_mode != "idle":
                    response.success = False
                    response.message = "release LT/LB and wait for bridge idle"
                    return response
                self.automatic_home_handoff = True
                self.gripper_target = None
                self.cartesian_commanded = None
                response.message = "rigid idle bridge handed off for automatic HOME"
            else:
                self.automatic_home_handoff = False
                response.message = "manual control restored"
            response.success = True
            return response
        finally:
            self.transition_lock.release()

    def shutdown_rigid(self) -> None:
        self.reconcile(False)


def spin_once_with_worker_yield(executor) -> None:
    # On the deployed rclpy executor a tight submit loop can starve worker
    # callbacks under the GIL. Synthetic 20 Hz input then reaches the existing
    # 250 ms receive watchdog despite continuous publication. Yield after each
    # dispatch; do not relax watchdogs or change control/heartbeat frequencies.
    executor.spin_once(timeout_sec=0.1)
    time.sleep(0)


def main() -> int:
    rclpy.init()
    node = RightKinestheticBridge()
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    try:
        while rclpy.ok():
            spin_once_with_worker_yield(executor)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown_rigid()
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0
