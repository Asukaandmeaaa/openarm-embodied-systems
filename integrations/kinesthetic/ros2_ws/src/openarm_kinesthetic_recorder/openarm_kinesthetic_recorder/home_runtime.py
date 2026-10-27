"""Minimal MoveIt runtime used by the commissioned two-stage HOME executor."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import time

from controller_manager_msgs.srv import ListControllers
from moveit_msgs.action import ExecuteTrajectory
from moveit_msgs.msg import (
    Constraints,
    DisplayTrajectory,
    JointConstraint,
    MoveItErrorCodes,
    PlanningSceneComponents,
)
from moveit_msgs.srv import GetMotionPlan, GetPlanningScene, GetPositionFK
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from rosidl_runtime_py.convert import message_to_ordereddict
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool


MAX_REAL_EXECUTION_SCALING = 0.8


def validate_real_execution_scaling(
    velocity_scaling: float, acceleration_scaling: float
) -> tuple[float, float]:
    values = (float(velocity_scaling), float(acceleration_scaling))
    if any(
        not math.isfinite(value)
        or not 0.0 < value <= MAX_REAL_EXECUTION_SCALING
        for value in values
    ):
        raise ValueError(
            "real execution scaling factors must be > 0 and "
            f"<= {MAX_REAL_EXECUTION_SCALING}"
        )
    return values


def trajectory_sha256(trajectory) -> str:
    payload = message_to_ordereddict(trajectory)
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class HomeRuntimeNode(Node):
    """Fresh-state planning and guarded execution with no legacy search/profile API."""

    def __init__(self, node_name: str = "home_guarded_executor") -> None:
        super().__init__(node_name)
        self.expected_names = [f"openarm_right_joint{index}" for index in range(1, 8)]
        self.current_positions: list[float] | None = None
        self.current_velocities: list[float] | None = None
        self.current_efforts: list[float] | None = None
        self.current_state_monotonic_s: float | None = None
        self.passive_state: bool | None = None
        self.passive_state_monotonic_s: float | None = None
        self.passive_subscription = None
        self.execution_feedback_state = ""
        self.execution_abort_detail: dict | None = None
        self.execution_goal_sent = False

        self.scene_client = self.create_client(GetPlanningScene, "/get_planning_scene")
        self.fk_client = self.create_client(GetPositionFK, "/compute_fk")
        self.plan_client = self.create_client(GetMotionPlan, "/plan_kinematic_path")
        self.controller_client = self.create_client(
            ListControllers, "/controller_manager/list_controllers"
        )
        self.execute_client = ActionClient(
            self, ExecuteTrajectory, "/execute_trajectory"
        )
        self.create_subscription(
            JointState, "/joint_states", self._joint_state, qos_profile_sensor_data
        )
        self.display_publisher = self.create_publisher(
            DisplayTrajectory,
            "/display_planned_path",
            QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            ),
        )
        self.passive_state_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._renew_passive_state_subscription()

    def _joint_state(self, message: JointState) -> None:
        indices = {name: index for index, name in enumerate(message.name)}
        if any(name not in indices for name in self.expected_names):
            return
        positions = [
            float(message.position[indices[name]]) for name in self.expected_names
        ]
        if not all(math.isfinite(value) for value in positions):
            return
        self.current_positions = positions
        self.current_velocities = self._ordered_optional_joint_field(
            message.name, message.velocity
        )
        self.current_efforts = self._ordered_optional_joint_field(
            message.name, message.effort
        )
        self.current_state_monotonic_s = time.monotonic()

    def _ordered_optional_joint_field(self, names, values) -> list[float] | None:
        if len(values) != len(names):
            return None
        indices = {name: index for index, name in enumerate(names)}
        if any(name not in indices for name in self.expected_names):
            return None
        ordered = [float(values[indices[name]]) for name in self.expected_names]
        return ordered if all(math.isfinite(value) for value in ordered) else None

    def _passive_state(self, message: Bool) -> None:
        self.passive_state = bool(message.data)
        self.passive_state_monotonic_s = time.monotonic()

    def _renew_passive_state_subscription(self) -> None:
        if self.passive_subscription is not None:
            self.destroy_subscription(self.passive_subscription)
        self.passive_state = None
        self.passive_state_monotonic_s = None
        self.passive_subscription = self.create_subscription(
            Bool,
            "/openarm_right_hardware/passive_state",
            self._passive_state,
            self.passive_state_qos,
        )

    def refresh_rigid_state(self, timeout_s: float) -> list[float]:
        self.current_positions = None
        self.current_velocities = None
        self.current_efforts = None
        self.current_state_monotonic_s = None
        self._renew_passive_state_subscription()
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline and (
            self.current_positions is None or self.passive_state is None
        ):
            rclpy.spin_once(self, timeout_sec=0.1)
        if self.current_positions is None or self.current_state_monotonic_s is None:
            raise RuntimeError("fresh /joint_states unavailable")
        if time.monotonic() - self.current_state_monotonic_s > 0.5:
            raise RuntimeError("/joint_states became stale")
        if self.passive_state is True:
            raise RuntimeError("passive_state=true")
        return list(self.current_positions)

    @staticmethod
    def wait_for_service(client, name: str, timeout_s: float) -> None:
        if not client.wait_for_service(timeout_sec=timeout_s):
            raise RuntimeError(f"required MoveIt service unavailable: {name}")

    def guard_snapshot(
        self, object_id: str, required_dimensions: list[float], timeout_s: float
    ) -> dict:
        self.wait_for_service(self.scene_client, "/get_planning_scene", timeout_s)
        request = GetPlanningScene.Request()
        request.components.components = PlanningSceneComponents.WORLD_OBJECT_GEOMETRY
        future = self.scene_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout_s)
        response = future.result()
        if response is None:
            raise RuntimeError("timed out reading MoveIt PlanningScene")
        objects = {item.id: item for item in response.scene.world.collision_objects}
        if object_id not in objects:
            raise RuntimeError(
                f"required collision guard {object_id!r} is absent; "
                f"scene has {list(objects)}"
            )
        collision = objects[object_id]
        if (
            not collision.primitives
            or not collision.primitive_poses
            or len(collision.primitives[0].dimensions) != 3
        ):
            raise RuntimeError("table collision object has no valid box primitive")
        dimensions = [float(value) for value in collision.primitives[0].dimensions]
        for actual, required in zip(dimensions, required_dimensions):
            if actual + 1.0e-6 < float(required) or actual > float(required) + 0.10:
                raise RuntimeError(
                    f"collision dimensions {dimensions} do not cover "
                    f"{required_dimensions}"
                )

        def pose_values(pose) -> dict[str, list[float]]:
            return {
                "position_m": [pose.position.x, pose.position.y, pose.position.z],
                "orientation_xyzw": [
                    pose.orientation.x,
                    pose.orientation.y,
                    pose.orientation.z,
                    pose.orientation.w,
                ],
            }

        snapshot = {
            "id": collision.id,
            "frame_id": collision.header.frame_id,
            "dimensions_m": dimensions,
            "object_pose": pose_values(collision.pose),
            "primitive_pose": pose_values(collision.primitive_poses[0]),
        }
        if not all(
            math.isfinite(float(value))
            for values in (
                snapshot["dimensions_m"],
                snapshot["object_pose"]["position_m"],
                snapshot["object_pose"]["orientation_xyzw"],
                snapshot["primitive_pose"]["position_m"],
                snapshot["primitive_pose"]["orientation_xyzw"],
            )
            for value in values
        ):
            raise RuntimeError("table collision snapshot contains non-finite values")
        return snapshot

    def plan(
        self,
        payload: dict,
        goal: list[float],
        timeout_s: float,
        velocity_scaling: float,
        acceleration_scaling: float,
    ):
        validate_real_execution_scaling(velocity_scaling, acceleration_scaling)
        self.wait_for_service(self.plan_client, "/plan_kinematic_path", timeout_s)
        request = GetMotionPlan.Request()
        motion = request.motion_plan_request
        motion.group_name = str(payload.get("move_group", "right_arm"))
        motion.pipeline_id = "ompl"
        motion.num_planning_attempts = 10
        motion.allowed_planning_time = 8.0
        motion.max_velocity_scaling_factor = float(velocity_scaling)
        motion.max_acceleration_scaling_factor = float(acceleration_scaling)
        motion.start_state.joint_state.name = list(payload["joint_names"])
        motion.start_state.joint_state.position = [
            float(value) for value in payload["start_positions_rad"]
        ]
        motion.start_state.is_diff = True
        constraints = Constraints()
        constraints.name = "fixed_lifted_home"
        for name, value in zip(payload["joint_names"], goal):
            constraint = JointConstraint()
            constraint.joint_name = str(name)
            constraint.position = float(value)
            constraint.tolerance_above = 0.005
            constraint.tolerance_below = 0.005
            constraint.weight = 1.0
            constraints.joint_constraints.append(constraint)
        motion.goal_constraints = [constraints]
        future = self.plan_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout_s + 10.0)
        response = future.result()
        if response is None:
            raise RuntimeError("MoveIt planning request timed out")
        return response.motion_plan_response

    def fk_pose(self, payload: dict, positions: list[float], timeout_s: float):
        self.wait_for_service(self.fk_client, "/compute_fk", timeout_s)
        request = GetPositionFK.Request()
        request.header.frame_id = str(payload.get("world_frame", "world"))
        request.fk_link_names = [
            str(payload.get("end_effector_link", "openarm_right_ee_base_link"))
        ]
        request.robot_state.joint_state.name = list(payload["joint_names"])
        request.robot_state.joint_state.position = [float(value) for value in positions]
        request.robot_state.is_diff = True
        future = self.fk_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout_s)
        response = future.result()
        if response is None:
            raise RuntimeError("MoveIt FK request timed out")
        if response.error_code.val != MoveItErrorCodes.SUCCESS or not response.pose_stamped:
            raise RuntimeError(f"MoveIt FK failed with error {response.error_code.val}")
        return response.pose_stamped[0]

    def publish_display(self, response) -> None:
        message = DisplayTrajectory()
        message.model_id = "openarm"
        message.trajectory_start = deepcopy(response.trajectory_start)
        message.trajectory = [deepcopy(response.trajectory)]
        self.display_publisher.publish(message)

    def verify_controller_active(self, name: str, timeout_s: float) -> dict:
        self.wait_for_service(
            self.controller_client, "/controller_manager/list_controllers", timeout_s
        )
        future = self.controller_client.call_async(ListControllers.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout_s)
        response = future.result()
        if response is None:
            raise RuntimeError("controller-manager query timed out")
        controllers = {item.name: item for item in response.controller}
        if name not in controllers:
            raise RuntimeError(
                f"required controller {name!r} is absent; found {list(controllers)}"
            )
        controller = controllers[name]
        if controller.state != "active":
            raise RuntimeError(
                f"required controller {name!r} is {controller.state!r}, not active"
            )
        return {
            "name": controller.name,
            "state": controller.state,
            "type": controller.type,
            "claimed_interfaces": list(controller.claimed_interfaces),
        }

    def execution_table_guard_failure(self, now: float | None = None) -> str:
        return ""

    def _execution_feedback(self, message) -> None:
        self.execution_feedback_state = str(message.feedback.state)

    def execute_same_trajectory(
        self,
        trajectory,
        expected_duration_s: float,
        timeout_margin_s: float,
    ) -> dict:
        self.execution_goal_sent = False
        if not self.execute_client.wait_for_server(timeout_sec=10.0):
            raise RuntimeError("MoveIt /execute_trajectory action is unavailable")
        goal = ExecuteTrajectory.Goal()
        goal.trajectory = trajectory
        goal.controller_names = []
        self.execution_feedback_state = ""
        self.execution_abort_detail = None
        send_future = self.execute_client.send_goal_async(
            goal, feedback_callback=self._execution_feedback
        )
        rclpy.spin_until_future_complete(self, send_future, timeout_sec=10.0)
        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            raise RuntimeError("MoveIt rejected the guarded execution goal")
        self.execution_goal_sent = True
        result_future = goal_handle.get_result_async()
        deadline = time.monotonic() + max(
            expected_duration_s * 3.0, expected_duration_s + timeout_margin_s
        )
        abort_reason = ""
        while not result_future.done() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.passive_state is not False:
                abort_reason = "right arm left rigid mode during execution"
                break
            if (
                self.current_state_monotonic_s is None
                or time.monotonic() - self.current_state_monotonic_s > 0.5
            ):
                abort_reason = "joint-state feedback became stale during execution"
                break
            guard_failure = self.execution_table_guard_failure()
            if guard_failure:
                abort_reason = guard_failure
                break
        if abort_reason or not result_future.done():
            cancel_future = goal_handle.cancel_goal_async()
            rclpy.spin_until_future_complete(self, cancel_future, timeout_sec=3.0)
            if not abort_reason:
                abort_reason = "MoveIt execution timed out"
            self.execution_abort_detail = {
                "reason": abort_reason,
                "moveit_feedback_state": self.execution_feedback_state,
            }
            raise RuntimeError(abort_reason)
        wrapper = result_future.result()
        if wrapper is None:
            raise RuntimeError("MoveIt execution returned no result")
        result = wrapper.result
        return {
            "action_status": int(wrapper.status),
            "moveit_error_code": int(result.error_code.val),
            "moveit_message": result.error_code.message,
            "moveit_source": result.error_code.source,
            "execution_state": self.execution_feedback_state,
        }
