#!/usr/bin/env python3
"""Real DDS permission tests with dummy endpoints, never robot/CAN endpoints."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def worker(role, output):
    import rclpy
    from rclpy.action import ActionClient
    from rclpy.node import Node
    from action_msgs.srv import CancelGoal
    from control_msgs.action import FollowJointTrajectory
    from controller_manager_msgs.srv import ListControllers, SwitchController, SetHardwareComponentState
    from std_srvs.srv import SetBool
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
    from std_msgs.msg import Bool, Float64, Float64MultiArray, Int32
    names = {"backend": "dummy_ownership_backend", "gateway": "openarm_command_gateway",
             "h2": "home_guarded_executor", "observer": "openarm_ownership_observer",
             "moveit": "move_group", "insecure": "insecure_dummy_writer"}
    rclpy.init()
    node = Node(names[role])
    raw = "/right_joint_trajectory_controller/joint_trajectory"
    report = {"role": role, "checks": {}}
    handles = []
    try:
        if role == "backend":
            messages = []
            for side in ("right", "left"):
                for kind, topic in ((Int32, f"/{side}_gripper/contact_state"),
                                    (Float64, f"/{side}_gripper/motor_torque"),
                                    (Bool, f"/openarm_{side}_hardware/passive_state")):
                    handles.append(node.create_publisher(kind, topic, 10))
                    report["checks"][topic + "_publisher_allowed"] = True
            handles.append(node.create_subscription(JointTrajectory, raw,
                           lambda message: messages.append(list(message.points[0].positions)), 10))
            (output / "backend.ready").write_text("ready")
            until = time.monotonic() + 45
            while time.monotonic() < until and not (output / "backend.stop").exists():
                rclpy.spin_once(node, timeout_sec=0.05)
            report["messages"] = messages
        elif role in ("gateway", "insecure"):
            publisher = node.create_publisher(JointTrajectory, raw, 10)
            message = JointTrajectory()
            message.joint_names = [f"openarm_right_joint{i}" for i in range(1, 8)]
            message.points = [JointTrajectoryPoint(positions=[1.0 if role == "gateway" else 2.0] * 7)]
            until = time.monotonic() + 2
            while time.monotonic() < until:
                publisher.publish(message)
                rclpy.spin_once(node, timeout_sec=0.05)
            report["checks"]["publisher_created"] = True
            if role == "gateway":
                handles.append(ActionClient(node, FollowJointTrajectory,
                               "/right_joint_trajectory_controller/follow_joint_trajectory"))
                report["checks"]["raw_action_client_created"] = True
        else:
            checks = {
                "right_arm_raw_topic_denied": lambda: node.create_publisher(JointTrajectory, raw, 10),
                "left_arm_raw_topic_denied": lambda: node.create_publisher(JointTrajectory,
                    "/left_joint_trajectory_controller/joint_trajectory", 10),
                "right_gripper_raw_topic_denied": lambda: node.create_publisher(JointTrajectory,
                    "/right_gripper_controller/joint_trajectory", 10),
                "left_gripper_raw_topic_denied": lambda: node.create_publisher(JointTrajectory,
                    "/left_gripper_controller/joint_trajectory", 10),
                "raw_action_denied": lambda: ActionClient(node, FollowJointTrajectory,
                    "/right_joint_trajectory_controller/follow_joint_trajectory"),
                "left_raw_action_denied": lambda: ActionClient(node, FollowJointTrajectory,
                    "/left_joint_trajectory_controller/follow_joint_trajectory"),
                "right_gripper_raw_action_denied": lambda: ActionClient(node, FollowJointTrajectory,
                    "/right_gripper_controller/follow_joint_trajectory"),
                "left_gripper_raw_action_denied": lambda: ActionClient(node, FollowJointTrajectory,
                    "/left_gripper_controller/follow_joint_trajectory"),
                "raw_cancel_denied": lambda: node.create_client(CancelGoal,
                    "/right_joint_trajectory_controller/follow_joint_trajectory/_action/cancel_goal"),
                "passive_service_denied": lambda: node.create_client(SetBool,
                    "/openarm_right_hardware/set_passive"),
                "passive_heartbeat_denied": lambda: node.create_publisher(Bool,
                    "/openarm_right_hardware/passive_heartbeat", 10),
                "switch_controller_denied": lambda: node.create_client(SwitchController,
                    "/controller_manager/switch_controller"),
                "hardware_lifecycle_denied": lambda: node.create_client(SetHardwareComponentState,
                    "/controller_manager/set_hardware_component_state"),
                "forward_controller_topic_denied": lambda: node.create_publisher(Float64MultiArray,
                    "/right_forward_position_controller/commands", 10),
                "velocity_controller_topic_denied": lambda: node.create_publisher(Float64MultiArray,
                    "/right_forward_velocity_controller/commands", 10),
            }
            for label, operation in checks.items():
                try:
                    handles.append(operation())
                    report["checks"][label] = False
                except Exception:
                    report["checks"][label] = True
            handles.append(node.create_client(ListControllers, "/controller_manager/list_controllers"))
            report["checks"]["readonly_list_client_allowed"] = True
            if role == "moveit":
                handles.append(ActionClient(node, FollowJointTrajectory,
                               "/authorized/right_joint_trajectory_controller/follow_joint_trajectory"))
                report["checks"]["authorized_action_client_allowed"] = True
            if role == "h2":
                handles.append(ActionClient(node, FollowJointTrajectory,
                               "/authorized/right_gripper_controller/follow_joint_trajectory"))
                report["checks"]["authorized_gripper_action_client_allowed"] = True
    finally:
        for handle in handles:
            if isinstance(handle, ActionClient):
                handle.destroy()
        node.destroy_node()
        rclpy.try_shutdown()
        (output / f"{role}.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", type=int, default=161)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", choices=["backend", "gateway", "h2", "observer", "moveit", "insecure"])
    args = parser.parse_args()
    if args.worker:
        worker(args.worker, args.output)
        return
    if not 150 <= args.domain <= 232:
        parser.error("only isolated software test domains150..232 are permitted")
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / "dds_security_report.json"
    if report_path.exists():
        parser.error("refusing to overwrite previous results")
    from scripts.control_ownership.security import provision, environment
    with tempfile.TemporaryDirectory(prefix="openarm-security-test-") as temporary:
        keystore = Path(temporary) / "keys"
        provision(keystore, args.domain)
        base_env = os.environ.copy()
        base_env.update(environment(keystore, args.domain))
        base_env.pop("ROS_LOCALHOST_ONLY", None)
        def run(role):
            env = base_env.copy()
            env["ROS_SECURITY_ENCLAVE_OVERRIDE"] = "/openarm/" + role
            if role == "insecure":
                env["ROS_SECURITY_ENABLE"] = "false"
                env.pop("ROS_SECURITY_ENCLAVE_OVERRIDE", None)
            log = (args.output / f"{role}.log").open("x")
            child = subprocess.Popen([sys.executable, __file__, "--output", str(args.output),
                                      "--worker", role], cwd=ROOT, env=env,
                                     stdout=log, stderr=subprocess.STDOUT)
            return child, log
        backend, backend_log = run("backend")
        report = {"physical_hardware": False, "CAN_sockets": False,
                  "ROS_DOMAIN_ID": args.domain, "DDS_security_test": True, "checks": {},
                  "policy_sha256": hashlib.sha256((ROOT / "scripts/control_ownership/policies/experiment.xml").read_bytes()).hexdigest()}
        try:
            deadline = time.monotonic() + 8
            while not (args.output / "backend.ready").exists() and time.monotonic() < deadline:
                if backend.poll() is not None:
                    raise AssertionError("secure dummy backend failed; inspect backend.log")
                time.sleep(0.05)
            assert (args.output / "backend.ready").exists()
            for role in ("gateway", "insecure", "h2", "moveit", "observer"):
                process, log = run(role)
                try:
                    assert process.wait(timeout=10) == 0, f"{role} worker failed; inspect {role}.log"
                finally:
                    if process.poll() is None:
                        process.kill(); process.wait(timeout=3)
                    log.close()
                result = json.loads((args.output / f"{role}.json").read_text())
                assert all(result["checks"].values()), result
                report["checks"][role] = result["checks"]
            (args.output / "backend.stop").write_text("stop")
            assert backend.wait(timeout=5) == 0
            backend_result = json.loads((args.output / "backend.json").read_text())
            messages = backend_result["messages"]
            report["checks"]["backend"] = backend_result["checks"]
            assert [1.0] * 7 in messages, "allowed gateway publication did not reach backend"
            assert [2.0] * 7 not in messages, "insecure publication reached secure backend"
            report["checks"]["secure_gateway_delivery"] = True
            report["checks"]["insecure_delivery_denied"] = True
            report["passed"] = True
        except BaseException as error:
            report["passed"] = False
            report["error"] = repr(error)
            raise
        finally:
            if backend.poll() is None:
                backend.terminate()
                try: backend.wait(timeout=3)
                except subprocess.TimeoutExpired: backend.kill(); backend.wait(timeout=3)
            backend_log.close()
            report_path.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
