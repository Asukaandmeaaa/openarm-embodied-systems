#!/usr/bin/env python3
"""Exercise ownership against fake ROS endpoints in a separate test domain.

No robot launch, hardware plugin, CAN socket, camera, or policy is used. This
tests gateway routing and lease loss, not DDS security or physical stopping.
"""
import argparse
import base64
import json
import os
from pathlib import Path
import signal
import secrets
import subprocess
import sys
import tempfile
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.control_ownership.client import Client

CONTROLLERS = {
    "right_joint_trajectory_controller": [f"openarm_right_joint{i}" for i in range(1, 8)],
    "right_gripper_controller": ["openarm_right_finger_joint1"],
    "left_joint_trajectory_controller": [f"openarm_left_joint{i}" for i in range(1, 8)],
    "left_gripper_controller": ["openarm_left_finger_joint1"],
}
RIGHT = "right_joint_trajectory_controller"


def eventually(predicate, timeout=5.0, detail="condition"):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(0.01)
    raise AssertionError(f"timed out waiting for {detail}")


def rejected(call):
    try:
        call()
    except ValueError:
        return
    raise AssertionError("request was accepted but should have been rejected")


class FakeControllers:
    def __init__(self):
        import rclpy
        from rclpy.action import ActionServer, CancelResponse, GoalResponse
        from rclpy.callback_groups import ReentrantCallbackGroup
        from rclpy.executors import MultiThreadedExecutor
        from rclpy.node import Node
        from rclpy.qos import DurabilityPolicy, QoSProfile
        from control_msgs.action import FollowJointTrajectory
        from control_msgs.msg import JointTrajectoryControllerState
        from controller_manager_msgs.srv import ListControllers
        from std_msgs.msg import Bool
        from sensor_msgs.msg import JointState
        from trajectory_msgs.msg import JointTrajectory

        self.rclpy = rclpy
        rclpy.init()
        self.node = Node("fake_ownership_controllers")
        self.lock = threading.RLock()
        self.output = {name: [0.0] * len(joints) for name, joints in CONTROLLERS.items()}
        self.output["right_gripper_controller"] = [-0.35]
        self.output["left_gripper_controller"] = [-0.019]
        self.topics = {name: [] for name in CONTROLLERS}
        self.actions = {name: [] for name in CONTROLLERS}
        self.cancels = {name: 0 for name in CONTROLLERS}
        self.preemptions = {name: 0 for name in CONTROLLERS}
        self.active = {}
        self.group = ReentrantCallbackGroup()
        self.servers = []
        self.subscriptions = []
        self.publishers = {}
        for name in CONTROLLERS:
            self.subscriptions.append(self.node.create_subscription(
                JointTrajectory, f"/{name}/joint_trajectory", self._topic(name), 10,
                callback_group=self.group))
            self.publishers[name] = self.node.create_publisher(
                JointTrajectoryControllerState, f"/{name}/controller_state", 10)
            self.servers.append(ActionServer(
                self.node, FollowJointTrajectory, f"/{name}/follow_joint_trajectory",
                execute_callback=self._execute(name),
                goal_callback=lambda _: GoalResponse.ACCEPT,
                cancel_callback=lambda _: CancelResponse.ACCEPT,
                callback_group=self.group))
        self.service = self.node.create_service(
            ListControllers, "/controller_manager/list_controllers", self._controllers,
            callback_group=self.group)
        self.passive = [self.node.create_publisher(
            Bool, f"/openarm_{side}_hardware/passive_state",
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
            for side in ("right", "left")]
        self.joint_state = self.node.create_publisher(JointState, "/joint_states", 10)
        self.timer = self.node.create_timer(0.01, self._publish_state)
        self.executor = MultiThreadedExecutor(num_threads=6)
        self.executor.add_node(self.node)
        self.thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.thread.start()

    def _controllers(self, request, response):
        from controller_manager_msgs.msg import ControllerState
        response.controller = [ControllerState(name=name, state="active") for name in CONTROLLERS]
        return response

    def _topic(self, name):
        def receive(message):
            with self.lock:
                self.topics[name].append(message)
                if message.points:
                    self.output[name] = list(message.points[-1].positions)
        return receive

    def _execute(self, name):
        def execute(handle):
            from control_msgs.action import FollowJointTrajectory
            from rclpy.impl.implementation_singleton import rclpy_implementation as _rclpy
            trajectory = handle.request.trajectory
            with self.lock:
                previous = self.active.get(name)
                if previous is not None and previous.is_active:
                    # Test-only emulation of JTC replacing an active goal. The
                    # new goal preserves controller output until its own point
                    # is applied; preemption never seeds measured position.
                    previous._update_state(_rclpy.GoalEvent.CANCEL_GOAL)
                    self.preemptions[name] += 1
                self.active[name] = handle
                self.actions[name].append(trajectory)
            duration = max(0.12, trajectory.points[-1].time_from_start.sec +
                           trajectory.points[-1].time_from_start.nanosec / 1e9)
            until = time.monotonic() + duration
            while time.monotonic() < until:
                if handle.is_cancel_requested:
                    with self.lock:
                        self.cancels[name] += 1
                    handle.canceled()
                    return FollowJointTrajectory.Result(error_code=0)
                time.sleep(0.01)
            with self.lock:
                self.output[name] = list(trajectory.points[-1].positions)
            handle.succeed()
            return FollowJointTrajectory.Result(error_code=0)
        return execute

    def _publish_state(self):
        from control_msgs.msg import JointTrajectoryControllerState
        from sensor_msgs.msg import JointState
        from std_msgs.msg import Bool
        for name, publisher in self.publishers.items():
            message = JointTrajectoryControllerState()
            message.header.stamp = self.node.get_clock().now().to_msg()
            message.joint_names = CONTROLLERS[name]
            with self.lock:
                positions = list(self.output[name])
            message.output.positions = positions
            message.reference.positions = positions
            message.feedback.positions = positions
            # Real validated JTCs export position state only. Velocities are
            # available in /joint_states, not controller_state.feedback.
            message.feedback.velocities = []
            publisher.publish(message)
        state = JointState()
        state.header.stamp = self.node.get_clock().now().to_msg()
        with self.lock:
            for name, joints in CONTROLLERS.items():
                state.name.extend(joints)
                state.position.extend(self.output[name])
                state.velocity.extend([0.0] * len(joints))
        self.joint_state.publish(state)
        for publisher in self.passive:
            publisher.publish(Bool(data=False))

    def counts(self):
        with self.lock:
            return {name: len(items) for name, items in self.topics.items()}

    def close(self):
        self.executor.shutdown(timeout_sec=2)
        self.thread.join(timeout=2)
        for server in self.servers:
            server.destroy()
        self.node.destroy_node()
        self.rclpy.try_shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", type=int, default=163)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 150 <= args.domain <= 232:
        parser.error("choose a dedicated test domain between 150 and 232; hardware domain 97 is forbidden")
    if os.environ.get("ROS_SECURITY_ENABLE", "").lower() == "true":
        parser.error("this fake routing test needs its own explicitly insecure test environment")
    os.environ["ROS_DOMAIN_ID"] = str(args.domain)
    os.environ["ROS_AUTOMATIC_DISCOVERY_RANGE"] = "LOCALHOST"
    os.environ.pop("ROS_LOCALHOST_ONLY", None)
    os.environ["ROS_SECURITY_ENABLE"] = "false"
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / "gateway_fake_report.json"
    if report_path.exists():
        parser.error("refusing to overwrite previous report")

    from control_msgs.action import FollowJointTrajectory
    from rclpy.action import ActionClient
    from trajectory_msgs.msg import JointTrajectoryPoint

    fake = FakeControllers()
    process = None
    checks = []
    clients = []
    log_stream = (args.output / "gateway_fake_terminal.log").open("x")
    report = {"physical_hardware": False, "CAN_sockets": False,
              "DDS_security_test": False, "ROS_DOMAIN_ID": args.domain, "checks": checks}
    try:
        with tempfile.TemporaryDirectory(prefix="openarm-gateway-test-") as temporary:
            preexisting = ActionClient(fake.node, FollowJointTrajectory,
                                       f"/{RIGHT}/follow_joint_trajectory")
            assert preexisting.wait_for_server(timeout_sec=3)
            previous_goal = FollowJointTrajectory.Goal()
            previous_goal.trajectory.joint_names = CONTROLLERS[RIGHT]
            previous_goal.trajectory.points = [JointTrajectoryPoint(positions=[0.0] * 7)]
            previous_goal.trajectory.points[0].time_from_start.sec = 10
            previous_future = preexisting.send_goal_async(previous_goal)
            eventually(previous_future.done, detail="preexisting raw goal")
            previous_handle = previous_future.result()
            assert previous_handle.accepted
            socket_path = str(Path(temporary) / "gateway.sock")
            process = subprocess.Popen([
                sys.executable, "-m", "scripts.control_ownership.gateway",
                "--test-mode", "--socket", socket_path,
                "--log", str(args.output / "gateway_fake.jsonl")],
                cwd=ROOT, env=os.environ.copy(), stdout=log_stream, stderr=subprocess.STDOUT)
            eventually(lambda: Path(socket_path).exists() or process.poll() is not None,
                       detail="gateway socket")
            assert process.poll() is None, "gateway exited; inspect gateway_fake_terminal.log"
            # Allow source-age/controller readiness checks to receive samples.
            observer = Client(socket_path); clients.append(observer)
            eventually(lambda: observer.request("status")["status"]["state"] == "idle",
                       detail="initial idle state")
            dora = Client(socket_path); clients.append(dora)
            eventually(lambda: observer.request("status")["raw_active"][RIGHT],
                       detail="latched active goal discovery")
            rejected(lambda: dora.acquire("dora"))
            cancel = previous_handle.cancel_goal_async()
            eventually(cancel.done, detail="test-owned preexisting goal cancellation")
            previous_result = previous_handle.get_result_async()
            eventually(previous_result.done, detail="preexisting goal drains")
            eventually(lambda: not observer.request("status")["raw_active"][RIGHT],
                       detail="raw idle status")
            preexisting.destroy()
            checks.append("preexisting raw action discovered and blocks new source admission")
            rejected(lambda: dora.acquire("dora"))
            def synchronize_when_ready():
                try:
                    return observer.request("synchronize-hold")
                except ValueError:
                    return False
            eventually(synchronize_when_ready, detail="explicit hold synchronization accepted")
            eventually(lambda: observer.request("status")["synchronized"] and
                       observer.request("status")["status"]["state"] == "idle",
                       detail="explicit hold synchronization acknowledged")
            checks.append("fresh gateway requires explicit acknowledged hold synchronization")
            def acquire_when_ready(client, owner):
                try:
                    return client.acquire(owner)
                except ValueError:
                    return False
            eventually(lambda: acquire_when_ready(dora, "dora"), detail="settled source admission")
            second = Client(socket_path); clients.append(second)
            rejected(lambda: second.acquire("h2"))
            rejected(lambda: second.acquire("dora"))
            checks.append("second owner and duplicate Dora owner rejected")

            initial = fake.counts()
            dora.command([0.0005] * 7)
            eventually(lambda: fake.counts()[RIGHT] > initial[RIGHT], detail="authorized right command")
            assert fake.output[RIGHT] == [0.0005] * 7
            assert all(fake.counts()[name] == initial[name] for name in CONTROLLERS if name != RIGHT)
            checks.append("authorized Dora command reaches right7 only")

            dora.renew()
            baseline = fake.counts()
            for values in ([0.0] * 6, [0.0] * 8, [float("nan")] * 7):
                rejected(lambda values=values: dora.command(values))
            second.token = dora.token
            rejected(lambda: second.command([0.02] * 7))
            second.token = None
            time.sleep(0.06)
            assert fake.counts() == baseline
            checks.append("invalid dimensions/nonfinite and copied token cause no raw target write")

            action = ActionClient(fake.node, FollowJointTrajectory,
                                  f"/authorized/{RIGHT}/follow_joint_trajectory")
            assert action.wait_for_server(timeout_sec=5)
            def send_goal(value=0.02, duration=0.12, capability=None, nonce=None, frame=""):
                goal = FollowJointTrajectory.Goal()
                goal.trajectory.joint_names = CONTROLLERS[RIGHT]
                if capability is not None:
                    frame64 = base64.urlsafe_b64encode(frame.encode()).decode()
                    goal.trajectory.header.frame_id = (
                        f"openarm-lease:{capability}:{nonce or secrets.token_hex(16)}:{frame64}")
                point = JointTrajectoryPoint(positions=[value] * 7)
                point.time_from_start.sec = int(duration)
                point.time_from_start.nanosec = int((duration - int(duration)) * 1e9)
                goal.trajectory.points = [point]
                future = action.send_goal_async(goal)
                eventually(future.done, detail="proxy goal response")
                return future.result()

            dora.renew()
            raw_actions = len(fake.actions[RIGHT])
            assert not send_goal().accepted
            assert len(fake.actions[RIGHT]) == raw_actions
            checks.append("H2 action rejected before raw action server while Dora owns")

            old_token = dora.token
            dora.release()
            eventually(lambda: observer.request("status")["status"]["state"] == "idle",
                       detail="release drain completes")
            h2 = Client(socket_path); clients.append(h2)
            h2.acquire("h2")
            raw_actions = len(fake.actions[RIGHT])
            dora.token = old_token
            rejected(lambda: dora.command([0.03] * 7))
            dora.token = None
            rejected(lambda: h2.command([0.03] * 7))
            assert not send_goal().accepted
            assert not send_goal(capability=old_token).accepted
            used_nonce = secrets.token_hex(16)
            handle = send_goal(capability=h2.token, nonce=used_nonce, frame="world")
            assert handle.accepted
            h2.renew()
            result = handle.get_result_async()
            eventually(result.done, detail="authorized H2 result")
            assert result.result().result.error_code == 0
            assert len(fake.actions[RIGHT]) == raw_actions + 1
            assert fake.actions[RIGHT][-1].header.frame_id == "world"
            assert list(fake.actions[RIGHT][-1].points[0].positions) == [0.02] * 7
            assert not send_goal(capability=h2.token, nonce=used_nonce).accepted
            checks.append("certified H2 handoff forwards action; old token and H2 direct command rejected")
            checks.append("H2 capability required, replay rejected, original frame and numeric targets restored")

            h2.renew()
            long_handle = send_goal(duration=2.0, capability=h2.token)
            assert long_handle.accepted
            assert not send_goal(capability=h2.token).accepted
            eventually(lambda: len(fake.actions[RIGHT]) == raw_actions + 2,
                       detail="long H2 raw goal")
            preemptions_before = fake.preemptions[RIGHT]
            h2.close(); clients.remove(h2)
            long_result = long_handle.get_result_async()
            eventually(long_result.done, detail="gateway cancellation after H2 disconnect")
            assert fake.preemptions[RIGHT] == preemptions_before + 1
            eventually(lambda: observer.request("status")["status"]["state"] == "idle",
                       detail="H2 disconnect cancels and drains")
            assert fake.output[RIGHT] == [0.02] * 7
            checks.append("second H2 goal rejected and disconnect preempts to exact output before handoff")
            orphan = Client(socket_path); clients.append(orphan)
            orphan.acquire("dora")
            orphan.command([0.0205] * 7)
            eventually(lambda: fake.output[RIGHT] == [0.0205] * 7, detail="orphan command")
            orphan.close(); clients.remove(orphan)
            eventually(lambda: observer.request("status")["status"]["state"] == "idle",
                       detail="disconnection drains")
            assert fake.output[RIGHT] == [0.0205] * 7
            checks.append("Dora disconnect holds last target and drains without a lifecycle operation")
            expiring = Client(socket_path); clients.append(expiring)
            expiring.acquire("dora")
            expiring.command([0.021] * 7)
            eventually(lambda: fake.output[RIGHT] == [0.021] * 7, detail="expiring source target")
            eventually(lambda: observer.request("status")["status"]["state"] == "idle",
                       detail="unrenewed lease expires and drains")
            rejected(lambda: expiring.command([0.0215] * 7))
            assert fake.output[RIGHT] == [0.021] * 7
            checks.append("lease expiry revokes commands and holds last accepted target")
            before_cli = fake.counts()
            with fake.lock:
                held_before_cli = {name: list(q) for name, q in fake.output.items()}
            hold_report = args.output / "manual_hold_cli_report.json"
            with (args.output / "manual_hold_cli_terminal.log").open("x") as cli_log:
                cli = subprocess.run([
                    sys.executable, "-m", "scripts.control_ownership.client",
                    "--socket", socket_path, "hold-test", "--report", str(hold_report)],
                    cwd=ROOT, env=os.environ.copy(), stdout=cli_log,
                    stderr=subprocess.STDOUT, timeout=15, check=False)
            assert cli.returncode == 0, "manual hold CLI failed; inspect its terminal log"
            cli_result = json.loads(hold_report.read_text())
            assert cli_result["passed"] is True and cli_result["ACT_executed"] is False
            assert set(cli_result["competing_sources_rejected"]) == {"dora", "h2"}
            assert cli_result["right_reference"] == held_before_cli[RIGHT]
            assert fake.counts()[RIGHT] - before_cli[RIGHT] == 10
            assert all(fake.counts()[name] == before_cli[name] for name in CONTROLLERS if name != RIGHT)
            with fake.lock:
                assert fake.output == held_before_cli
            assert observer.request("status")["status"]["state"] == "idle"
            checks.append("manual hold CLI sends ten unchanged right targets, rejects competitors, and returns idle without ACT")
            shutting = Client(socket_path); clients.append(shutting)
            shutting.acquire("dora")
            shutting.command([0.0215] * 7)
            eventually(lambda: fake.output[RIGHT] == [0.0215] * 7,
                       detail="last accepted target before gateway shutdown")
            with fake.lock:
                held_before_shutdown = {name: list(q) for name, q in fake.output.items()}
            process.send_signal(signal.SIGTERM)
            eventually(lambda: observer.request("status")["status"]["reason"] == "gateway_shutdown",
                       detail="gateway enters graceful shutdown")
            try:
                second.acquire("h2")
            except ValueError as error:
                assert "shutting down" in str(error)
            else:
                raise AssertionError("late source admitted during gateway shutdown")
            rejected(lambda: shutting.command([0.022] * 7))
            assert process.wait(timeout=8) == 0
            records = [json.loads(line) for line in (args.output / "gateway_fake.jsonl").read_text().splitlines()]
            shutdown = [record for record in records if record["event"] == "gateway_shutdown"]
            assert len(shutdown) == 1 and shutdown[0]["hold_verified"] is True
            assert shutdown[0]["motor_lifecycle_operations"] is False
            with fake.lock:
                assert fake.output == held_before_shutdown
            checks.append("SIGTERM rejects late commands, acknowledges exact output holds, and exits cleanly without motor lifecycle operations")
            action.destroy()
            report["passed"] = True
    except BaseException as error:
        report["passed"] = False
        report["error"] = repr(error)
        raise
    finally:
        for client in clients:
            try: client.close()
            except Exception: pass
        if process is not None and process.poll() is None:
            process.send_signal(signal.SIGINT)
            try: process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait(timeout=5)
        fake.close()
        log_stream.close()
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
