"""Sole ROS command gateway. Never activates hardware or changes controllers.

Production requires DDS Security Enforce and the gateway enclave. Raw controller
endpoints must use the matching generated policy. The Unix socket is mode0600;
leases are bound to a live same-UID connection, with monotonic expiry. The trust
boundary is the configured experiment processes, not hostile root/same-UID code.
"""
import argparse
import base64
import copy
import fcntl
import json
import math
import os
from pathlib import Path
import secrets
import signal
import socket
import socketserver
import struct
import threading
import time

from .authority import LeaseAuthority
from .client import DEFAULT_SOCKET

CONTROLLERS = ('right_joint_trajectory_controller', 'right_gripper_controller',
               'left_joint_trajectory_controller', 'left_gripper_controller')
JOINTS = {
    CONTROLLERS[0]: [f'openarm_right_joint{i}' for i in range(1, 8)],
    CONTROLLERS[1]: ['openarm_right_finger_joint1'],
    CONTROLLERS[2]: [f'openarm_left_joint{i}' for i in range(1, 8)],
    CONTROLLERS[3]: ['openarm_left_finger_joint1'],
}
ACTIVE_CODES = (1, 2, 3)
CONTROLLER_QUERY_PERIOD_S = .25
CONTROLLER_QUERY_TIMEOUT_S = .75
CONTROLLER_LIST_MAX_AGE_S = 1.


def finite_positions(value, count):
    if (not isinstance(value, (list, tuple)) or len(value) != count
            or any(isinstance(v, bool) or not isinstance(v, (float, int))
                   or not math.isfinite(v) for v in value)):
        raise ValueError(f'expected {count} finite positions')
    return [float(v) for v in value]


def capability_frame(frame):
    parts = frame.split(':')
    if len(parts) != 4 or parts[0] != 'openarm-lease':
        raise ValueError('missing session capability')
    if len(parts[1]) != 64 or any(c not in '0123456789abcdef' for c in parts[1]):
        raise ValueError('invalid session capability')
    if len(parts[2]) != 32 or any(c not in '0123456789abcdef' for c in parts[2]):
        raise ValueError('invalid capability nonce')
    if len(parts[3]) > 1400: raise ValueError('original frame too large')
    try:
        original = base64.b64decode(parts[3], altchars=b'-_', validate=True).decode('utf-8')
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError('invalid original frame encoding') from exc
    if len(original.encode()) > 1024: raise ValueError('original frame too large')
    return parts[1], parts[2], original


class Gateway:
    def __init__(self, node, log):
        from action_msgs.msg import GoalStatusArray
        from control_msgs.action import FollowJointTrajectory
        from control_msgs.msg import JointTrajectoryControllerState
        from controller_manager_msgs.srv import ListControllers
        from rclpy.action import ActionClient, ActionServer
        from rclpy.callback_groups import ReentrantCallbackGroup
        from rclpy.qos import QoSProfile, DurabilityPolicy
        from trajectory_msgs.msg import JointTrajectory
        from sensor_msgs.msg import JointState
        self.node = node
        self.mutex = threading.RLock()
        self.authority = LeaseAuthority()
        self.log_file = Path(log).open('x', buffering=1)
        self.states = {}
        self.joint_state = None
        self.controllers = {}
        self.controller_command_interfaces = {}
        self.controllers_at = 0.
        self.list_future = None
        self.list_sent_at = None
        self.list_next_query_at = 0.
        self.list_service_ready = False
        self.list_requests = 0
        self.list_timeouts = 0
        self.list_failures = 0
        self.list_last_error = None
        self.list_last_rtt_s = None
        self.pending = set()
        self.handles = {}
        self.raw_active = {c: set() for c in CONTROLLERS}
        self.raw_status_seen = set()
        self.drain_at = None
        self.holds = {}
        self.required_holds = set()
        self.hold_pending = set()
        self.synchronized = False
        self.used_nonces = set()
        self.cancel_sent = set()
        self.last_dora = None
        self.last_dora_gripper = None
        self.command_scope = 'right_arm'
        self.last_dora_at = None
        self.last_publish_hold = 0.
        self.stationary_since = None
        self.stop_event = threading.Event()
        self.closing = False
        self.cb = ReentrantCallbackGroup()
        self.list_type = ListControllers
        self.list_client = node.create_client(ListControllers, '/controller_manager/list_controllers', callback_group=self.cb)
        self.publishers = {c: node.create_publisher(JointTrajectory, f'/{c}/joint_trajectory', 10) for c in CONTROLLERS}
        self.action_clients = {c: ActionClient(node, FollowJointTrajectory, f'/{c}/follow_joint_trajectory', callback_group=self.cb) for c in CONTROLLERS}
        self.subscriptions = []
        self.subscriptions.append(node.create_subscription(JointState, '/joint_states', self.receive_joints, 10, callback_group=self.cb))
        self.action_servers = []
        for c in CONTROLLERS:
            async def execute_callback(goal, controller=c):
                return await self.execute(controller, goal)
            self.subscriptions.append(node.create_subscription(JointTrajectoryControllerState,
                f'/{c}/controller_state', lambda msg, c=c: self.state(c, msg), 10, callback_group=self.cb))
            self.subscriptions.append(node.create_subscription(GoalStatusArray,
                f'/{c}/follow_joint_trajectory/_action/status', lambda msg, c=c: self.status(c, msg),
                QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL), callback_group=self.cb))
            self.action_servers.append(ActionServer(node, FollowJointTrajectory,
                f'/authorized/{c}/follow_joint_trajectory',
                execute_callback=execute_callback,
                goal_callback=lambda goal, c=c: self.accept(c, goal),
                cancel_callback=lambda goal, c=c: self.cancel(c, goal), callback_group=self.cb))
        self.timer = node.create_timer(.02, self.tick, callback_group=self.cb)
        self.log('gateway_started', motor_lifecycle_operations=False)

    def log(self, event, **fields):
        # Deliberately exclude capabilities/session secrets and raw UDS requests.
        self.log_file.write(json.dumps({'event': event, 'monotonic_ns': time.monotonic_ns(),
                                       'wall_ns': time.time_ns(), **fields}, allow_nan=False)+'\n')

    def state(self, controller, msg):
        with self.mutex:
            self.states[controller] = (msg, time.monotonic())

    def receive_joints(self, msg):
        with self.mutex:
            self.joint_state = (msg, time.monotonic())

    def measured_velocities(self, names):
        if self.joint_state is None: raise ValueError('no joint-state velocity feedback')
        msg, received = self.joint_state
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec/1e9
        if time.monotonic()-received > .2 or not -.02 <= time.time()-stamp <= .2:
            raise ValueError('stale joint-state velocity feedback')
        if len(msg.name) != len(msg.velocity) or len(set(msg.name)) != len(msg.name):
            raise ValueError('invalid joint-state velocity dimensions')
        values = dict(zip(msg.name, msg.velocity))
        return finite_positions([values.get(name, float('nan')) for name in names], len(names))

    def status(self, controller, msg):
        with self.mutex:
            self.raw_status_seen.add(controller)
            self.raw_active[controller] = {bytes(s.goal_info.goal_id.uuid).hex()
                                          for s in msg.status_list if s.status in ACTIVE_CODES}

    def reference(self, controller):
        if controller not in self.states:
            raise ValueError('no holding reference: '+controller)
        msg, received = self.states[controller]
        now = time.monotonic()
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec/1e9
        if now-received > .2 or not -.02 <= time.time()-stamp <= .2:
            raise ValueError('stale controller state: '+controller)
        if list(msg.joint_names) != JOINTS[controller]:
            raise ValueError('unexpected joint ordering: '+controller)
        return finite_positions(list(msg.output.positions), len(JOINTS[controller]))

    def readiness(self):
        if not self.synchronized:
            raise ValueError('manual synchronize-hold required before command admission')
        self.controller_readiness()
        if self.pending or self.hold_pending or any(self.raw_active.values()):
            raise ValueError('prior trajectory is still active')

    def controller_readiness(self):
        if not self.controller_query_status()['fresh']:
            raise ValueError('controller list unavailable/stale')
        for c in CONTROLLERS[:3]:
            if self.controllers.get(c) != 'active':
                raise ValueError('required controller inactive: '+c)
            self.reference(c)
        allowed = {*CONTROLLERS, 'joint_state_broadcaster'}
        if any(state == 'active' and c not in allowed for c, state in self.controllers.items()):
            raise ValueError('unexpected active controller')
        if any(name.startswith('spawner_') for name, _ in self.node.get_node_names_and_namespaces()):
            raise ValueError('trusted bootstrap spawner is still running')

    def publish(self, controller, positions, velocities=None):
        from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
        msg = JointTrajectory()
        msg.joint_names = JOINTS[controller]
        point = JointTrajectoryPoint()
        point.positions = positions
        if velocities is not None:
            point.velocities = finite_positions(velocities, len(positions))
        elif 'velocity' in self.controller_command_interfaces.get(controller, []):
            # Repeated drain/hold publications must never retain a moving velocity.
            point.velocities = [0.0]*len(positions)
        msg.points = [point]
        self.publishers[controller].publish(msg)

    def request(self, session, request):
        with self.mutex:
            now = time.monotonic()
            op = request.get('op')
            token = request.get('token')
            if self.closing and op != 'status':
                raise ValueError('gateway is shutting down; no new commands admitted')
            if op == 'status':
                return {'supported_dora_scopes':['right_arm','right_arm_gripper'],
                        'supports_dora_arm_velocity': True,
                        'status': self.authority.snapshot(now), 'pending': sorted(self.pending),
                        'controllers': self.controllers, 'raw_active': {k: sorted(v) for k, v in self.raw_active.items()},
                        'controller_query': self.controller_query_status(now),
                        'controller_command_interfaces': self.controller_command_interfaces,
                        'synchronized': self.synchronized, 'hold_pending': sorted(self.hold_pending),
                        'drain_ready': self.drain_ready()}
            if op == 'synchronize-hold':
                if self.authority.snapshot(now)['state'] not in ('idle', 'fault') or self.pending or self.hold_pending:
                    raise ValueError('cannot synchronize while an owner or unresolved dispatch exists')
                self.controller_readiness()
                # Explicit operator operation, never automatic startup movement.
                # Replaces any previously accepted timed goals with the exact
                # current controller OUTPUT and waits for terminal acknowledgments.
                self.drain_at = None
                self.synchronized = False
                self.authority.fault('manual_hold_synchronization')
                self.begin_drain(now, synchronization=True)
                return {'synchronizing': True}
            if op == 'acquire':
                self.readiness()
                scope = request.get('scope', 'right_arm')
                if scope not in ('right_arm', 'right_arm_gripper') or (request.get('owner') != 'dora' and scope != 'right_arm'):
                    raise ValueError('unsupported command scope for owner')
                right_reference = self.reference(CONTROLLERS[0])
                gripper_reference = self.reference(CONTROLLERS[1])
                token = self.authority.acquire(request.get('owner'), session, now)
                self.drain_at = None; self.holds = {}; self.cancel_sent = set()
                self.used_nonces = set()
                self.stationary_since = None
                self.command_scope = scope
                self.last_dora = right_reference
                self.last_dora_gripper = gripper_reference
                self.last_dora_at = None
                self.log('lease_acquired', owner=self.authority.owner, scope=scope)
                return {'token': token, 'ttl': self.authority.ttl,
                        'scope':scope, 'right_reference': list(self.last_dora)}
            if op == 'renew':
                self.authority.renew(token, session, now)
            elif op == 'command':
                self.authority.authorize(token, session, now, 'command')
                with_gripper = self.command_scope == 'right_arm_gripper'
                q = finite_positions(request.get('positions'), 8 if with_gripper else 7)
                interfaces = self.controller_command_interfaces.get(CONTROLLERS[0])
                if interfaces not in (['position'], ['position', 'velocity']):
                    raise ValueError('unknown/unsupported right arm command interfaces')
                velocity = None
                if interfaces == ['position', 'velocity']:
                    velocity = finite_positions(request.get('velocities'), 7)
                elif request.get('velocities') is not None:
                    raise ValueError('velocity supplied to a position-only controller')
                # The existing ACT Guard owns the numerical motion envelope.
                # This layer enforces source ownership without another filter,
                # clipping rule or scheduler-dependent motion threshold.
                self.reference(CONTROLLERS[0])
                if with_gripper:
                    self.reference(CONTROLLERS[1])
                # Both outputs are validated before either publication. DDS
                # delivery is not atomic; on failure hold the attempted pair
                # and latch the lease instead of admitting another producer.
                self.last_dora, self.last_dora_at = q[:7], now
                if with_gripper: self.last_dora_gripper = q[7:]
                try:
                    if velocity is None:
                        self.publish(CONTROLLERS[0], q[:7])
                    else:
                        self.publish(CONTROLLERS[0], q[:7], velocities=velocity)
                    if with_gripper: self.publish(CONTROLLERS[1], q[7:])
                except Exception as exc:
                    self.authority.fault('right command dispatch uncertain: '+str(exc))
                    self.begin_drain(now)
                    raise ValueError('right command dispatch uncertain; holding and fault latched') from exc
                self.log('dora_target', positions=q, velocities=velocity, scope=self.command_scope)
            elif op == 'release':
                self.authority.release(token, session, now)
                self.begin_drain(now)
            elif op == 'freeze':
                self.authority._check(token, session, now)
                self.authority.fault(str(request.get('reason', 'owner_freeze')))
                self.begin_drain(now)
            elif op == 'reset-fault':
                if not self.drain_ready():
                    raise ValueError('cannot reset before goals drained and hold verified')
                self.authority.reset_fault(now)
                self.log('fault_reset')
            else:
                raise ValueError('unsupported authority request')
            return {'status': self.authority.snapshot(time.monotonic())}

    def disconnect(self, session):
        with self.mutex:
            self.authority.disconnect(session, time.monotonic())

    def accept(self, controller, goal):
        from rclpy.action import GoalResponse
        with self.mutex:
            status = self.authority.snapshot(time.monotonic())
            if (self.closing or status['state'] != 'owned' or status['owner'] != 'h2'
                    or controller in self.pending or self.controllers.get(controller) != 'active'
                    or not self.controller_query_status()['fresh']):
                self.log('action_rejected', controller=controller, reason='owner/state/controller')
                return GoalResponse.REJECT
            try:
                token, nonce, original_frame = capability_frame(goal.trajectory.header.frame_id)
            except ValueError:
                return GoalResponse.REJECT
            if (not secrets.compare_digest(token, self.authority.token or '') or nonce in self.used_nonces):
                self.log('action_rejected', controller=controller, reason='missing/stale/replayed session capability')
                return GoalResponse.REJECT
            if list(goal.trajectory.joint_names) != JOINTS[controller] or not goal.trajectory.points:
                return GoalResponse.REJECT
            try:
                previous = -1.
                for point in goal.trajectory.points:
                    finite_positions(list(point.positions), len(JOINTS[controller]))
                    for values in (point.velocities, point.accelerations):
                        if values: finite_positions(list(values), len(JOINTS[controller]))
                    if point.effort: raise ValueError('effort not permitted for position controller')
                    t = point.time_from_start.sec + point.time_from_start.nanosec/1e9
                    if t <= previous or t < 0: raise ValueError('invalid action time')
                    previous = t
                for tolerance in [*goal.path_tolerance, *goal.goal_tolerance]:
                    if (tolerance.name not in JOINTS[controller]
                            or any(not math.isfinite(v) for v in (tolerance.position, tolerance.velocity, tolerance.acceleration))):
                        raise ValueError('invalid action tolerance')
                self.reference(controller)
            except ValueError:
                return GoalResponse.REJECT
            self.pending.add(controller)
            self.used_nonces.add(nonce)
            self.log('action_admitted', controller=controller, points=len(goal.trajectory.points))
            return GoalResponse.ACCEPT

    def cancel(self, controller, goal):
        from rclpy.action import CancelResponse
        with self.mutex:
            # Only the authenticated H2/MoveIt DDS identities can reach this
            # service. Cancellation never grants a new lease or disables torque.
            if controller not in self.pending: return CancelResponse.REJECT
            self.authority.fault('H2 action cancelled; owner revoked')
            # Preempt with an exact OUTPUT hold rather than raw CancelGoal,
            # whose JTC4.40.1 callback resets its reference to measured q.
            self.begin_drain(time.monotonic())
            return CancelResponse.ACCEPT

    async def execute(self, controller, outer):
        from control_msgs.action import FollowJointTrajectory
        result = FollowJointTrajectory.Result()
        raw = None
        dispatched = False
        terminal_known = False
        try:
            with self.mutex:
                status = self.authority.snapshot(time.monotonic())
                if status['state'] != 'owned' or status['owner'] != 'h2':
                    raise ValueError('authority lost before dispatch')
                client = self.action_clients[controller]
                if not client.server_is_ready(): raise ValueError('backend action unavailable')
                forwarded = copy.deepcopy(outer.request)
                forwarded.trajectory.header.frame_id = capability_frame(outer.request.trajectory.header.frame_id)[2]
                future = client.send_goal_async(forwarded, feedback_callback=lambda f: outer.publish_feedback(f.feedback))
                dispatched = True
            raw = await future
            if not raw.accepted:
                terminal_known = True
                raise ValueError('backend rejected trajectory')
            with self.mutex:
                self.handles[controller] = raw
                self.log('action_forwarded', controller=controller, goal_id=bytes(raw.goal_id.uuid).hex())
            result_future = raw.get_result_async()
            response = await result_future
            terminal_known = response.status in (4, 5, 6)
            result = response.result
            if response.status == 4: outer.succeed()
            elif response.status == 5 and outer.is_cancel_requested: outer.canceled()
            else: outer.abort()
            return result
        except Exception as exc:
            with self.mutex:
                self.authority.fault('action forwarding failed: '+str(exc))
                self.log('action_failed', controller=controller, reason=str(exc))
            result.error_code = -1
            result.error_string = str(exc)
            outer.abort()
            return result
        finally:
            with self.mutex:
                if terminal_known or not dispatched:
                    self.pending.discard(controller)
                    self.handles.pop(controller, None)
                else:
                    self.log('dispatch_quarantined', controller=controller)

    def begin_drain(self, now, synchronization=False):
        if self.drain_at is not None: return
        self.drain_at = now
        self.stationary_since = None
        self.holds = {}
        self.required_holds = set(CONTROLLERS[:3]) | {c for c in CONTROLLERS if self.controllers.get(c) == 'active'}
        try:
            self.controller_readiness()
            for c in self.required_holds:
                if not self.action_clients[c].server_is_ready():
                    raise ValueError('hold action server unavailable: '+c)
                self.holds[c] = self.drain_reference(c, synchronization)
        except ValueError as exc:
            self.authority.fault('cannot establish complete fresh hold: '+str(exc))
            self.holds = {}
            return
        self.log('draining', owner=self.authority.owner, reason=self.authority.reason,
                 references=self.holds)
        self.hold_pending = set(self.required_holds)
        from control_msgs.action import FollowJointTrajectory
        from trajectory_msgs.msg import JointTrajectoryPoint
        for c, q in self.holds.items():
            goal = FollowJointTrajectory.Goal()
            goal.trajectory.joint_names = JOINTS[c]
            point = JointTrajectoryPoint(); point.positions = q
            point.velocities = [0.0]*len(q)
            point.accelerations = [0.0]*len(q)
            goal.trajectory.points = [point]
            try:
                future = self.action_clients[c].send_goal_async(goal)
                future.add_done_callback(lambda future, c=c: self.hold_accepted(c, future, synchronization))
            except Exception as exc:
                self.authority.fault('hold dispatch uncertain: '+str(exc))

    def drain_reference(self, controller, synchronization=False):
        if not synchronization and self.authority.owner == 'dora':
            if controller == CONTROLLERS[0] and self.last_dora is not None:
                return list(self.last_dora)
            if (controller == CONTROLLERS[1] and self.command_scope == 'right_arm_gripper'
                    and self.last_dora_gripper is not None):
                return list(self.last_dora_gripper)
        return self.reference(controller)

    def hold_accepted(self, controller, future, synchronization):
        with self.mutex:
            try:
                handle = future.result()
                if not handle.accepted: raise ValueError('backend rejected hold')
                handle.get_result_async().add_done_callback(
                    lambda result: self.hold_completed(controller, result, synchronization))
            except Exception as exc:
                # Keep hold_pending latched: never mistake failed/uncertain
                # dispatch for successful completion or admit another source.
                self.authority.fault('hold acceptance failed: '+str(exc))

    def hold_completed(self, controller, future, synchronization):
        with self.mutex:
            try:
                result = future.result()
                if result.status != 4 or result.result.error_code != 0:
                    raise ValueError('holding goal did not succeed')
                self.hold_pending.discard(controller)
                self.log('hold_acknowledged', controller=controller, positions=self.holds[controller])
                if synchronization and not self.hold_pending:
                    self.synchronized = True
                    self.authority.reset_fault(time.monotonic())
                    # Admission remains blocked until all acknowledged holds
                    # have settled; no lease is issued by synchronization.
                    self.authority.state = 'draining'
                    self.authority.reason = 'manual_hold_synchronization'
            except Exception as exc:
                self.authority.fault('hold result failed: '+str(exc))

    def drain_ready(self):
        if self.drain_at is None or time.monotonic()-self.drain_at < .5:
            return False
        if (self.pending or self.hold_pending or any(self.raw_active.values()) or not self.required_holds
                or set(self.holds) != self.required_holds
                or not self.controller_query_status()['fresh']
                or any(self.controllers.get(c) != 'active' for c in self.required_holds)):
            self.stationary_since = None
            return False
        try:
            for c, q in self.holds.items():
                if max(abs(a-b) for a,b in zip(self.reference(c), q)) > 1e-8:
                    self.stationary_since = None
                    return False
                velocities = self.measured_velocities(JOINTS[c])
                if any(abs(v) > .05 for v in velocities):
                    self.stationary_since = None
                    return False
            if self.stationary_since is None: self.stationary_since = time.monotonic()
            return time.monotonic()-self.stationary_since >= .5
        except ValueError:
            self.stationary_since = None
            return False

    def controller_query_status(self, now=None):
        now = time.monotonic() if now is None else now
        age = now-self.controllers_at if self.controllers_at > 0 else None
        # The timestamp is conservatively the REQUEST time, never receipt of
        # an old response. Cached "active" must not masquerade as fresh state.
        return dict(fresh=age is not None and 0 <= age <= CONTROLLER_LIST_MAX_AGE_S,
                    age_s=age, max_age_s=CONTROLLER_LIST_MAX_AGE_S,
                    in_flight=self.list_future is not None,
                    pending_age_s=None if self.list_sent_at is None else now-self.list_sent_at,
                    service_ready=self.list_service_ready, requests=self.list_requests,
                    timeouts=self.list_timeouts, failures=self.list_failures,
                    response_timeout_s=CONTROLLER_QUERY_TIMEOUT_S,
                    last_rtt_s=self.list_last_rtt_s, last_error=self.list_last_error)

    def poll_controllers(self, now):
        """One bounded read-only RPC, no concurrent backlog or motion retry."""
        if self.list_future is not None:
            future = self.list_future
            elapsed = now-self.list_sent_at
            if elapsed >= CONTROLLER_QUERY_TIMEOUT_S or elapsed < 0:
                # Retire even an already-completed but overdue response.
                self.list_client.remove_pending_request(future)
                future.cancel()
                self.list_future = None
                self.list_sent_at = None
                self.list_timeouts += 1
                self.list_last_error = 'controller-manager response timed out'
                self.list_next_query_at = now+CONTROLLER_QUERY_PERIOD_S
                self.log('controller_query_timeout', elapsed_s=elapsed,
                         response_timeout_s=CONTROLLER_QUERY_TIMEOUT_S)
            elif future.done():
                try:
                    result = future.result()
                    if result is None:
                        raise ValueError('empty controller-manager response')
                    controllers = {c.name:c.state for c in result.controller}
                    interfaces = {c.name:sorted({name.rsplit('/',1)[-1] for name in c.claimed_interfaces})
                                  for c in result.controller}
                    self.controllers = controllers
                    self.controller_command_interfaces = interfaces
                    self.controllers_at = self.list_sent_at
                    self.list_last_rtt_s = elapsed
                    self.list_last_error = None
                except Exception as exc:
                    self.list_failures += 1
                    self.list_last_error = str(exc) or type(exc).__name__
                    self.log('controller_query_failed', reason=self.list_last_error)
                finally:
                    self.list_client.remove_pending_request(future)
                    self.list_future = None
                    self.list_sent_at = None
        if self.list_future is None and now >= self.list_next_query_at:
            self.list_next_query_at = now+CONTROLLER_QUERY_PERIOD_S
            self.list_service_ready = self.list_client.service_is_ready()
            if not self.list_service_ready:
                self.list_last_error = 'controller-manager service unavailable'
                return
            try:
                self.list_sent_at = now
                self.list_future = self.list_client.call_async(self.list_type.Request())
                self.list_requests += 1
            except Exception as exc:
                self.list_sent_at = None
                self.list_failures += 1
                self.list_last_error = str(exc) or type(exc).__name__
                self.log('controller_query_failed', reason=self.list_last_error)

    def tick(self):
        with self.mutex:
            now = time.monotonic()
            self.poll_controllers(now)
            status = self.authority.snapshot(now)
            if status['state'] in ('draining', 'fault'):
                self.begin_drain(now)
                if status['state'] == 'draining' and self.drain_ready():
                    self.authority.drain_completed(now)
                    self.log('drain_completed')


class Server(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--socket', default=DEFAULT_SOCKET)
    parser.add_argument('--log', required=True)
    parser.add_argument('--test-mode', action='store_true')
    args, rosargs = parser.parse_known_args()
    if not args.test_mode:
        if os.environ.get('ROS_SECURITY_ENABLE') != 'true' or os.environ.get('ROS_SECURITY_STRATEGY') != 'Enforce':
            parser.error('production gateway requires DDS security Enforce')
        if '/openarm/gateway' not in rosargs:
            parser.error('production gateway requires explicit gateway enclave')
        from .security import environment
        expected = environment(os.environ.get('ROS_SECURITY_KEYSTORE', ''), os.environ.get('ROS_DOMAIN_ID'))
        if any(os.environ.get(key) != value for key, value in expected.items()):
            parser.error('security environment differs from reviewed ownership policy')
    elif os.environ.get('ROS_DOMAIN_ID') in (None, '0', '97', '117'):
        parser.error('test mode must use an isolated non-hardware domain')
    path = Path(args.socket)
    if not path.parent.is_dir() or path.parent.stat().st_uid != os.getuid() or path.parent.stat().st_mode & 0o022:
        parser.error('Unix socket parent must be owned by current user and not group/world writable')
    lockpath = (str(path)+'.lock' if args.test_mode else
                f'/run/user/{os.getuid()}/openarm-command-domain{os.environ["ROS_DOMAIN_ID"]}.lock')
    lockfd = os.open(lockpath, os.O_CREAT|os.O_RDWR|os.O_CLOEXEC|os.O_NOFOLLOW, 0o600)
    fcntl.flock(lockfd, fcntl.LOCK_EX|fcntl.LOCK_NB)
    if path.exists():
        parser.error('existing gateway socket: inspect old owner; no automatic takeover')
    import rclpy
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.signals import SignalHandlerOptions
    rclpy.init(args=rosargs, signal_handler_options=SignalHandlerOptions.NO)
    node = rclpy.create_node('openarm_command_gateway')
    gateway = Gateway(node, args.log)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: gateway.stop_event.set())

    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            _, uid, _ = struct.unpack('3i', self.request.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
            if uid != os.getuid(): return
            session = secrets.token_hex(32)
            try:
                while True:
                    line = self.rfile.readline(65537)
                    if not line or len(line) > 65536: break
                    try:
                        request = json.loads(line)
                        if not isinstance(request, dict): raise ValueError('request must be object')
                        result = {'ok': True, **gateway.request(session, request)}
                    except (ValueError, TypeError, KeyError) as exc:
                        result = {'ok': False, 'error': str(exc)}
                    self.wfile.write((json.dumps(result, allow_nan=False)+'\n').encode())
                    self.wfile.flush()
            finally:
                gateway.disconnect(session)

    server = Server(str(path), Handler)
    os.chmod(path, 0o600)
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    executor = MultiThreadedExecutor(num_threads=4); executor.add_node(node)
    try:
        while rclpy.ok() and not gateway.stop_event.is_set():
            executor.spin_once(timeout_sec=.05)
    except KeyboardInterrupt:
        pass
    finally:
        # Graceful exit revokes the producer and acknowledges OUTPUT holds
        # before closing DDS. SIGKILL/process failure cannot run this cleanup:
        # already accepted timed H2 goals may continue in ROS. A replacement
        # gateway always requires explicit hold synchronization before admission.
        with gateway.mutex:
            gateway.closing = True
            needs_hold = gateway.authority.snapshot(time.monotonic())['state'] != 'idle' or bool(gateway.pending)
            if needs_hold:
                gateway.authority.fault('gateway_shutdown')
                gateway.begin_drain(time.monotonic())
        deadline = time.monotonic()+5.
        while needs_hold and rclpy.ok() and time.monotonic() < deadline:
            executor.spin_once(timeout_sec=.05)
            with gateway.mutex:
                if gateway.drain_ready(): break
        gateway.log('gateway_shutdown', hold_verified=(not needs_hold or gateway.drain_ready()),
                    motor_lifecycle_operations=False)
        server.shutdown(); server.server_close()
        executor.shutdown(timeout_sec=2.)
        node.destroy_node(); rclpy.try_shutdown()
        gateway.log_file.close()
        path.unlink(missing_ok=True)
        os.close(lockfd)


if __name__ == '__main__': main()
