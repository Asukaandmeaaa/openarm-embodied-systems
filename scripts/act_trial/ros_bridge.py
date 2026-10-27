"""Attended right ACT experiments via already-running ROS.

This process never enables/disables motors, launches HOME, or commands the left
arm. The task profile also controls the right gripper. Before START it only
observes. ROS remains holding after it exits.
Right references are bounded before authorized gateway publication;
passive CAN records are the separate evidence of actual hardware commands.
"""
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time
import xml.etree.ElementTree as ET

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.act_trial.guard import Guard
from scripts.act_trial.episode_completion import MechanicalZeroCompletion
from scripts.act_trial.preflight import DEFAULT_BUNDLE, inspect_start
from scripts.act_trial.telemetry import Telemetry
from scripts.act_trial.status_writer import StatusWriter
from scripts.act_trial.speed_contract import check_measured_speed
from scripts.act_trial.profiles import profile_from_config, collision_checks_enabled
from scripts.act_trial.motion_generator import from_config as motion_from_config, check_live_host
from scripts.control_ownership.client import Client as AuthorityClient
from core import NAMES, preserve_nonfinite, write_json

ARM_TOPIC = '/right_joint_trajectory_controller/joint_trajectory'
ARM_CONTROLLER = 'right_joint_trajectory_controller'
GRIP_CONTROLLER = 'right_gripper_controller'


def check_dispatch_freshness(tick_started, checked_at, feedback_age, observation_ns,
                             now, *, watchdog=.10, feedback_limit=.10, action_limit=.50):
    """Recheck the ORIGINAL evidence after expensive work, before sending.

    Freshly reading a different pose here would not validate the already-checked
    collision sweep. Never dispatch a stale proposal then notice next tick.
    """
    elapsed = now-tick_started
    if not np.isfinite(elapsed) or elapsed < 0 or elapsed > watchdog:
        raise ValueError(f'pre-dispatch control tick deadline exceeded: {elapsed:.6f}s')
    if not np.isfinite(checked_at) or not tick_started <= checked_at <= now:
        raise ValueError('invalid pre-dispatch check timestamp')
    age = feedback_age+elapsed  # Conservative: samples were read after tick entry.
    if not np.isfinite(age) or age < 0 or age > feedback_limit:
        raise ValueError(f'pre-dispatch checked feedback stale: {age:.6f}s')
    if observation_ns is None:
        raise ValueError('pre-dispatch observation timestamp missing')
    action_age = now-int(observation_ns)/1e9
    if not np.isfinite(action_age) or not 0 <= action_age <= action_limit:
        raise ValueError(f'pre-dispatch observation stale: {action_age:.6f}s')


class TickTiming:
    """Small per-phase wall/CPU measurements; no I/O until the tick finishes."""
    def __init__(self):
        self.wall = time.monotonic()
        self.cpu = time.thread_time()
        self.phases = {}

    def mark(self, phase):
        wall, cpu = time.monotonic(), time.thread_time()
        self.phases[phase] = {'wall_s': wall-self.wall, 'thread_cpu_s': cpu-self.cpu}
        self.wall, self.cpu = wall, cpu


def source_age(stamp, wall_ns, received_mono, now):
    source_ns = int(stamp.sec)*10**9 + int(stamp.nanosec)
    lag = (wall_ns-source_ns)/1e9
    if source_ns <= 0 or lag < -.02:
        raise ValueError('invalid ROS hardware clock or future source timestamp')
    return max(0., lag) + now-received_mono


def ordered(names, values, expected):
    if len(names) != len(values) or len(set(names)) != len(names):
        raise ValueError('invalid feedback joint names/dimensions')
    mapping = dict(zip(names, values))
    result = np.array([mapping[name] for name in expected], dtype=float)
    if not np.all(np.isfinite(result)):
        raise ValueError('nonfinite joint feedback/reference')
    return result


def trajectory_for(q7):
    """JTC4.40.1 zero-time position point: no second interpolation or ramp.

    Its final-point path returns this target at the next controller update.
    Position-only retains ROS PD/gravity and holds the target on publisher loss.
    """
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
    q = np.asarray(q7, dtype=float)
    if q.shape != (7,) or not np.all(np.isfinite(q)):
        raise ValueError('dispatch requires seven finite right joint positions')
    message = JointTrajectory()
    message.joint_names = NAMES[:7]
    point = JointTrajectoryPoint()
    point.positions = q.tolist()
    message.points = [point]  # Default stamp/duration zero; no derivative fields.
    return message


def check_reference_history(output8, commands8, history, fixed_gripper):
    """Catch competing action goals as well as topic publishers.

    Controller output is unquantized; MIT position encoder truncates to16bits,
    so passive CAN comparison permits one documented quantization step only.
    Per-axis CAN comparisons allow frames straddling a100Hz update boundary.
    """
    candidates = np.asarray(history, dtype=float)
    output = np.asarray(output8, dtype=float)
    commands = np.asarray(commands8, dtype=float)
    if candidates.ndim != 2 or candidates.shape[1] != 8 or not len(candidates):
        raise ValueError('missing dispatch reference history')
    if fixed_gripper is not None and abs(output[7]-fixed_gripper) > 1e-8:
        raise ValueError('existing gripper controller reference changed')
    if np.min(np.max(np.abs(candidates[:,:7]-output[:7]), axis=1)) > 1e-8:
        raise ValueError('ROS output not in this trial reference history')
    if fixed_gripper is None and np.min(np.abs(candidates[:,7]-output[7])) > 1e-8:
        raise ValueError('ROS gripper output not in this trial reference history')
    encoded = np.floor((candidates+12.5)/25*65535)*25/65535-12.5
    if np.any(np.min(np.abs(encoded-commands), axis=0) > 1e-8):
        raise ValueError('CAN command not in this trial reference history')


def input_node_options(start_parameter_services):
    options = {'start_parameter_services': start_parameter_services}
    if not start_parameter_services:
        # Jazzy creates this independent service even with parameter services
        # disabled. The H2 enclave has no reply permission for this observer's
        # services; disable the unused diagnostic service, not DDS enforcement.
        from rclpy.parameter import Parameter
        options['parameter_overrides'] = [Parameter(
            'start_type_description_service', Parameter.Type.BOOL, False)]
    return options


class Inputs:
    def __init__(self, *, start_parameter_services=True):
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data, QoSProfile, DurabilityPolicy
        from sensor_msgs.msg import JointState
        from std_msgs.msg import String
        from control_msgs.msg import JointTrajectoryControllerState
        from controller_manager_msgs.srv import ListControllers
        self.rclpy = rclpy
        rclpy.init()
        try:
            self.node = Node('right_act_attended_trial', **input_node_options(start_parameter_services))
        except BaseException:
            rclpy.try_shutdown()
            raise
        self.lock = threading.Lock()
        self.messages = {}
        for topic, kind in [('/joint_states', JointState),
                            (f'/{ARM_CONTROLLER}/controller_state', JointTrajectoryControllerState),
                            (f'/{GRIP_CONTROLLER}/controller_state', JointTrajectoryControllerState)]:
            self.node.create_subscription(kind, topic, self.callback(topic), qos_profile_sensor_data)
        self.node.create_subscription(String, '/robot_description', self.callback('/robot_description'),
                                      QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.list_type = ListControllers
        self.list_client = self.node.create_client(ListControllers, '/controller_manager/list_controllers')
        self.controller_future = None
        self.controllers_at = None
        self.controllers = None
        self.controller_error = None
        # Audit/model construction can block the Dora main loop for >1 second.
        # Poll on the already-running ROS executor, not on the command loop.
        self.controller_timer = self.node.create_timer(.1, self.poll_controllers)
        from rclpy.executors import SingleThreadedExecutor
        self.executor = SingleThreadedExecutor(context=self.node.context)
        self.executor.add_node(self.node)
        self.thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.thread.start()

    def callback(self, topic):
        def receive(message):
            with self.lock:
                self.messages[topic] = (message, time.monotonic(), time.time_ns())
        return receive

    def get(self, topic):
        with self.lock:
            if topic not in self.messages:
                raise ValueError('waiting for '+topic)
            return self.messages[topic]

    def poll_controllers(self):
        now = time.monotonic()
        with self.lock:
            if (self.controller_future is not None or not self.list_client.service_is_ready()
                    or (self.controllers_at is not None and now-self.controllers_at <= .25)):
                return
            try:
                future = self.list_client.call_async(self.list_type.Request())
            except Exception as exc:
                self.controllers_at = None
                self.controller_error = str(exc)
                return
            self.controller_future = future
        # Register outside the lock: an already-completed future can call back
        # immediately. Timestamp the response callback, never the next Dora tick.
        future.add_done_callback(self.controllers_received)

    def controllers_received(self, future):
        with self.lock:
            if future is not self.controller_future:
                return
            self.controller_future = None
            try:
                result = future.result()
                self.controllers = {item.name: item.state for item in result.controller}
                self.controllers_at = time.monotonic()
                self.controller_error = None
            except Exception as exc:
                self.controllers_at = None
                self.controller_error = str(exc)

    def query_controllers(self, now):
        with self.lock:
            received = self.controllers_at
            controllers = dict(self.controllers or {})
        # `now` was captured before snapshot; a concurrent callback can be newer.
        if received is None or max(now, time.monotonic())-received > 1.0:
            raise ValueError('controller activation status unavailable or stale')
        for name in (ARM_CONTROLLER, GRIP_CONTROLLER, 'left_joint_trajectory_controller'):
            if controllers.get(name) != 'active':
                raise ValueError(name+' is not active')

    def parameters(self, node_name, names):
        from rcl_interfaces.srv import GetParameters
        from rclpy.parameter import parameter_value_to_python
        client = self.node.create_client(GetParameters, '/'+node_name+'/get_parameters')
        if not client.wait_for_service(timeout_sec=3):
            raise ValueError('missing ROS parameter service '+node_name)
        req = GetParameters.Request(); req.names = names
        future = client.call_async(req)
        deadline = time.monotonic()+3
        while not future.done() and time.monotonic() < deadline:
            time.sleep(.01)
        if not future.done():
            raise ValueError('ROS parameter timeout '+node_name)
        return dict(zip(names, map(parameter_value_to_python, future.result().values)))

    def snapshot(self, now):
        self.query_controllers(now)
        js, received, wall = self.get('/joint_states')
        age = source_age(js.header.stamp, wall, received, now)
        ages = {'/joint_states': age}
        actual = ordered(js.name, js.position, NAMES)
        references = []; outputs = []; states = {}
        for controller, expected in ((ARM_CONTROLLER, NAMES[:7]), (GRIP_CONTROLLER, NAMES[7:8])):
            msg, received, wall = self.get('/'+controller+'/controller_state')
            ages[controller] = source_age(msg.header.stamp, wall, received, now)
            age = max(age, ages[controller])
            reference = ordered(msg.joint_names, msg.reference.positions, expected)
            output = ordered(msg.joint_names, msg.output.positions, expected)
            feedback = ordered(msg.joint_names, msg.feedback.positions, expected)
            references.extend(reference); outputs.extend(output)
            states[controller] = {'reference': reference.tolist(), 'output': output.tolist(),
                                  'feedback': feedback.tolist()}
        if age > .10:
            raise ValueError(f'ROS state age {age:.4f}s exceeds .10s; topic_ages_s={ages}')
        return {'actual': actual, 'reference': np.array(references), 'output': np.array(outputs),
                'age': age, 'topic_ages_s': ages, 'states': states, 'controllers': self.controllers}

    def audit(self, bundle, profile='commissioning'):
        description, _, _ = self.get('/robot_description')
        saved = Path(bundle['capture_directory'])/'snapshots/robot_description.urdf'
        current_hash = hashlib.sha256(description.data.encode()).hexdigest()
        if profile == 'visual32':
            from scripts.act_trial.closed_contract import verify_live_model
            verify_live_model(description.data)
        elif current_hash != hashlib.sha256(saved.read_bytes()).hexdigest():
            raise ValueError('live robot description differs from validated ROS H2 capture')
        params = {}
        for controller in (ARM_CONTROLLER, GRIP_CONTROLLER):
            names = ['joints', 'command_interfaces', 'state_interfaces',
                                             'interpolation_method', 'interpolate_from_desired_state',
                                             'use_sim_time']
            if profile == 'visual32' and controller == ARM_CONTROLLER:
                names += ['allow_nonzero_velocity_at_trajectory_end', 'cmd_timeout', 'constraints.goal_time']
            p = self.parameters(controller, names)
            expected = NAMES[:7] if controller == ARM_CONTROLLER else NAMES[7:8]
            interfaces = ['position', 'velocity'] if profile == 'visual32' and controller == ARM_CONTROLLER else ['position']
            if (p['joints'] != expected or p['command_interfaces'] != interfaces
                    or 'position' not in p['state_interfaces'] or p['use_sim_time'] is not False
                    or p['interpolation_method'] not in ('splines', 'none')):
                raise ValueError('unsupported live ROS controller configuration')
            if profile == 'visual32' and controller == ARM_CONTROLLER:
                if (p['allow_nonzero_velocity_at_trajectory_end'] is not True
                        or p['cmd_timeout'] != .1 or p['constraints.goal_time'] != 0.):
                    raise ValueError('ACT streaming q/qd requires nonzero endpoint support and 0.1s command timeout')
            params[controller] = p
        params['controller_manager'] = self.parameters('controller_manager', ['update_rate', 'use_sim_time'])
        if params['controller_manager'] != {'update_rate': 100, 'use_sim_time': False}:
            raise ValueError('controller manager differs from validated 100Hz hardware execution')
        hw = ET.fromstring(description.data).find("ros2_control[@name='openarm_right_hardware_interface']/hardware")
        hw_params = {p.attrib['name']: p.text for p in hw.findall('param')}
        gains = {name: [float(hw_params[name+str(i)]) for i in range(1, 8)]
                       + [float(hw_params[name+'_hand'])] for name in ('kp', 'kd')}
        return {'live_URDF_sha256': current_hash, 'live_controller_parameters': params,
                'expected_right_gains': gains,
                'gravity_and_hardware_config': ('pinned closed-transit collection URDF' if profile == 'visual32'
                                                else 'exact match to sealed successful ROS H2 URDF'),
                'trajectory_contract': 'single position-only point at zero time, seven right arm joints',
                'gripper_commanded_by_bridge': False, 'left_commanded_by_bridge': False}

    def close(self):
        self.executor.shutdown(timeout_sec=3)
        self.thread.join(timeout=3)
        self.node.destroy_node()
        self.rclpy.try_shutdown()


def action_ready_for_start(now, received_at, observation_age, *, rolling=False):
    """Admit a validated chunk by its original lifetime, not inference cadence.

    Legacy shaped targets arrive at control rate. Rolling chunks arrive only
    at inference rate, so their 100 ms receipt gate would flicker between
    healthy predictions. Never extend a chunk's original observation lifetime.
    """
    if received_at is None or observation_age is None:
        return False
    receipt_age = now-received_at
    if not np.isfinite([receipt_age, observation_age]).all():
        return False
    if rolling:
        from scripts.act_trial.rolling_actions import EXPIRY_NS
        lifetime = EXPIRY_NS/1e9
        return bool(0 <= receipt_age < lifetime and 0 <= observation_age < lifetime)
    return bool(0 <= receipt_age <= .10 and 0 <= observation_age <= .50)


def start_request_ready(request, now_ns, eligible, profile, duration):
    """Allow the same fresh START to wait for the next eligible queue window."""
    age = (now_ns - int(request['created_monotonic_ns'])) / 1e9
    if not 0 <= age <= 2:
        raise ValueError('START rejected: request freshness lost')
    if request.get('profile') != profile or request.get('duration_s') != duration:
        raise ValueError('START profile/duration differs from live trial')
    return bool(eligible)


def main():
    import dora
    import pyarrow as pa
    if os.environ.get('ROS_SECURITY_ENABLE') != 'true' or os.environ.get('ROS_SECURITY_STRATEGY') != 'Enforce':
        raise ValueError('physical ACT bridge requires enforced command ownership security')
    out = Path(os.environ['ACT_TRIAL_DIR']).resolve()
    if not out.is_dir() or (out/'start.request').exists() or (out/'bridge.jsonl').exists():
        raise ValueError('fresh trial directory without START or previous bridge log is required')
    bundle = json.loads(DEFAULT_BUNDLE.read_text())
    config = json.loads((out/'trial_config.json').read_text())
    from scripts.act_trial.action_queue import contract as action_queue_contract
    profile = profile_from_config(config)
    action_schedule = action_queue_contract(profile['name'])
    if config.get('action_schedule') != action_schedule:
        raise ValueError('action schedule changed; prepare a new trial')
    from scripts.act_trial.rolling_actions import RollingActions
    rolling = RollingActions() if profile['name'] == 'visual32' else None
    motion_generator = motion_from_config(config)
    check_live_host(motion_generator)
    stream = (out/'bridge.jsonl').open('x', buffering=1)
    def log(kind, **data):
        row = {'kind': kind, 'wall_ns': time.time_ns(), 'monotonic_ns': time.monotonic_ns(), **data}
        stream.write(json.dumps(preserve_nonfinite(row), allow_nan=False, default=str)+'\n')
    if rolling is not None:
        sources = [Path(__file__).with_name(name) for name in
                   ('rolling_actions.py', 'queued_policy_client.py', 'guard.py', 'ros_bridge.py')]
        log('rolling_executor_settings', action_schedule=action_schedule,
            source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
            timing='same bridge clock for q/qd; original contributor timestamps retained')
    # Refuse a missing authority before creating ROS or passive CAN observers.
    authority = AuthorityClient()
    if profile['name'] == 'visual32' and not authority.request('status').get('supports_dora_arm_velocity'):
        authority.close()
        raise ValueError('gateway must be restarted with position+velocity command support')
    if profile['command_gripper'] and 'right_arm_gripper' not in authority.request('status').get('supported_dora_scopes', []):
        authority.close()
        raise ValueError('gateway lacks task profile support; operator must restart gateway and synchronize hold')
    ros = Inputs()
    telemetry = Telemetry(out/'bridge_raw_can.jsonl')
    node = dora.Node()
    status_writer = StatusWriter(out/'ready.json')
    last_renewal = 0.
    guard = None; action = None; action_meta = {}; action_at = None
    began = time.monotonic(); initialized = None; stationary_since = None
    last_policy_fault = began; last_status = 0.; last_reference = None
    started = False; dispatched = False; stopped = False; terminal = None
    reference_history = []; expected_gains = None; episode_completion = None
    quit_requested = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: quit_requested.set())

    def stop(reason, completion=None):
        nonlocal stopped, terminal
        if stopped:
            return
        stopped = True
        if rolling is not None:
            rolling.stop()
        status_writer.close()
        # Explicit last target hold. Never empty trajectory or measured-position reset.
        hold_error = None
        if authority.token is not None:
            try:
                if reason in ('completed_three_second_trial', 'duration_limit_reached',
                              'returned_to_mechanical_zero'):
                    authority.release()
                else:
                    authority.request('freeze', reason=reason)
                log('gateway_hold_requested', reason=reason,
                    positions_rad=None if last_reference is None else last_reference.tolist(),
                    actual_motor_dispatch='see bridge_raw_can.jsonl')
            except Exception as exc:
                hold_error = repr(exc)
        terminal = {'reason': reason, 'started': started, 'ROS_remains_enabled_holding': True,
                    'execution_profile':profile, 'task_success':None,
                    'motion_generator':motion_generator,
                    'action_schedule':action_schedule,
                    'collision_checks_enabled':collision_checks_enabled(profile['name']),
                    'episode_completion': completion,
                    'last_reference': None if last_reference is None else last_reference.tolist(),
                    'explicit_hold_publication_error': hold_error,
                    'wall_ns': time.time_ns(), 'monotonic_ns': time.monotonic_ns()}
        write_json(out/'result.json', terminal)
        write_json(out/'ready.json', {'ready_for_start': False, 'started': started,
                                     'stage': 'stopped', **terminal})
        try: node.send_output('observer_command', pa.array(['stop']))
        except Exception: pass
        try: node.send_output('execution_stop', pa.array([reason]))
        except Exception as exc:
            log('executor_stop_notification_failed', error=repr(exc))
        log('terminal', **terminal)
        print('ACT_STOPPED: '+reason+'; ROS remains enabled; inspect result.json', flush=True)

    try:
        for event in node:
            if quit_requested.is_set() or event['type'] == 'STOP':
                stop('operator_or_graph_stop'); break
            if event['type'] != 'INPUT':
                continue
            event_id = event['id']
            if event_id == 'policy_actions':
                if stopped:
                    continue
                try:
                    if rolling is None:
                        raise ValueError('unexpected full prediction on legacy bridge')
                    from observation_timing import validate_timing
                    metadata = dict(event.get('metadata') or {})
                    received_ns = time.monotonic_ns()
                    validate_timing(metadata, received_ns, max_source_age_ns=200_000_000,
                                    max_source_skew_ns=100_000_000,
                                    max_observation_age_ns=500_000_000)
                    positions = event['value'].to_pylist()
                    accepted = rolling.add(positions, metadata, received_ns)
                    log('rolling_chunk_received', accepted=accepted,
                        full_positions_rad=positions, metadata=metadata)
                    if accepted:
                        action_meta = metadata
                        action_at = received_ns/1e9
                        # Readiness evidence only; active q/qd are evaluated on
                        # the bridge's control clock below, not on IPC arrival.
                        action = np.asarray(positions[0][:8], dtype=float)
                except (ValueError, KeyError, TypeError) as exc:
                    last_policy_fault = time.monotonic()
                    log('rolling_chunk_rejected', reason=str(exc))
                    if started:
                        stop('rolling_prediction_fault: '+str(exc))
                continue
            if event_id == 'shaped_action':
                if rolling is not None:
                    stop('unexpected legacy shaped action in rolling graph')
                    continue
                action = np.asarray(event['value'].to_pylist(), dtype=float)
                action_meta = dict(event.get('metadata') or {})
                action_at = time.monotonic()
                log('executor_request_received', positions_rad=action.tolist(), metadata=action_meta)
                continue
            if event_id == 'timing_fault':
                last_policy_fault = time.monotonic()
                log('policy_timing_fault', detail=event['value'].to_pylist())
                if started: stop('policy_or_executor_timing_fault')
                continue
            if event_id != 'tick':
                continue
            now = time.monotonic()
            timing = TickTiming()
            dispatch_attempted = False
            if (out/'stop.request').exists(): stop('operator_stop_file')
            try:
                state = ros.snapshot(now)
                timing.mark('ros_snapshot')
                can = telemetry.snapshot()
                timing.mark('can_snapshot')
                measured = np.array(can['positions_rad'])
                if np.max(np.abs(state['actual']-measured)) > .03:
                    raise ValueError('ROS/CAN state disagreement exceeds .03 rad')
                if guard is None:
                    audit = ros.audit(bundle, profile=profile['name'])
                    guard = Guard(measured, bundle, now=time.monotonic(), initial_command8=state['output'],
                                  profile=profile['name'], motion_generator=motion_generator)
                    if profile.get('mechanical_zero_completion') is not None:
                        episode_completion = MechanicalZeroCompletion(
                            profile['mechanical_zero_completion'])
                    # Use the same canonical complete ROS limit contract as Guard.
                    from validation_contract import ros_limits
                    limits, _, _, _ = ros_limits(bundle, None, None, required=True)
                    if profile['name'] == 'visual32':
                        from scripts.act_trial.closed_contract import limits as closed_limits
                        limits, _, _, _ = closed_limits()
                    neighborhood = inspect_start(measured, guard.home, limits, guard.scene, settings=config)
                    if not neighborhood['supplied_start_checks_passed']:
                        raise ValueError('live starting neighborhood rejected: '+str(neighborhood['failures']))
                    audit.update(profile=guard.profile(), live_neighborhood=neighborhood)
                    audit['gripper_commanded_by_bridge'] = guard.command_gripper
                    audit['trajectory_contract'] = 'zero-time positions, right arm and gripper' if guard.command_gripper else audit['trajectory_contract']
                    if profile['name'] == 'visual32':
                        audit['trajectory_contract'] = 'right arm q/qd from the same generator; gripper position only'
                    expected_gains = audit['expected_right_gains']
                    reference_history = [(time.monotonic(), guard.initial_command.copy())]
                    write_json(out/'live_preflight.json', audit)
                    log('live_preflight_passed', **audit)
                    initialized = time.monotonic()
                    continue  # Fetch fresh feedback after model/service work.
                if stopped:
                    log('post_stop_feedback', can=can, ROS=state['states'])
                    continue
                if ros.node.count_publishers(ARM_TOPIC) != 1:
                    raise ValueError('expected exactly one authorized gateway publisher')
                if guard.command_gripper and ros.node.count_publishers('/'+GRIP_CONTROLLER+'/joint_trajectory') != 1:
                    raise ValueError('expected exactly one authorized gripper publisher')
                if started and time.monotonic()-last_renewal >= .1:
                    authority.renew()
                    last_renewal = time.monotonic()
                if started:
                    speed_check = check_measured_speed(can['velocity_rad_s'][:8 if guard.command_gripper else 7], guard.measured_speed_contract)
                    log('measured_speed_check', **speed_check)
                    if not speed_check['passed']:
                        raise ValueError('measured right arm/gripper speed exceeds reviewed trial limits: '
                                         +json.dumps(speed_check))
                if started:
                    reference_history = [(t, q) for t, q in reference_history if now-t <= .15]
                else:
                    reference_history = [(now, guard.initial_command.copy())]
                check_reference_history(state['output'], can['command_positions_rad'][:8],
                                        [q for _, q in reference_history], None if guard.command_gripper else guard.initial_command[7])
                for key, quantization in (('kp', 500/4095), ('kd', 5/4095)):
                    if np.max(np.abs(np.array(can['command_'+key])[:8]-expected_gains[key])) > quantization+1e-8:
                        raise ValueError('motor '+key+' differs from validated ROS holding gains')
                action_velocity = None
                evaluated_ns = time.monotonic_ns()
                if rolling is not None and started:
                    sample = rolling.sample(evaluated_ns)
                    if sample is None:
                        # START discards dry-run predictions. Only hold until
                        # a post-START observation has produced a new chunk.
                        action = guard.command.copy()
                        action_velocity = np.zeros(8)
                    else:
                        action, action_velocity, action_meta = sample
                        log('rolling_curve_sample', positions_rad=action.tolist(),
                            velocities_rad_s=action_velocity.tolist(), metadata=action_meta)
                action_age = None
                if action is not None and 'observation_monotonic_ns' in action_meta:
                    action_age = (time.monotonic_ns()-int(action_meta['observation_monotonic_ns']))/1e9
                timing.mark('admission_and_renewal')
                decision = guard.tick(action, measured, evaluated_ns/1e9,
                                      max(can['max_age_s'], state['age']), action_age,
                                      requested_velocity=action_velocity)
                timing.mark('guard')
                log('guard', decision=decision, source_metadata=action_meta,
                    ROS=state['states'], can_health=can)
                timing.mark('guard_log')
                if decision['stage'] == 'fault':
                    stop('; '.join(decision['errors'])); continue
                if decision['stage'] == 'complete':
                    stop('duration_limit_reached' if guard.command_gripper else 'completed_three_second_trial'); continue
                if np.max(np.abs(np.array(can['velocity_rad_s'])[:8 if guard.command_gripper else 7])) <= .05:
                    if stationary_since is None: stationary_since = now
                else:
                    stationary_since = None
                action_ready = action_ready_for_start(
                    now, action_at, action_age, rolling=rolling is not None)
                eligible = (not started and stationary_since is not None and now-stationary_since >= 2
                            and action_ready
                            and now-last_policy_fault >= 1 and now-initialized >= 2)
                if (out/'start.request').exists() and not started:
                    request = json.loads((out/'start.request').read_text())
                    if start_request_ready(request, time.monotonic_ns(), eligible, profile['name'], guard.DURATION):
                        authority.acquire('dora', scope='right_arm_gripper' if guard.command_gripper else 'right_arm')
                        last_renewal = time.monotonic()
                        guard.start(time.monotonic())
                        if rolling is not None:
                            rolling.begin(guard.command, time.monotonic_ns())
                        started = True
                        log('ACT_START', request=request)
                        write_json(out/'started.json', {'wall_ns': time.time_ns(), 'monotonic_ns': time.monotonic_ns()})
                if started and decision['stage'] == 'active':
                    if episode_completion is not None:
                        completion = episode_completion.update(
                            measured, np.asarray(can['velocity_rad_s']),
                            time.monotonic(), guard.started_at, command8=guard.command)
                        log('episode_completion_check', **completion)
                        if completion['complete']:
                            stop('returned_to_mechanical_zero', completion=completion)
                            continue
                    command = np.array(decision['dispatched_proposal'])
                    if not guard.command_gripper and command[7] != guard.initial_command[7]:
                        raise ValueError('locked gripper changed')
                    count = 8 if guard.command_gripper else 7
                    velocity = np.asarray(decision['proposal_velocity_rad_s'])[:7]
                    check_dispatch_freshness(
                        now, decision['now'], max(can['max_age_s'], state['age']),
                        action_meta.get('observation_monotonic_ns'), time.monotonic(),
                        watchdog=guard.WATCHDOG, feedback_limit=guard.FEEDBACK_AGE,
                        action_limit=guard.ACTION_AGE)
                    authority.command(command[:count].tolist(),
                                      velocities=velocity.tolist() if profile['name'] == 'visual32' else None)
                    timing.mark('dispatch')
                    dispatched = True; last_reference = command.copy()
                    reference_history.append((now, command.copy()))
                    log('ROS_target_published', joint_names=NAMES[:count], positions_rad=command[:count].tolist(),
                        arm_velocities_rad_s=velocity.tolist() if profile['name'] == 'visual32' else None,
                        time_from_start_ns=0, source_metadata=action_meta,
                        actual_motor_dispatch='see bridge_raw_can.jsonl')
                source_ns = min(can['source_monotonic_ns'])
                metadata = {'source_monotonic_ns': source_ns}
                node.send_output('position_right', pa.array(measured[:8], type=pa.float32()), metadata)
                timing.mark('output_position_right')
                node.send_output('position_left', pa.array(measured[8:], type=pa.float32()), metadata)
                timing.mark('output_position_left')
                if now-last_status >= .2:
                    for name in ('status_right', 'status_left'):
                        node.send_output(name, pa.array(['started']))
                        timing.mark('output_'+name)
                    node.send_output('observer_command', pa.array(['start' if started else 'ready']))
                    timing.mark('output_observer_command')
                    node.send_output('task_prompt', pa.array(['pick up cube']))
                    timing.mark('output_task_prompt')
                    status = {'ready_for_start': eligible, 'started': started, 'stage': guard.stage,
                              'profile':profile['name'], 'duration_s':guard.DURATION,
                              'motion_generator':motion_generator,
                              'action_schedule':action_schedule,
                              'action_ready_for_start': action_ready,
                              'action_receipt_age_s': None if action_at is None else now-action_at,
                              'action_age_s': action_age, 'wall_ns': time.time_ns(),
                              'monotonic_ns': time.monotonic_ns(), 'feedback_age_s': can['max_age_s']}
                    node.send_output('trial_status', pa.array([json.dumps(status)]))
                    timing.mark('output_trial_status')
                    status_writer.submit(status); last_status = now
                    timing.mark('status_enqueue')
                timing.mark('outputs_and_status')
            except (ValueError, RuntimeError, KeyError) as exc:
                if guard is not None or now-began > 20:
                    stop(str(exc))
                elif now-last_status >= 1:
                    log('waiting_for_live_start', reason=str(exc)); last_status = now
            finally:
                if started and ('guard' in timing.phases or not stopped):
                    log('tick_timing', phases=timing.phases)
    except BaseException as exc:
        stop('bridge_exception: '+repr(exc))
        raise
    finally:
        stop('bridge_exit')
        try: node.send_output('observer_command', pa.array(['stop']))
        except Exception: pass
        authority.close(); telemetry.close(); ros.close(); stream.close()


if __name__ == '__main__':
    main()
