"""Single-session direct-CAN startup guard; no ROS handoff or automatic restart."""

import json
import time

import numpy as np


def _start_like_ros(arm):
    """Borrow ROS H1's enable/settle order, without its return_to_zero motion.

    openarm-driver 0.5.1's ``SingleArmDriver.start`` performs an immediate
    state read and refresh immediately after ``enable_all``.  The ROS
    hardware plugin instead waits 100 ms and receives once. This adaptation
    addresses a suspected startup timing issue; it does not establish the
    cause of the previously observed write_no_buffer failures. Guard below
    implements a separate, more restrictive zero/HOME motion sequence.
    """
    import openarm_can as oa

    arm.started = False
    arm._safety_stop_reason = None
    arm._last_safety_warning_time_s = None
    arm.openarm.set_callback_mode_all(oa.CallbackMode.STATE)
    arm.openarm.enable_all()
    time.sleep(0.1)
    arm.openarm.recv_all()
    arm.set_latest_state(timeout_us=500)
    arm.last_command = arm.latest_state['qpos'].copy()
    arm.last_command_time_s = time.monotonic()
    arm.last_command_dispatch_timestamp_ns = None
    arm.started = True
    return True


def _construct_without_pre_enable_refresh(driver, config):
    """Construct the Python driver without its 20 pre-enable CAN refreshes.

    ``SingleArmDriver.__init__`` primes the cache by calling
    ``fetch_position(refresh=True)`` twenty times.  ROS H1 does not issue that
    burst before enabling the motors.  Temporarily route those constructor
    calls through the driver's cache-only path; the real feedback transaction
    begins after the ROS-equivalent enable/settle sequence above.
    """
    cls = driver.SingleArmDriver
    original_fetch_position = cls.fetch_position

    def cache_only_fetch_position(self, refresh=True):
        return original_fetch_position(self, refresh=False)

    cls.fetch_position = cache_only_fetch_position
    try:
        return cls('right_arm', config)
    finally:
        cls.fetch_position = original_fetch_position


def _recovery_checker(guard):
    """Use the same driver delta/rate checks with an activation-only envelope."""
    from openarm_driver.base_safety import CompositeChecker
    from openarm_driver.safety import JointPosChecker, JointDeltaPosChecker, JointVelocityChecker

    return CompositeChecker([
        JointPosChecker(np.column_stack((guard.recovery_lower, guard.recovery_upper))),
        JointDeltaPosChecker(guard.single_jump),
        JointVelocityChecker(guard.velocity),
    ])


def _send_guarded_position(arm, guard, command, recovery_checker):
    """Scope the alternate checker to this send; always restore the driver."""
    normal_checker = arm.safety_checker
    guard.last_requested_command = np.asarray(command, dtype=float).copy()
    try:
        if guard.recovery_scope:
            guard.recovery_position(command, 'startup dispatch')
            arm.safety_checker = recovery_checker
        else:
            guard.position(command, 'HOME/ACT dispatch')
        if arm.send_position(command) is False:
            raise RuntimeError('CAN rejected guarded command')
        # The driver also rate-limits using its actual dispatch time. Track
        # what was sent, so settling/tracking do not use an unsent waypoint.
        guard.command = np.asarray(arm.last_command, dtype=float).copy()
        guard.last_dispatched_command = guard.command.copy()
    finally:
        arm.safety_checker = normal_checker


def _require_can_health(arm):
    """An enabled cache alone is insufficient; every axis must reply recently."""
    motors, bus = arm.get_health()
    if len(motors) != 8 or any(status != 'ENABLED' for status in motors):
        raise RuntimeError(f'CAN motors not healthy on {arm.can_interface}: {motors}')
    errors = {name: count for name, count in bus.items() if name != 'carrier' and count}
    if not bus.get('carrier') or errors:
        raise RuntimeError(f'CAN bus not healthy on {arm.can_interface}: {bus}')
    # This guarded configuration uses eight MIT axes, including J8.
    collection = arm.openarm.get_arm()
    for i in range(8):
        link = collection.get_link_stats(i)
        if link.responses == 0 or link.is_stale(0.2):
            raise RuntimeError(f'CAN feedback stale or missing: J{i + 1}')
    return motors, bus


def _validate_driver_config(config, guard):
    """Fail before enabling if Guard and driver disagree about coordinates."""
    comparisons = (
        (config.get_joint_limits('right_arm'), np.column_stack((guard.lower, guard.upper))),
        (config.get_joint_offsets('right_arm'), np.zeros(8)),
        (config.get_joint_velocity_limits(), guard.velocity),
        (config.get_joint_delta_position_limits(), guard.single_jump),
    )
    for actual, expected in comparisons:
        if np.shape(actual) != np.shape(expected) or not np.allclose(actual, expected):
            raise ValueError('Guard/driver limits or ROS coordinate frame disagree')
    if config.get_gripper_posforce():
        raise ValueError('guarded startup requires eight MIT axes')
    if config.get_start_config()['moves'] or config.get_stop_config()['moves']:
        raise ValueError('guarded startup requires empty driver start/stop moves')


class Guard:
    """Bound commands in elapsed time and latch every fault until process restart."""

    def __init__(self, cfg, measured, now, hold_only=False, verify_home_only=False):
        if hold_only and verify_home_only:
            raise ValueError('hold-only and verify-home-only are mutually exclusive')
        self.cfg = cfg
        self.lower = self.vector(cfg['lower'])
        self.upper = self.vector(cfg['upper'])
        if np.any(self.lower >= self.upper):
            raise ValueError('invalid joint limits')
        for key in ('zero_tolerance', 'tolerance', 'tracking_tolerance',
                    'timeout', 'settle', 'watchdog', 'action_timeout', 'episode_timeout'):
            if not np.isfinite(cfg[key]) or cfg[key] <= 0:
                raise ValueError(f'invalid {key}')
        self.velocity = self.vector(cfg['joint_velocity_limits'])
        self.single_jump = self.vector(cfg['joint_delta_position_limits'])
        if np.any(self.velocity <= 0) or np.any(self.single_jump <= 0):
            raise ValueError('invalid official joint limits')
        self.feedback_period = float(cfg.get('feedback_period', 0.02))
        if not np.isfinite(self.feedback_period) or self.feedback_period <= 0:
            raise ValueError('invalid feedback_period')

        # The ROS controller receives a smooth trajectory from MoveIt.  Dora
        # used to change ``target`` at a phase boundary and then slew toward
        # it at the driver's *maximum* joint velocity.  A delayed 20--50 ms
        # feedback sample therefore saw a command lead of one full velocity
        # step (J7: 6.3 rad/s * 20 ms = 0.126 rad), which is already greater
        # than the 0.120 rad tracking guard.  Keep the endpoint contract, but
        # give HOME its own bounded speed and time parameterization.
        trajectory = cfg.get('home_trajectory', {})
        if not isinstance(trajectory, dict):
            raise ValueError('invalid home_trajectory configuration')
        try:
            speed_fraction = float(trajectory.get('speed_fraction', 0.25))
            min_phase_duration = float(
                trajectory.get('min_phase_duration', 2.0 * self.feedback_period)
            )
            self.progress_timeout = float(
                trajectory.get('progress_timeout', cfg['watchdog'])
            )
        except (TypeError, ValueError):
            raise ValueError('invalid HOME trajectory timing') from None
        if (
            not np.isfinite(speed_fraction)
            or speed_fraction <= 0
            or speed_fraction > 1
            or not np.isfinite(min_phase_duration)
            or min_phase_duration <= 0
            or not np.isfinite(self.progress_timeout)
            or self.progress_timeout <= 0
        ):
            raise ValueError('invalid HOME trajectory timing')
        self.min_phase_duration = min_phase_duration
        # Leave enough room for one delayed feedback sample before the
        # existing tracking threshold is reached.  This is a command-side
        # bound, not a relaxation of the tracking guard.
        self.home_speed = np.minimum(
            self.velocity * speed_fraction,
            self.cfg_tracking_limit(cfg) / (4.0 * self.feedback_period),
        )
        if np.any(self.home_speed <= 0) or not np.all(np.isfinite(self.home_speed)):
            raise ValueError('invalid HOME trajectory speed')

        # Dora explicitly bounds H1's activation-time recovery. It is only
        # active while returning from the measured startup pose to nominal
        # zero (or while running the CAN hold diagnostic); normal HOME/ACT
        # commands continue to use ``lower``/``upper`` below.  Keeping this
        # scope explicit prevents an out-of-range startup feedback value from
        # silently widening the policy command limits.
        recovery = cfg['return_to_zero']
        if not isinstance(recovery, dict):
            raise ValueError('invalid return_to_zero configuration')
        self.recovery_lower = self.vector(recovery['lower'])
        self.recovery_upper = self.vector(recovery['upper'])
        if np.any(self.recovery_lower >= self.recovery_upper):
            raise ValueError('invalid return_to_zero envelope')
        self.zero_steps = recovery['steps']
        try:
            self.zero_period = float(recovery['period'])
        except (TypeError, ValueError):
            raise ValueError('invalid return_to_zero period') from None
        if type(self.zero_steps) is not int:
            raise ValueError('invalid return_to_zero steps')
        if self.zero_steps <= 0 or not np.isfinite(self.zero_period) or self.zero_period <= 0:
            raise ValueError('invalid return_to_zero timing')
        try:
            self.zero_timeout = float(recovery['timeout'])
        except (TypeError, ValueError):
            raise ValueError('invalid return_to_zero timeout') from None
        if not np.isfinite(self.zero_timeout) or self.zero_timeout <= 0:
            raise ValueError('invalid return_to_zero timeout')
        zero = self.position(cfg['zero'], 'configured zero target')
        if np.any(zero[:-1] != 0):
            raise ValueError('return-to-zero requires J1-J7 nominal zero')
        side = self.position(cfg['side_prehome'], 'configured side-prehome target')
        home = self.position(cfg['home'], 'configured HOME target')
        measured = self.recovery_position(
            measured, 'initial feedback before return-to-zero',
        )
        # J8 is held through the arm-zero and fixed HOME phases, so it must be
        # valid under the normal command envelope even though J1-J7 use the
        # activation-only recovery envelope.
        if measured[-1] < self.lower[-1] or measured[-1] > self.upper[-1]:
            raise ValueError(
                'initial gripper feedback before return-to-zero: '
                f'J8={measured[-1]:.6f} rad outside '
                f'[{self.lower[-1]:.6f}, {self.upper[-1]:.6f}]'
            )
        # H1 returns only J1-J7 to nominal zero. J8 is a separate gripper
        # actuator and may already be open or closed when the arm starts.
        zero_arm = zero.copy()
        zero_arm[-1] = measured[-1]
        self.recovery_position(zero_arm, 'configured return-to-zero endpoint')
        # Keep the measured gripper pose until the arm reaches HOME.
        side[-1] = measured[-1]
        lifted = home.copy()
        lifted[-1] = measured[-1]
        self.targets = [zero_arm, side, lifted, home]
        self.home_phase_durations = [
            self._phase_duration(start, end)
            for start, end in zip(self.targets[0:-1], self.targets[1:])
        ]
        self.command = measured.copy()
        self.target = measured.copy() if hold_only else zero_arm.copy()
        self.state = 'hold_only' if hold_only else 'returning_zero'
        # A fault may only hold the last checked startup command. It never
        # reactivates recovery, even if STOP/START or a policy target arrives.
        self.recovery_scope = True
        self.recovery_start = measured.copy()
        self.recovery_started = now
        self.phase = 0
        self.since = self.last_tick = now
        self.phase_started = now
        self.phase_start = self.command.copy()
        self.progress_distance = None
        self.progress_at = now
        self.last_feedback = measured.copy()
        self.last_feedback_age = self.feedback_period
        self.last_dt = 0.0
        self.last_waypoint = self.command.copy()
        self.last_requested_command = self.command.copy()
        self.last_dispatched_command = self.command.copy()
        self.settled = None
        self.last_action = None
        self.error = ''
        self.verify_target = None
        if verify_home_only:
            self.position(measured, 'verify-HOME hold command')
            if np.any(np.abs(measured - home) > cfg['tolerance']):
                raise ValueError('physical HOME absent; run right-arm HOME smoke flow first')
            self.recovery_scope = False
            self.state = 'verifying_home'
            self.phase = 3
            self.verify_target = home.copy()
            self.target = measured.copy()

    @staticmethod
    def vector(value):
        result = np.asarray(value, dtype=float).copy()
        if result.shape != (8,) or not np.all(np.isfinite(result)):
            raise ValueError('expected eight finite joint positions')
        return result

    @staticmethod
    def cfg_tracking_limit(cfg):
        """Read the unchanged tracking guard while validating trajectory data."""
        try:
            value = float(cfg['tracking_tolerance'])
        except (KeyError, TypeError, ValueError):
            raise ValueError('invalid tracking_tolerance') from None
        if not np.isfinite(value) or value <= 0:
            raise ValueError('invalid tracking_tolerance')
        return value

    def _phase_duration(self, start, end):
        """Compute a smoothstep duration for the actual phase start pose."""
        # Cubic smoothstep has a maximum normalized derivative of 1.5.  The
        # measured phase start can differ slightly from the previous endpoint,
        # so recompute this after every ROS H2-style phase transition.
        duration = np.max(1.5 * np.abs(end - start) / self.home_speed)
        return max(self.min_phase_duration, float(duration))

    def position(self, value, context='position'):
        value = self.vector(value)
        outside = (value < self.lower) | (value > self.upper)
        if np.any(outside):
            details = '; '.join(
                f'J{i + 1}={value[i]:.6f} rad outside '
                f'[{self.lower[i]:.6f}, {self.upper[i]:.6f}]'
                for i in np.flatnonzero(outside)
            )
            raise ValueError(f'{context}: joint limit exceeded: {details}')
        return value

    def recovery_position(self, value, context='return-to-zero position'):
        """Validate feedback/commands in the activation-only recovery scope."""
        value = self.vector(value)
        outside = (value < self.recovery_lower) | (value > self.recovery_upper)
        if np.any(outside):
            details = '; '.join(
                f'J{i + 1}={value[i]:.6f} rad outside '
                f'[{self.recovery_lower[i]:.6f}, {self.recovery_upper[i]:.6f}]'
                for i in np.flatnonzero(outside)
            )
            raise ValueError(f'{context}: startup envelope exceeded: {details}')
        return value

    def halt(self, reason):
        # Preserve the first cause instead of hiding it behind later faults.
        # Smoke tests have no UI subscriber, so the fault must reach stdout.
        if self.state in ('home_error', 'disabled'):
            return
        print(f'HOME_ERROR: phase={self.phase} state={self.state}: {reason}', flush=True)
        self.state, self.error = 'home_error', reason
        self.target = self.command.copy()

    def start(self, measured, now):
        if self.state != 'home_ready':
            return False
        measured = self.position(measured, 'START feedback')
        if np.any(np.abs(measured - self.targets[-1]) > self.cfg['tolerance']):
            self.halt('HOME verification lost before START')
            return False
        self.state = 'active'
        self.since = self.last_action = now
        return True

    def action(self, value, now):
        if self.state != 'active':
            return
        value = self.position(value, 'ACT target')
        if np.any(np.abs(value - self.command) > self.single_jump):
            raise ValueError('ACT target jump exceeds official single-command limit')
        self.target = value
        self.last_action = now

    def timing_fault(self, reason):
        """Latch an upstream observation/action freshness fault.

        The policy and action-executor nodes deliberately fail closed when a
        source timestamp expires.  Holding the previous command would leave
        the arm apparently active, so route the fault through the same latched
        halt path used for tracking and watchdog failures.
        """
        self.halt(f'action timing fault: {reason}')

    def tick(self, measured, now, feedback_age=None):
        # run_guarded synchronizes command from driver.last_command after
        # every dispatch. Settling must use that confirmed command as well as
        # the proposed next waypoint: the driver may rate-limit the latter.
        dispatched_command = self.command.copy()
        if self.recovery_scope:
            measured = self.recovery_position(measured, 'runtime feedback')
        else:
            measured = self.position(measured, 'runtime feedback')
        dt = now - self.last_tick
        self.last_tick = now
        self.last_feedback = measured.copy()
        self.last_dt = dt
        if feedback_age is None:
            feedback_age = self.feedback_period
        try:
            feedback_age = float(feedback_age)
        except (TypeError, ValueError):
            self.halt('invalid feedback age')
            feedback_age = self.feedback_period
        if not np.isfinite(feedback_age) or feedback_age < 0:
            self.halt('invalid feedback age')
            feedback_age = self.feedback_period
        self.last_feedback_age = feedback_age
        if dt < 0 or dt > self.cfg['watchdog']:
            self.halt('control tick watchdog expired')
        tracking_error = np.abs(measured - self.command)
        if np.any(tracking_error > self.cfg['tracking_tolerance']):
            i = int(np.argmax(tracking_error))
            self.halt(
                f'command tracking error: J{i + 1} measured={measured[i]:.6f} '
                f'command={self.command[i]:.6f} error={tracking_error[i]:.6f} rad '
                f'limit={self.cfg["tracking_tolerance"]:.6f}'
            )
        if self.state == 'returning_zero' and now - self.recovery_started > self.zero_timeout:
            self.halt(f'return-to-zero timeout: measured={measured.tolist()} '
                      f'zero={self.targets[0].tolist()}; require zero tolerance '
                      'and normal HOME feedback bounds')
        if self.state in ('homing', 'verifying_home') and now - self.since > self.cfg['timeout']:
            self.halt('HOME timeout')
        if self.state == 'active' and (now - self.last_action > self.cfg['action_timeout']
                                      or now - self.since > self.cfg['episode_timeout']):
            self.halt('episode/action timeout')
        if self.state == 'homing':
            # A slow command must still make physical progress.  Tracking
            # error alone cannot catch a command that remains just under the
            # threshold forever, so latch a separate no-progress fault.
            distance = float(np.max(np.abs(self.target - measured)))
            # Once feedback is inside the normal HOME acceptance band, leave
            # the progress timer idle while the required settle interval runs.
            # Otherwise a healthy endpoint would be mistaken for a stalled
            # trajectory when ``progress_timeout`` equals ``settle``.
            if distance <= self.cfg['tolerance']:
                self.progress_distance = distance
                self.progress_at = now
            elif self.progress_distance is None or distance < self.progress_distance - 1e-5:
                self.progress_distance = distance
                self.progress_at = now
            elif now - self.progress_at >= self.progress_timeout - 1e-9:
                self.halt(
                    f'HOME progress timeout: measured={measured.tolist()} '
                    f'target={self.target.tolist()} '
                    f'error={distance:.6f} rad'
                )
        if self.state == 'returning_zero':
            # Match ROS H1's 200 x 10 ms measured-pose interpolation.  The
            # subsequent velocity/single-jump clamp keeps a delayed Dora tick
            # from producing a large catch-up command.
            elapsed = max(0.0, now - self.recovery_started)
            step = min(self.zero_steps, int(elapsed / self.zero_period))
            alpha = step / self.zero_steps
            waypoint = self.recovery_start + alpha * (
                self.targets[0] - self.recovery_start
            )
        elif self.state == 'homing':
            phase_index = self.phase - 1
            if 0 <= phase_index < len(self.home_phase_durations):
                phase_elapsed = max(0.0, now - self.phase_started)
                phase_duration = self.home_phase_durations[phase_index]
                u = np.clip(phase_elapsed / phase_duration, 0.0, 1.0)
                # Smooth first and last derivatives at each ROS-equivalent
                # waypoint boundary.  This target is still slew-limited below
                # so delayed Dora ticks cannot issue a catch-up jump.
                blend = u * u * (3.0 - 2.0 * u)
                waypoint = self.phase_start + blend * (
                    self.target - self.phase_start
                )
            else:
                waypoint = self.target
        else:
            waypoint = self.target
        self.last_waypoint = waypoint.copy()
        if self.state in ('returning_zero', 'homing', 'active'):
            # Match the driver's MAX_COMMAND_DT_S, including delayed ticks.
            # HOME keeps its configured speed envelope; ACT uses the same
            # tracking margin with the actual feedback age. Recovery retains
            # the ROS H1 velocity limits.
            if self.state == 'returning_zero':
                speed = self.velocity
            else:
                # The explicit CAN refresh is paced independently of the Dora
                # timer. Bound command lead using the age of the actual reply.
                age_bound = max(self.feedback_period, feedback_age)
                feedback_speed = self.cfg_tracking_limit(self.cfg) / (2.0 * age_bound)
                nominal_speed = self.home_speed if self.state == 'homing' else self.velocity
                speed = np.minimum(nominal_speed, feedback_speed)
            step = np.minimum(self.single_jump, speed * min(.04, max(0, dt)))
            self.command += np.clip(waypoint - self.command, -step, step)
        if self.state in ('returning_zero', 'homing', 'verifying_home', 'home_ready'):
            if self.state == 'returning_zero':
                trajectory_done = (
                    now - self.recovery_started >= self.zero_steps * self.zero_period
                )
                at_target = trajectory_done and np.max(
                    np.abs(self.command - self.targets[0])
                ) < 1e-8 and np.max(
                    np.abs(dispatched_command - self.targets[0])
                ) < 1e-8
                verified = np.max(
                    np.abs(measured[:-1] - self.targets[0][:-1])
                ) <= self.cfg['zero_tolerance']
                # Do not hand a still-negative J4 over to normal HOME. Keep
                # holding zero until feedback meets both gates, or timeout.
                verified = verified and np.all(
                    (measured >= self.lower) & (measured <= self.upper)
                )
            else:
                at_target = (np.max(np.abs(self.command - self.target)) < 1e-8
                             and np.max(np.abs(dispatched_command - self.target)) < 1e-8)
                verification_target = (self.verify_target
                                       if self.state == 'verifying_home'
                                       else self.target)
                verified = np.max(np.abs(measured - verification_target)) <= self.cfg['tolerance']
            if self.state == 'home_ready' and not verified:
                self.halt('HOME hold verification lost')
            elif self.state in ('returning_zero', 'homing', 'verifying_home'):
                self.settled = (now if self.settled is None else self.settled) if at_target and verified else None
                if self.settled is not None and now - self.settled >= self.cfg['settle']:
                    if self.state == 'returning_zero':
                        self.position(self.command, 'zero command before HOME')
                        self.position(measured, 'zero feedback before HOME')
                        self.recovery_scope = False
                        self.since = now
                        print('RETURN_TO_ZERO: complete; HOME/ACT command limits active', flush=True)
                    print(f'HOME verified phase {self.phase}: {measured.tolist()}', flush=True)
                    self.phase += 1
                    self.settled = None
                    if self.phase == len(self.targets):
                        self.state = 'home_ready'
                        print('HOME_STATE: home_ready', flush=True)
                    else:
                        self.state = 'homing'
                        self.target = self.targets[self.phase].copy()
                        self.phase_started = now
                        # ROS H2 discards the previous stage endpoint and
                        # starts the next MoveIt plan from fresh measured
                        # feedback.  Keep that contract in Dora so a small
                        # endpoint tracking error is not turned into a hidden
                        # command jump at the phase boundary.
                        self.phase_start = measured.copy()
                        self.home_phase_durations[self.phase - 1] = self._phase_duration(
                            self.phase_start, self.target
                        )
                        self.progress_distance = float(
                            np.max(np.abs(self.target - measured))
                        )
                        self.progress_at = now
                        print(
                            f'HOME_TARGET: phase={self.phase} state={self.state} '
                            f'q={self.target.tolist()}', flush=True,
                        )
        if self.recovery_scope:
            return self.recovery_position(self.command, 'return-to-zero command')
        return self.position(self.command)


def run_guarded(node, args, config, driver, pa, build_qpos, extract):
    """Retain one CAN driver from startup until exit; normal STOP holds torque."""
    with open(args.startup_home) as stream:
        cfg = json.load(stream)
    # Validate configuration before enabling hardware.
    validated = Guard(cfg, cfg['zero'], time.monotonic())
    _validate_driver_config(config, validated)
    # 0.5.1 constructor performs 20 CAN feedback refreshes before arm.start().
    # Publish this boundary before constructing, otherwise a blocked CAN read
    # looks like a slow control loop while the Dora tick queue grows.
    print('STARTUP: SingleArmDriver constructor begin (can0, CAN-FD)', flush=True)
    node.send_output('status', pa.array(['starting']))
    node.send_output('home_status', pa.array(['starting']))
    arm = _construct_without_pre_enable_refresh(driver, config)
    print('STARTUP: SingleArmDriver constructor complete', flush=True)
    guard = None
    disabled = False
    last_publish = 0
    last_feedback_refresh = float('-inf')

    def publish():
        nonlocal last_publish
        home_state = guard.state if guard else 'home_error'
        status = 'started' if home_state == 'active' else 'stopped'
        node.send_output('status', pa.array([status]))
        node.send_output('home_status', pa.array([home_state]),
                         {'detail': guard.error if guard else 'startup failed'})
        last_publish = time.monotonic()

    try:
        print('STARTUP: arm.start begin (ROS H1 enable/settle order only; no zero motion yet)', flush=True)
        if not _start_like_ros(arm):
            raise RuntimeError('CAN start failed')
        print('STARTUP: arm.start complete', flush=True)
        # Match the verified ROS hardware startup: enable_all() is followed by
        # a short settle window before the first feedback transaction. The
        # driver's start() helper performs enable and read back-to-back, which
        # can leave the first read in the pre-enable window on this hardware.
        time.sleep(0.1)
        print('STARTUP: feedback read begin', flush=True)
        initial_position = arm.fetch_position(refresh=True)
        print(f'STARTUP: feedback read complete {list(initial_position)}', flush=True)
        # openarm-driver 0.5.1 can return cached qpos even when no motor has
        # answered the CAN request. Do not treat that cache as real feedback.
        time.sleep(0.25)
        initial_position = arm.fetch_position(refresh=True)
        health = arm.get_health()
        motor_status, bus_status = health
        print(f'STARTUP: CAN health motors={motor_status} bus={bus_status}', flush=True)
        _require_can_health(arm)
        print('STARTUP: constructing guarded HOME state', flush=True)
        guard = Guard(
            cfg,
            initial_position,
            time.monotonic(),
            getattr(args, 'guard_hold_only', False),
            getattr(args, 'guard_verify_home_only', False),
        )
        recovery_checker = _recovery_checker(guard)
        # H1 interpolation starts at measured feedback; dispatch that hold
        # before any zero-directed waypoint. This is not an encoder reset.
        arm.last_command = guard.command.copy()
        arm.last_command_time_s = time.monotonic()
        _send_guarded_position(arm, guard, guard.command, recovery_checker)
        guard.recovery_started = guard.since = guard.last_tick = time.monotonic()
        print('STARTUP: guarded HOME state ready', flush=True)
        if guard.state == 'returning_zero':
            print(
                f'RETURN_TO_ZERO: measured_start={guard.recovery_start.tolist()} '
                f'steps={guard.zero_steps} period={guard.zero_period} '
                f'envelope={list(zip(guard.recovery_lower, guard.recovery_upper))}',
                flush=True,
            )
        print(
            f'HOME_TARGET: phase={guard.phase} state={guard.state} '
            f'q={guard.target.tolist()}', flush=True,
        )
        publish()
        while True:
            now = time.monotonic()
            if now - guard.last_tick > cfg['watchdog'] and guard.state not in ('home_error', 'disabled'):
                guard.halt('control input watchdog expired')
                publish()
            if now - last_publish >= 0.5:
                publish()
            if node.is_empty():
                time.sleep(0.002)
                continue
            event = node.next()
            if event is None or event['type'] == 'STOP':
                break
            if event['type'] != 'INPUT':
                continue
            event_id = event['id']
            if event_id == 'command':
                command = event['value'][0].as_py()
                if command == 'quit':
                    break
                if command == 'emergency_stop':
                    guard.halt('emergency disable; restart required')
                    guard.state = 'disabled'
                    arm.stop()
                    disabled = True
                elif command == 'stop':
                    guard.halt('operator STOP; holding last command, restart required')
                elif command == 'start' and not disabled:
                    start_position = arm.fetch_position(refresh=True)
                    _require_can_health(arm)
                    last_feedback_refresh = time.monotonic()
                    guard.start(start_position, last_feedback_refresh)
                publish()
            elif event_id == 'timing_fault':
                reason = event['value'][0].as_py()
                if guard is not None:
                    guard.timing_fault(reason)
                    publish()
            elif not disabled and event_id == 'request_position':
                # A full CAN refresh reads all eight motors and is synchronous.
                # Refreshing on every Dora tick overran the Nano input queue.
                # Keep command output at the configured tick rate, but refresh
                # real feedback at a bounded rate and use the driver's latest
                # cached position between refreshes.
                request_now = time.monotonic()
                if request_now - last_feedback_refresh >= guard.feedback_period:
                    measured = arm.fetch_position(refresh=True)
                    last_feedback_refresh = time.monotonic()
                else:
                    measured = arm.fetch_position(refresh=False)
                _require_can_health(arm)
                metadata = arm.observation_metadata()
                feedback_age = max(
                    0.0,
                    (time.monotonic_ns() - int(metadata['source_monotonic_ns'])) / 1e9,
                )
                try:
                    command = guard.tick(measured, time.monotonic(), feedback_age)
                except ValueError as exc:
                    guard.halt(str(exc))
                    publish()
                    continue
                _send_guarded_position(arm, guard, command, recovery_checker)
                node.send_output(
                    'position',
                    build_qpos(np.asarray(measured, dtype=np.float32)),
                    metadata,
                )
            elif not disabled and event_id == 'move_position':
                try:
                    if guard.state == 'active':
                        guard.action(extract(event['value'], 'qpos'), time.monotonic())
                except (ValueError, TypeError) as exc:
                    guard.halt(str(exc))
                    publish()
    except Exception as exc:
        if guard:
            guard.halt(str(exc))
        print(f'HOME_ERROR: {exc}', flush=True)
        publish()
        raise
    finally:
        # Explicit E-stop, quit, process exit or transport failure disables motors.
        # Operator must support the arm before quitting. Physical E-stop is independent.
        if not disabled:
            arm.stop()
