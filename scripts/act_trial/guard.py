"""Pure right-arm experiment guard; importing this module never opens CAN.

The caller supplies already blended/filtered ACT requests. This class leaves
those requests intact in each returned record, clips the execution target to a
selected profile's limits. Closed-transit streams reject excessive speed without
reshaping positions; legacy modes bound command speed and acceleration. The
returned ``dispatched_proposal`` is only a proposal, never proof of dispatch.
The motor boundary must log its actual dispatched command separately and must
not silently modify a proposal without reconciling the guard state.

Usage::

    guard = Guard(initial_measured_16, recorded_bundle, now=monotonic())
    record = guard.tick(None, measured_16, now, feedback_age)
    guard.start(now)  # after the caller has verified physical holding
    record = guard.tick(filtered_request_8, measured_16, now, feedback_age,
                        action_age=age_of_original_observation)

No HOME/zero motion, left-arm command, or automatic restart is generated.
The task and visual32 profiles permit right gripper motion; their legacy
MuJoCo collision admission is disabled.
the default commissioning profile holds the gripper and lasts3 seconds.
A fault or the selected duration limit latches a position hold.
That hold is not a torque-disable command or a dynamic braking guarantee.
"""
from pathlib import Path
import sys
import time

import numpy as np
from scripts.act_trial.speed_contract import measured_speed_contract
from scripts.act_trial.profiles import execution_profile, collision_checks_enabled
from scripts.act_trial.tracking_contract import tracking_contract
from scripts.act_trial.motion_generator import STREAM_MODE

_REPLAY = Path(__file__).resolve().parents[1] / 'home_replay'
if str(_REPLAY) not in sys.path:
    sys.path.insert(0, str(_REPLAY))
from core import NAMES, preserve_nonfinite
from simulate import Scene
from validation_contract import ros_limits


class Guard:
    """Legacy conservative guards and the closed-transit position stream."""

    DURATION = 3.0
    DISPLACEMENT = .04
    SPEED = .25
    ACCELERATION = .25
    TRACKING = .03
    FEEDBACK_AGE = .10
    WATCHDOG = .10
    ACTION_AGE = .50  # Existing Dora observation-age gate; not relaxed here.
    NOMINAL_PERIOD = .01
    COLLISION_STEP = .002
    _EPS = 1e-10

    def __init__(self, initial16, bundle, *, now, scene=None, initial_command8=None, profile='commissioning',
                 motion_generator=None):
        motion_generator = motion_generator or (STREAM_MODE if profile == 'visual32' else 'legacy')
        if profile == 'visual32' and motion_generator != STREAM_MODE:
            raise ValueError('closed-transit ACT requires position_stream_v1; prepare a new trial')
        if profile != 'visual32' and motion_generator == STREAM_MODE:
            raise ValueError('position_stream_v1 is only supported for closed-transit ACT')
        self.direct_stream = motion_generator == STREAM_MODE
        self.execution_profile = execution_profile(profile)
        self.DURATION = self.execution_profile['duration_s']
        self.DISPLACEMENT = self.execution_profile['arm_envelope_rad']
        self.command_gripper = self.execution_profile['command_gripper']
        self.tracking_contract = tracking_contract(bundle,profile)
        self.tracking_warning = np.array(self.tracking_contract['warning_thresholds_rad'])
        self.initial = self._vector(initial16, 16, 'initial measured state')
        self.created_at = self._time(now)
        # Model construction may take longer than a control period. Watchdog
        # timing begins with the first fresh checked holding observation, not
        # before construction. START is impossible until that check succeeds.
        self.last_tick = None
        self.initial_command = (self.initial[:8].copy() if initial_command8 is None
                                else self._vector(initial_command8, 8,
                                                  'initial controller reference'))
        self.command = self.initial_command.copy()
        self.velocity = np.zeros(8)
        self.previous_measured = self.initial.copy()
        self.started_at = None
        self.stage = 'holding'
        self.error = None
        self.last_hold_check_at = None
        self.collision_checks_enabled = collision_checks_enabled(profile)
        self.scene = ((Scene(bundle, position_only=True) if scene is None else scene)
                      if self.collision_checks_enabled else None)
        limits, velocities, self.limit_sources, self.limit_policy = ros_limits(
            bundle, np.zeros((16, 2)), np.zeros(16), required=True)
        if profile == 'visual32':
            from scripts.act_trial.closed_contract import limits as closed_limits
            limits, velocities, self.limit_sources, self.limit_policy = closed_limits()
        limits = np.asarray(limits, dtype=float)
        velocities = np.asarray(velocities, dtype=float)
        if (limits.shape != (16, 2) or velocities.shape != (16,)
                or not np.all(np.isfinite(limits))
                or not np.all(np.isfinite(velocities))
                or np.any(limits[:, 0] >= limits[:, 1])
                or np.any(velocities <= 0)):
            raise ValueError('invalid canonical ROS/MoveIt limits')
        self.lower, self.upper = limits[:8].T.copy()
        if self.direct_stream:
            self.speed = velocities[:8].copy()
            self.ACCELERATION = None
            self.measured_speed_contract = {
                'limits_rad_s': self.speed.tolist(),
                'command_limits_rad_s': self.speed.tolist(),
                'basis': 'pinned collection URDF intersect current MoveIt; reject, never slew',
                'limit_sources': self.limit_sources,
                'not_a_motor_rating_or_ACT_dynamic_validation': True,
            }
        else:
            self.measured_speed_contract = measured_speed_contract(velocities[:7])
            self.speed = np.minimum(velocities[:8], self.SPEED)
            self.speed[:7] = np.minimum(self.speed[:7], self.measured_speed_contract['command_limits_rad_s'])
            if self.command_gripper:
                self.measured_speed_contract['limits_rad_s'].append(min(.50, velocities[7]*self.measured_speed_contract['effective_planning_scale']))
                self.speed[-1] = min(self.SPEED, self.measured_speed_contract['source_command_cap_rad_s'],
                                     self.measured_speed_contract['limits_rad_s'][-1])
                self.measured_speed_contract['command_limits_rad_s'].append(float(self.speed[-1]))
            else:
                self.speed[-1] = 0.0
        cfg = bundle['home_config']
        self.start_contract = self.execution_profile.get('start_contract', 'configured_HOME')
        self.zero_contract = self.execution_profile.get('mechanical_zero_completion')
        if self.start_contract == 'mechanical_zero':
            self.home = self._vector([0.0] * 7 + [self.initial_command[7]], 8,
                                     'mechanical zero start')
            self.home_tolerance = np.r_[
                [float(self.zero_contract['arm_tolerance_rad'])] * 7, .03]
        else:
            self.home = self._vector([
                *cfg['right_home_joint_positions_rad'],
                cfg['home_gripper']['open_position_rad']], 8, 'configured HOME')
            # A commissioning profile can tighten the ROS acceptance, never widen it.
            self.home_tolerance = np.r_[
                [min(.03, float(cfg['final_acceptance']['joint_tolerance_rad']))] * 7,
                min(.03, float(cfg['home_gripper']['position_tolerance_rad']))]
        if not np.all(np.isfinite(self.home_tolerance)) or np.any(self.home_tolerance <= 0):
            raise ValueError('invalid configured HOME acceptance')
        if not self._start_pose_accepted(self.initial[:8]):
            raise ValueError('initial right state does not match '+self.start_contract+' acceptance')
        if not self._start_pose_accepted(self.command):
            raise ValueError('initial controller reference does not match '+self.start_contract+' acceptance')
        admission = np.full(8, self.TRACKING)
        if profile == 'visual32':
            from scripts.act_trial.closed_contract import closed_start
            closed_start(self.initial[7], self.command[7])
            admission[7] = .035
        if np.any(np.abs(self.command - self.initial[:8]) > admission + self._EPS):
            raise ValueError('initial controller reference exceeds measured tracking allowance')
        self._check_right_limits(self.initial[:8])
        self._check_right_limits(self.command)
        self.envelope_lower = (self.lower.copy() if self.DISPLACEMENT is None else
                               np.maximum(self.lower, self.initial[:8] - self.DISPLACEMENT))
        self.envelope_upper = (self.upper.copy() if self.DISPLACEMENT is None else
                               np.minimum(self.upper, self.initial[:8] + self.DISPLACEMENT))
        if self.command_gripper:
            self.envelope_lower[-1] = max(self.lower[-1], self.execution_profile['gripper_limits_rad'][0])
            self.envelope_upper[-1] = min(self.upper[-1], self.execution_profile['gripper_limits_rad'][1])
        else:
            self.envelope_lower[-1] = self.envelope_upper[-1] = self.command[-1]
        if np.any((self.command < self.envelope_lower - self._EPS)
                  | (self.command > self.envelope_upper + self._EPS)):
            raise ValueError('initial controller reference outside commissioning envelope')
        margin = np.array(self.tracking_contract['measured_envelope_margin_rad'])
        self.measured_lower = np.maximum(self.lower, self.envelope_lower - margin)
        self.measured_upper = np.minimum(self.upper, self.envelope_upper + margin)
        if self.collision_checks_enabled and self.scene.put(self.initial):
            raise ValueError('collision in initial full16 measured state')
        _, contacts = self._sweep(self.initial, np.r_[self.command, self.initial[8:]])
        if contacts:
            raise ValueError('collision in initial controller reference or measured-to-reference sweep')
        from scripts.act_trial.motion_generator import MODES, SynchronizedGenerator
        if motion_generator not in MODES:
            raise ValueError('unknown motion generator')
        self.motion_generator = motion_generator
        self.generator = (SynchronizedGenerator(self.speed, self.ACCELERATION,
                                               self.envelope_lower, self.envelope_upper,
                                               self.command_gripper)
                          if motion_generator == 'synchronized_v1' else None)

    @staticmethod
    def _vector(value, count, label):
        try:
            q = np.asarray(value, dtype=float).copy()
        except (TypeError, ValueError):
            raise ValueError(f'{label} must contain {count} finite numbers') from None
        if q.shape != (count,) or not np.all(np.isfinite(q)):
            raise ValueError(f'{label} must contain {count} finite numbers')
        return q

    @staticmethod
    def _time(now):
        now = float(now)
        if not np.isfinite(now) or now < 0:
            raise ValueError('invalid monotonic time')
        return now

    def _check_right_limits(self, q):
        if np.any((q < self.lower - self._EPS) | (q > self.upper + self._EPS)):
            raise ValueError('right measured state outside canonical ROS/MoveIt limits')

    def _start_pose_accepted(self, q):
        if self.start_contract != 'mechanical_zero':
            return bool(np.all(np.abs(q - self.home) <= self.home_tolerance + self._EPS))
        lo, hi = self.zero_contract['closed_gripper_range_rad']
        return bool(np.all(np.abs(q[:7]) <= self.zero_contract['arm_tolerance_rad'] + self._EPS)
                    and lo - self._EPS <= q[7] <= hi + self._EPS)

    def start(self, now):
        """Start once, requiring a recent successful checked holding tick."""
        now = self._time(now)
        if self.stage != 'holding':
            raise ValueError('guard is not holding; a latched trial cannot restart')
        if (self.last_hold_check_at is None or now < self.last_hold_check_at
                or now - self.last_hold_check_at > self.FEEDBACK_AGE):
            raise ValueError('a fresh successful holding tick is required before START')
        if not self._start_pose_accepted(self.previous_measured[:8]):
            raise ValueError(self.start_contract+' acceptance lost before START')
        self.started_at = now
        self.stage = 'active'

    def _latch(self, reason):
        if self.error is None:
            self.error = str(reason)
        self.stage = 'fault'
        self.velocity[:] = 0.0

    def _sweep(self, start, end):
        if not self.collision_checks_enabled:
            return 0, []
        steps = max(1, int(np.ceil(np.max(np.abs(end - start)) / self.COLLISION_STEP)))
        if steps > 1000:
            raise ValueError('collision sweep exceeds bounded computation budget')
        for i in range(steps + 1):
            pose = start + (end - start) * (i / steps)
            contacts = self.scene.put(pose)
            if contacts:
                return i + 1, contacts
        return steps + 1, []

    def _propose(self, target, dt, requested_velocity=None):
        if self.direct_stream:
            if not np.isfinite(dt) or not 0 < dt <= self.WATCHDOG:
                raise ValueError('invalid position-stream dispatch interval')
            candidate = self._vector(target, 8, 'position-stream target')
            velocity = self._vector(requested_velocity, 8, 'rolling trajectory velocity')
            if (np.any(np.abs(velocity) > self.speed + self._EPS)
                    or np.any(np.abs(candidate-self.command)/dt > self.speed+self._EPS)):
                raise ValueError('position-stream jump exceeds canonical URDF/MoveIt velocity limits')
            # q/qd were evaluated at the same timestamp by the rolling curve.
            # Never differentiate independently sampled IPC arrivals here.
            return candidate, velocity, velocity.copy(), dt
        # Cap delayed ticks at one nominal period: do not catch up with a jump.
        dt = min(dt, self.NOMINAL_PERIOD)
        if dt <= 0:
            raise ValueError('control tick did not advance')
        if self.generator is not None:
            return self.generator.propose(self.command, self.velocity, target, dt)
        desired_velocity = np.clip((target - self.command) / dt, -self.speed, self.speed)
        # Bound the next velocity so it can still brake before either envelope
        # edge. Solve v*dt + v**2/(2*a) <= remaining distance. Keeping this
        # constraint prevents clipping a moving command abruptly at an edge.
        a = self.ACCELERATION
        positive = np.sqrt((a * dt)**2 + 2*a*np.maximum(
            self.envelope_upper - self.command, 0)) - a*dt
        negative = np.sqrt((a * dt)**2 + 2*a*np.maximum(
            self.command - self.envelope_lower, 0)) - a*dt
        lo = np.maximum.reduce([self.velocity - a*dt, -self.speed, -negative])
        hi = np.minimum.reduce([self.velocity + a*dt, self.speed, positive])
        if np.any(lo > hi + self._EPS):
            raise ValueError('no feasible acceleration-bounded command inside envelope')
        next_velocity = np.clip(desired_velocity, lo, hi)
        candidate = self.command + next_velocity*dt
        if not self.command_gripper:
            candidate[-1] = self.initial_command[7]
            next_velocity[-1] = 0
        if np.any((candidate < self.envelope_lower - self._EPS)
                  | (candidate > self.envelope_upper + self._EPS)):
            raise ValueError('proposal escaped commissioning command envelope')
        return candidate, next_velocity, desired_velocity, dt

    def profile(self):
        return {
            'execution_profile': self.execution_profile,
            'duration_s': self.DURATION, 'maximum_arm_displacement_rad': self.DISPLACEMENT,
            'speed_rad_s': self.speed.tolist(), 'acceleration_rad_s2': self.ACCELERATION,
            'measured_speed_contract': self.measured_speed_contract,
            'initial_reference_tolerance_rad': self.TRACKING, 'feedback_age_s': self.FEEDBACK_AGE,
            'running_tracking_contract': self.tracking_contract,
            'watchdog_s': self.WATCHDOG, 'action_age_s': self.ACTION_AGE,
            'nominal_period_s': self.NOMINAL_PERIOD,
            'collision_step_rad': self.COLLISION_STEP,
            'collision_checks_enabled': self.collision_checks_enabled,
            'command_lower_rad': self.envelope_lower.tolist(),
            'command_upper_rad': self.envelope_upper.tolist(),
            'measured_lower_rad': self.measured_lower.tolist(),
            'measured_upper_rad': self.measured_upper.tolist(),
            'canonical_right_limits_rad': np.c_[self.lower, self.upper].tolist(),
            'home_tolerance_rad': self.home_tolerance.tolist(),
            'start_contract': self.start_contract,
            'mechanical_zero_completion': self.zero_contract,
            'limit_sources': self.limit_sources, 'limit_policy': self.limit_policy,
            'left_role': ('measured_collision_context_no_command' if self.collision_checks_enabled
                          else 'measured_only_no_command'),
            'gripper_policy': (('ACT_with_visual32_dataset_endpoints_and_tracking_diagnostics'
                                if self.execution_profile['name'] == 'visual32' else
                                'ACT_with_ROS_teaching_endpoints_and_tracking_diagnostics')
                               if self.command_gripper else 'hold_existing_controller_reference'),
            'initial_measured_rad': self.initial.tolist(),
            'initial_controller_reference_rad': self.initial_command.tolist(),
            'initial_reference_minus_measured_rad': (
                self.initial_command - self.initial[:8]).tolist(),
            'motion_generator': self.motion_generator,
            'smoothing': ('bridge evaluates one time-aligned C1 q/qd curve; no extra slew'
                          if self.direct_stream else
                          'caller blends/filters; synchronized acceleration-limited generator'
                          if self.generator is not None else
                          'caller blends/filters; guard applies acceleration and velocity slew'),
            'profile_basis': ('visual32 dataset geometry/endpoints inside canonical robot limits'
                              if self.execution_profile['name'] == 'visual32' else
                              'conservative commissioning restrictions, not widened robot limits'),
            'acceleration_basis': ('no added acceleration cap; physical behavior unvalidated'
                                   if self.direct_stream else
                                   '0.5 rad/s2 MoveIt cap times H2 acceleration scaling 0.5'),
            'stop_behavior': 'latched last command hold; not a dynamic braking guarantee',
        }

    def tick(self, requested_right8, measured16, now, feedback_age, action_age=None,
             requested_velocity=None):
        """Return a JSON-safe decision record; never execute any robot command."""
        wall_start = time.perf_counter()
        cpu_start = time.thread_time()
        desired = None if requested_right8 is None else preserve_nonfinite(
            np.asarray(requested_right8).tolist())
        record = {
            'desired': desired, 'clamped': None,
            'measured': preserve_nonfinite(np.asarray(measured16).tolist()),
            'feedback_age_s': preserve_nonfinite(feedback_age),
            'action_age_s': preserve_nonfinite(action_age),
            'interventions': [], 'errors': [], 'performance_warnings': [], 'collision_checked_samples': 0,
            'collision_contacts': [], 'left_command_emitted': False,
            'collision_checks_enabled': self.collision_checks_enabled,
            'joint_names': NAMES[:8], 'dispatch_performed': False,
        }
        checked_at = None
        try:
            checked_at = self._time(now)
            dt = (self.NOMINAL_PERIOD if self.last_tick is None
                  else checked_at - self.last_tick)
            if checked_at < self.created_at or dt <= 0 or dt > self.WATCHDOG:
                raise ValueError('control tick watchdog expired or non-monotonic time')
            self.last_tick = checked_at
            age = float(feedback_age)
            if not np.isfinite(age) or age < 0 or age > self.FEEDBACK_AGE:
                raise ValueError('feedback age exceeds commissioning freshness limit')
            measured = self._vector(measured16, 16, 'measured state')
            self._check_right_limits(measured[:8])
            if np.any((measured[:8] < self.measured_lower - self._EPS)
                      | (measured[:8] > self.measured_upper + self._EPS)):
                raise ValueError('measured right arm escaped commissioning envelope')
            tracking = measured[:8] - self.command
            record['tracking_error_rad'] = tracking.tolist()
            record['tracking_reference_rad'] = self.command.tolist()
            record['tracking_reference_kind'] = self.tracking_contract['reference']
            record['tracking_warning_thresholds_rad'] = self.tracking_warning.tolist()
            warned = np.flatnonzero(np.abs(tracking) > self.tracking_warning + self._EPS)
            record['tracking_warning_joints'] = [NAMES[i] for i in warned]
            if len(warned):
                record['performance_warnings'].append('tracking_error_above_diagnostic_reference')
            # Admission/holding retains its strict check. Once START occurred,
            # task arm and gripper tracking are diagnostic, including the terminal hold.
            key = 'holding_stop_limits_rad' if self.started_at is None else 'active_stop_limits_rad'
            stop_limits = self.tracking_contract[key]
            record['tracking_stop_limits_rad'] = list(stop_limits)
            exceeded = [i for i, limit in enumerate(stop_limits)
                        if limit is not None and abs(tracking[i]) > limit + self._EPS]
            record['tracking_stop_joints'] = [NAMES[i] for i in exceeded]
            if len(exceeded):
                details = '; '.join(f'{NAMES[i]} error={tracking[i]:+.6f}rad limit={stop_limits[i]:.6f}rad'
                                    for i in exceeded)
                raise ValueError('measured right command tracking error: '+details)
            n, contacts = self._sweep(self.previous_measured, measured)
            record['collision_checked_samples'] += n
            if contacts:
                record['collision_contacts'] = contacts
                raise ValueError('collision in measured full16 state sweep')
            self.previous_measured = measured.copy()
            if (self.stage == 'active' and self.DURATION is not None
                    and checked_at - self.started_at >= self.DURATION):
                self.stage = 'complete'
                self.velocity[:] = 0
                record['interventions'].append('duration_limit_hold')
            if self.stage == 'active':
                age = float(action_age)
                if not np.isfinite(age) or age < 0 or age > self.ACTION_AGE:
                    raise ValueError('ACT observation/action age exceeds existing 0.50 s limit')
                request = self._vector(requested_right8, 8, 'ACT execution request')
                target = np.clip(request, self.envelope_lower, self.envelope_upper)
                record['clamped'] = target.tolist()
                record['clipped_joint_indices'] = np.flatnonzero(
                    np.abs(request - target) > self._EPS).tolist()
                if np.any(np.abs(request[:7] - target[:7]) > self._EPS):
                    record['interventions'].append('command_envelope_clipping')
                if not self.command_gripper and abs(request[7] - self.initial_command[7]) > self._EPS:
                    record['interventions'].append('gripper_hold_existing_reference')
                if self.command_gripper and abs(request[7]-target[7]) > self._EPS:
                    record['interventions'].append('gripper_endpoint_clipping')
                paired_velocity = requested_velocity
                if self.direct_stream:
                    paired_velocity = self._vector(requested_velocity, 8, 'rolling trajectory velocity')
                    # Clipped coordinates are held at a constant endpoint, so
                    # they must not retain an outward feed-forward velocity.
                    clipped = np.abs(request-target) > self._EPS
                    paired_velocity[clipped] = 0.
                    paired_velocity[(target <= self.envelope_lower+self._EPS) & (paired_velocity < 0)] = 0.
                    paired_velocity[(target >= self.envelope_upper-self._EPS) & (paired_velocity > 0)] = 0.
                candidate, next_velocity, desired_velocity, used_dt = self._propose(target, dt, paired_velocity)
                record['motion_generator'] = self.motion_generator
                record['generated_proposal_rad'] = candidate.tolist()
                if self.generator is not None:
                    record['motion_generation'] = dict(self.generator.diagnostics)
                record['slew_period_s'] = used_dt
                record['desired_velocity_rad_s'] = desired_velocity.tolist()
                if np.any(np.abs((target-self.command)/used_dt) > self.speed+self._EPS):
                    record['interventions'].append('velocity_limit')
                if np.any(np.abs(next_velocity - desired_velocity) > self._EPS):
                    record['interventions'].append('acceleration_or_braking_limit')
                if not self.direct_stream and dt > self.NOMINAL_PERIOD + self._EPS:
                    record['interventions'].append('late_tick_no_catchup')
                # Check both the proposed command sweep and the path from
                # actual feedback. Full measured left context is never dropped.
                for start in (np.r_[self.command, measured[8:]], measured):
                    n, contacts = self._sweep(start, np.r_[candidate, measured[8:]])
                    record['collision_checked_samples'] += n
                    if contacts:
                        record['collision_contacts'] = contacts
                        raise ValueError('collision in right command proposal sweep')
                self.command = candidate
                self.velocity = next_velocity
            else:
                record['clamped'] = self.command.tolist()
                record['interventions'].append('inactive_hold')
                # Hold commands also need collision checking if the left arm
                # has moved since the previous full16 observation.
                n, contacts = self._sweep(measured, np.r_[self.command, measured[8:]])
                record['collision_checked_samples'] += n
                if contacts:
                    record['collision_contacts'] = contacts
                    raise ValueError('collision in full16 hold sweep')
                if self.stage == 'holding':
                    self.last_hold_check_at = checked_at
        except (ValueError, TypeError, RuntimeError) as exc:
            self._latch(exc)
        record.update({
            'now': preserve_nonfinite(now), 'stage': self.stage,
            'active_elapsed_s': None if self.started_at is None or checked_at is None
                else max(0.0, checked_at-self.started_at),
            'dispatched_proposal': self.command.tolist(),
            'proposal_velocity_rad_s': self.velocity.tolist(),
            'errors': [] if self.error is None else [self.error],
        })
        if self.stage == 'fault':
            record['interventions'].append('latched_fault_hold')
        record['timing'] = {
            'wall_s': time.perf_counter()-wall_start,
            'thread_cpu_s': time.thread_time()-cpu_start,
            'collision_backend': ('disabled' if not self.collision_checks_enabled else
                                  'mj_fwdPosition' if getattr(self.scene, 'position_only', False)
                                  else 'reference_or_test_scene'),
        }
        return record
