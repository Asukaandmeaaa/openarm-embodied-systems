"""Pure visual32 episode-completion detector; no hardware or ROS imports."""

from __future__ import annotations

import numpy as np


class MechanicalZeroCompletion:
    """Complete only after a real departure and a settled return to zero."""

    def __init__(self, contract):
        self.contract = dict(contract)
        self.departed_at = None
        self.returned_since = None

    @staticmethod
    def _vector(value, label):
        result = np.asarray(value, dtype=float)
        if result.shape != (16,) or not np.all(np.isfinite(result)):
            raise ValueError(f'{label} must contain sixteen finite values')
        return result

    def update(self, measured16, velocity16, now, started_at, command8=None):
        q = self._vector(measured16, 'complete measured state')
        velocity = self._vector(velocity16, 'complete measured velocity')
        now = float(now)
        started_at = float(started_at)
        if not np.isfinite(now) or not np.isfinite(started_at) or now < started_at:
            raise ValueError('invalid completion timing')

        c = self.contract
        arm_axes = [*range(7), *range(8, 15)]
        arm_max = float(np.max(np.abs(q[arm_axes])))
        right_arm_max = float(np.max(np.abs(q[:7])))
        speed_max = float(np.max(np.abs(velocity[[*range(8), *range(8, 15)]])))
        gripper = float(q[7])
        closed = 'closed_gripper_range_rad' in c
        departed = (right_arm_max >= float(c['departure_arm_rad'])
                    or (gripper <= float(c['departure_gripper_rad']) if closed
                        else gripper >= float(c['departure_gripper_rad'])))
        if self.departed_at is None and departed:
            self.departed_at = now

        lo, hi = map(float, c['closed_gripper_range_rad'] if closed else c['open_gripper_range_rad'])
        at_zero = arm_max <= float(c['arm_tolerance_rad']) and lo <= gripper <= hi
        command_at_zero = True
        if closed:
            cmd = np.asarray(command8, dtype=float)
            command_at_zero = (cmd.shape == (8,) and np.isfinite(cmd).all()
                               and np.max(np.abs(cmd[:7])) <= float(c['arm_tolerance_rad'])
                               and abs(cmd[7]-float(c['closed_gripper_command_rad'])) <= .005)
            at_zero = at_zero and command_at_zero
        settled = speed_max <= float(c['return_velocity_rad_s'])
        eligible = (self.departed_at is not None
                    and now - started_at >= float(c['minimum_active_s']))
        if eligible and at_zero and settled:
            if self.returned_since is None:
                self.returned_since = now
        else:
            self.returned_since = None
        complete = (self.returned_since is not None
                    and now - self.returned_since >= float(c['return_settle_s']))
        return {
            'complete': complete,
            'departed': self.departed_at is not None,
            'departure_monotonic_s': self.departed_at,
            'returned_since_monotonic_s': self.returned_since,
            'arm_max_abs_rad': arm_max,
            'right_arm_max_abs_rad': right_arm_max,
            'gripper_rad': gripper,
            'controlled_and_parked_max_abs_velocity_rad_s': speed_max,
            'at_mechanical_zero': at_zero,
            'command_at_zero': bool(command_at_zero),
            'settled': settled,
            'active_elapsed_s': now - started_at,
            'contract': self.contract,
        }
