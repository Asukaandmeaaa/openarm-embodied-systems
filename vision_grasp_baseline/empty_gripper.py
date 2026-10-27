"""Measured empty-finger acceptance; never sends a motor/trajectory command."""
import math
import time

from .gripper_settings import (require_closed_transit,TRANSIT_STABLE_S,
                               TRANSIT_GUARD_MARGIN_S)


class StableEmptyClosure:
    def __init__(self):
        self.first_stamp=None
        self.last_stamp=None

    def update(self,measured,state_stamp,now):
        fresh=(state_stamp is not None and math.isfinite(state_stamp)
               and 0<=now-state_stamp<=.1)
        try:require_closed_transit(float(measured))
        except ValueError:fresh=False
        if not fresh:
            self.first_stamp=None
            self.last_stamp=None
            return False
        if (self.first_stamp is None or self.last_stamp is None
                or not 0<=state_stamp-self.last_stamp<=.1):
            self.first_stamp=state_stamp
        self.last_stamp=state_stamp
        return state_stamp-self.first_stamp>=TRANSIT_STABLE_S


class ReturnClosureGuard:
    def __init__(self,plan):
        events={e['name']:e['time_s'] for e in plan['events']}
        self.start=events['transit_close']
        self.return_time=events['return_closed']
        self.deadline=self.return_time-TRANSIT_GUARD_MARGIN_S
        self.stability=StableEmptyClosure()
        self.confirmed=False

    def check(self,elapsed,measured,state_stamp,now):
        if elapsed<self.start:return
        stable=self.stability.update(measured,state_stamp,now)
        if elapsed>=self.deadline:
            if not stable:
                raise RuntimeError('Empty gripper not stably closed before return; cancel both trajectories '
                                   f'(actual={measured}, return_time_s={self.return_time:.3f})')
            self.confirmed=True


def verify_stationary_empty_gripper(node,arm_reference,timeout=2.):
    """After zero-test's closing action: keep its target, observe, or fail."""
    import rclpy
    from .grasp import FINGER
    stability=StableEmptyClosure()
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        rclpy.spin_once(node,timeout_sec=.01)
        failure=node.execution_table_guard_failure()
        if failure:raise RuntimeError(failure)
        if node.passive_state is not False:raise RuntimeError('Rigid mode lost during empty closure acceptance')
        if max(abs(a-b) for a,b in zip(node.current_positions,arm_reference))>.015:
            raise RuntimeError('Arm moved during stationary empty-gripper acceptance')
        if node.full_joints is None:continue
        values=dict(zip(node.full_joints.name,node.full_joints.position))
        measured=float(values[FINGER])
        if stability.update(measured,node.full_joint_state_monotonic_s,time.monotonic()):
            return dict(passed=True,actual_gripper_rad=measured,stable_s=TRANSIT_STABLE_S)
    raise RuntimeError('Empty gripper did not stably close at zero; no collection started')
