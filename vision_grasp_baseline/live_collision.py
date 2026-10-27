"""Check measured robot geometry asynchronously during complete playback.

Tracking distance is diagnostic, not a substitute for collision geometry.
This is sampled checking (50 Hz target), not continuous collision certification.
"""
import time
import xml.etree.ElementTree as ET
import numpy as np
from moveit_msgs.srv import GetStateValidity

RUNTIME_TIMEOUT_S=.2
STARTUP_TIMEOUT_S=10.


class LiveCollisionMonitor:
    def __init__(self,node,model):
        self.node=node;self.pending=None;self.sent=0.;self.next_check=0.
        self.checked=0;self.max_response_s=0.;self.support_contact=False
        self.phase='reach';self.attached=None;self.pickup=None
        self.prepared=False;self.startup_probe_count=0;self.startup_elapsed_s=0.
        self.startup_max_response_s=0.;self.runtime_timeouts=0
        self.last_check=None;self.max_check_gap_s=0.;self.last_query=None;self.failure=None
        from .hover import NAMES
        from .grasp import FINGER
        xml=ET.parse(model).getroot()
        bounds=[xml.find(f"joint[@name='{name}']/limit") for name in [*NAMES,FINGER]]
        self.lower=np.array([float(b.get('lower')) for b in bounds])
        self.upper=np.array([float(b.get('upper')) for b in bounds])

    def measured_state(self):
        values=self.node.telemetry.snapshot()['positions_rad']
        actual=np.asarray(values[:8],float)
        if actual.shape!=(8,) or not np.isfinite(actual).all() or np.any(actual<self.lower) or np.any(actual>self.upper):
            raise RuntimeError('Measured joints outside URDF hard position limits')
        return values[:7],values[7]

    @staticmethod
    def require_valid(response):
        if response is None or not response.valid:
            contacts=[] if response is None else [(c.contact_body_1,c.contact_body_2) for c in response.contacts]
            raise RuntimeError('Actual robot state violates collision/model bounds: '+str(contacts))

    def prepare(self,timeout=STARTUP_TIMEOUT_S,*,clock=None,spin=None):
        """Discover and warm this exact client BEFORE recording/goal dispatch.

        Cold authenticated DDS discovery is not a runtime RPC. Once connected,
        require two consecutive valid replies within the unchanged 200ms budget.
        A stalled service fails here while the robot is still holding.
        """
        if self.pending is not None:raise RuntimeError('Cannot prepare with a pending runtime collision query')
        if clock is None:clock=time.monotonic
        if spin is None:
            import rclpy
            spin=lambda:rclpy.spin_once(self.node,timeout_sec=.005)
        started=clock();future=None;sent=None;fast=0
        self.prepared=False
        print('碰撞监控启动检查：等待服务发现及有效快速响应；尚未录制或下发抓取轨迹。',flush=True)
        try:
            while clock()-started<timeout:
                failure=self.node.execution_table_guard_failure()
                if failure:raise RuntimeError(failure)
                q,finger=self.measured_state()
                if self.node.valid_client.service_is_ready():
                    if future is None:
                        request=GetStateValidity.Request();request.group_name='right_arm'
                        request.robot_state=self.node.state(q,finger,self.attached)
                        sent=clock();future=self.node.valid_client.call_async(request)
                    if future.done():
                        age=clock()-sent
                        self.require_valid(future.result())
                        self.startup_probe_count+=1
                        self.startup_max_response_s=max(self.startup_max_response_s,age)
                        fast=fast+1 if age<=RUNTIME_TIMEOUT_S else 0
                        future=None
                        if fast>=2:
                            self.prepared=True;self.next_check=0.
                            print('碰撞监控启动检查通过；运行中仍使用200ms超时保护。',flush=True)
                            return
                spin()
            raise RuntimeError('Collision service startup discovery/warmup timed out; no trajectory dispatched')
        finally:
            self.startup_elapsed_s=clock()-started
            if future is not None and not future.done():self.discard(future)

    def discard(self,future):
        # cancel() alone leaves an rclpy client pending-request entry behind.
        self.node.valid_client.remove_pending_request(future)
        future.cancel()

    def accept(self,response):
        try:self.require_valid(response)
        except RuntimeError as exc:
            self.failure=dict(error=str(exc),query=self.last_query)
            raise

    def transition(self,phase,attached=None,pickup=None):
        # A query submitted for the previous attachment phase must not be
        # interpreted using the new phase's contact semantics.
        if self.pending is not None:
            # Do not erase an already received collision at a phase boundary.
            if self.pending.done():self.accept(self.pending.result())
            self.discard(self.pending)
        self.pending=None;self.next_check=0.;self.phase=phase
        self.attached=attached;self.pickup=pickup
        self.support_contact=(phase=='carry')
        self.node.finger_contacts(phase in ('grasp','carry'),support_contact=self.support_contact)

    def check(self,now=None):
        if not self.prepared:raise RuntimeError('Live collision monitor must be prepared before motion')
        supplied_now=now
        now=time.monotonic() if now is None else now
        gap=0. if self.last_check is None else now-self.last_check
        self.max_check_gap_s=max(self.max_check_gap_s,gap);self.last_check=now
        # Hard bounds and fresh telemetry must still be checked even while a
        # service response is pending, not only when the next query is sent.
        q,finger=self.measured_state()
        if self.pending is not None:
            if now-self.sent>RUNTIME_TIMEOUT_S:
                self.runtime_timeouts+=1
                self.failure=dict(error='collision_response_timeout',query=self.last_query,
                                  response_done=self.pending.done(),check_gap_s=gap,age_s=now-self.sent)
                raise RuntimeError(f'Live measured-state collision check timed out: '
                                   f'age_s={now-self.sent:.3f}, limit_s={RUNTIME_TIMEOUT_S}, '
                                   f'phase={self.phase}, completed_checks={self.checked}, prepared={self.prepared}, '
                                   f'check_gap_s={gap:.3f}, response_done={self.pending.done()}')
            if not self.pending.done():
                return
            response=self.pending.result();self.pending=None
            self.max_response_s=max(self.max_response_s,now-self.sent);self.checked+=1
            self.accept(response)
        if now<self.next_check:return
        if self.phase=='carry' and self.support_contact:
            from .hover import NAMES
            tcp=self.node.chain.pose(dict(zip(NAMES,q)))
            if tcp[2,3]>=self.pickup[2,3]+.025 or np.linalg.norm(tcp[:2,3]-self.pickup[:2,3])>=.02:
                self.node.finger_contacts(True,support_contact=False);self.support_contact=False
                q,finger=self.measured_state()
        request=GetStateValidity.Request();request.group_name='right_arm'
        request.robot_state=self.node.state(q,finger,self.attached)
        # Contact-scene updates above can block. Start the RPC timer at actual
        # dispatch, not at entry to check(), and use refreshed measured state.
        self.sent=time.monotonic() if supplied_now is None else now
        self.last_query=dict(phase=self.phase,sent_monotonic_s=self.sent,
                            positions_rad=[float(x) for x in q]+[float(finger)])
        self.pending=self.node.valid_client.call_async(request);self.next_check=self.sent+.02

    def close(self):
        if self.pending is not None:
            self.discard(self.pending);self.pending=None

    def report(self):
        return dict(checked_states=self.checked,target_hz=50,max_response_s=self.max_response_s,
                    startup_prepared=self.prepared,startup_probe_count=self.startup_probe_count,
                    startup_elapsed_s=self.startup_elapsed_s,startup_max_response_s=self.startup_max_response_s,
                    runtime_timeout_s=RUNTIME_TIMEOUT_S,runtime_timeouts=self.runtime_timeouts,
                    max_check_gap_s=self.max_check_gap_s,last_query=self.last_query,failure=self.failure,
                    tracking_error_policy='diagnostic_only',actual_geometry_check=True)
