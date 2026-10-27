"""Bounded pre-dispatch controller query; never infer active state on timeout."""
import math
import time

import rclpy
from controller_manager_msgs.srv import ListControllers


def verify_required_controllers(node,names,attempts=3,timeout_s=3.,total_timeout_s=12.):
    if (attempts < 1 or not all(math.isfinite(t) and t > 0 for t in (timeout_s,total_timeout_s))):
        raise ValueError('Positive bounded controller query limits required')
    started=time.monotonic();deadline=started+total_timeout_s
    errors=[]
    diagnostic=dict(passed=False,queries=0,client_recreations=0,errors=errors,
                    response_timeout_s=timeout_s,total_timeout_s=total_timeout_s)
    node.controller_readiness_diagnostics=diagnostic
    def check_stop():
        check=getattr(node,'execution_table_guard_failure',None)
        if callable(check):
            reason=check()
            if reason:raise RuntimeError(reason)
    for attempt in range(attempts):
        check_stop()
        remaining=deadline-time.monotonic()
        if remaining<=0:break
        client=node.controller_client
        if not client.wait_for_service(timeout_sec=min(timeout_s,remaining)):
            errors.append('controller-manager service unavailable')
        else:
            check_stop()
            remaining=deadline-time.monotonic()
            if remaining<=0:break
            future=client.call_async(ListControllers.Request())
            diagnostic['queries']+=1
            response_deadline=min(deadline,time.monotonic()+timeout_s)
            rclpy.spin_until_future_complete(node,future,timeout_sec=max(0.,response_deadline-time.monotonic()))
            response=None
            failure='controller-manager response timed out'
            if time.monotonic()<=response_deadline and future.done() and not future.cancelled():
                try:response=future.result()
                except Exception as exc:failure='controller-manager query failed: '+str(exc)
                else:
                    if response is None:failure='empty controller-manager response'
            if response is not None:
                check_stop()
                controllers={item.name:item for item in response.controller}
                invalid={name:controllers[name].state if name in controllers else 'missing' for name in names
                         if name not in controllers or controllers[name].state!='active'}
                if invalid:
                    diagnostic.update(invalid_controllers=invalid,elapsed_s=time.monotonic()-started)
                    raise RuntimeError('Required controllers are not active: '+str(invalid))
                diagnostic.update(passed=True,elapsed_s=time.monotonic()-started)
                return dict(attempts=attempt+1,controllers={name:controllers[name].state for name in names},
                            diagnostics=diagnostic)
            # Discard this request so a late response cannot satisfy a retry.
            client.remove_pending_request(future)
            future.cancel()
            errors.append(failure)
        diagnostic['elapsed_s']=time.monotonic()-started
        if attempt+1<attempts and time.monotonic()<deadline:
            # Rebuild only this read-only client, never restart a controller
            # or replace an action client with unresolved motion goals.
            if not node.destroy_client(client):
                raise RuntimeError('Could not retire controller-manager query client; no trajectory dispatched')
            node.controller_client=node.create_client(ListControllers,'/controller_manager/list_controllers')
            diagnostic['client_recreations']+=1
            print(f'控制器状态查询暂未响应，重建只读查询客户端（{attempt+2}/{attempts}）；尚未发送轨迹。',flush=True)
    diagnostic['elapsed_s']=time.monotonic()-started
    raise RuntimeError('Controller readiness unavailable after '+str(diagnostic['queries'])+
                       ' queries within '+str(total_timeout_s)+'s: '+str(errors)+'; no trajectory dispatched')
