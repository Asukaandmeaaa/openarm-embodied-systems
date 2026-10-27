import threading
import time

import pytest

from vision_grasp_baseline.runtime_recording import RuntimeRecording


def wait_until(predicate,timeout=1.):
    deadline=time.monotonic()+timeout
    while not predicate():
        assert time.monotonic()<deadline
        time.sleep(.002)


def test_slow_http_does_not_block_execution_and_events_keep_source_time():
    entered=threading.Event();release=threading.Event();calls=[]
    def call(operation,**payload):
        if operation=='event':
            entered.set();assert release.wait(1.)
        calls.append((operation,payload))
    io=RuntimeRecording(call=call)
    try:
        io.event('timeline_event',name='reach',observed_time_s=.01)
        assert entered.wait(1.)
        source=time.monotonic();checks=0
        # Simulated ROS/collision polling keeps running while HTTP takes 320ms.
        while time.monotonic()-source<.32:
            before=time.monotonic();io.check()
            assert time.monotonic()-before<.05
            checks+=1;time.sleep(.005)
        release.set();io.close(drain=True)
        event=next(payload for op,payload in calls if op=='event')
        assert checks>30
        assert event['detail']['source_monotonic_s']<=source
        assert event['detail']['observed_time_s']==.01
    finally:release.set();io.close()


def test_recorder_failure_is_not_silently_ignored():
    def call(operation,**payload):
        if operation=='event':raise OSError('disk or HTTP error')
    io=RuntimeRecording(call=call)
    try:
        io.event('timeline_event')
        wait_until(lambda:io.error is not None)
        with pytest.raises(RuntimeError,match='disk or HTTP error'):io.check()
        with pytest.raises(RuntimeError,match='disk or HTTP error'):io.close(drain=True)
    finally:io.close()


def test_stalled_worker_and_queue_overflow_fail_closed():
    entered=threading.Event();release=threading.Event();clock=[0.]
    def call(operation,**payload):
        if operation=='event':entered.set();assert release.wait(1.)
    io=RuntimeRecording(call=call,clock=lambda:clock[0])
    try:
        io.event('first');assert entered.wait(1.)
        for i in range(32):io.event('queued',index=i)
        with pytest.raises(RuntimeError,match='queue full'):io.event('overflow')
        clock[0]=.751
        with pytest.raises(RuntimeError,match='health stalled'):io.check()
    finally:release.set();io.close()


def test_success_requires_fifo_events_to_be_drained():
    calls=[]
    io=RuntimeRecording(call=lambda op,**kw:calls.append((op,kw)))
    for i in range(9):io.event('timeline_event',index=i)
    io.close(drain=True)
    assert [kw['detail']['index'] for op,kw in calls if op=='event']==list(range(9))
    assert not io.worker.is_alive()


def test_initial_health_failure_happens_before_worker_start():
    def fail(*args,**kwargs):raise RuntimeError('recorder unavailable')
    with pytest.raises(RuntimeError,match='unavailable'):RuntimeRecording(call=fail)
