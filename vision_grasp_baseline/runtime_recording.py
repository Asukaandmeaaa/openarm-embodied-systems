"""Bounded recorder I/O away from the ROS execution/spin thread.

No robot access. Errors and stalled health responses fail the execution caller;
events remain FIFO and a successful execution must drain them before finish.
"""
import queue
import threading
import time

from .collection_io import rpc


class RuntimeRecording:
    HEALTH_PERIOD_S=.25
    MAX_HEALTH_AGE_S=.75

    def __init__(self,call=None,clock=None):
        self.call=rpc if call is None else call
        self.clock=time.monotonic if clock is None else clock
        self.events=queue.Queue(maxsize=32)
        self.lock=threading.Lock();self.error=None;self.closed=False
        self.stop=threading.Event()
        # Establish health while holding, before action dispatch.
        self.call('status')
        self.last_health=self.clock()
        self.worker=threading.Thread(target=self._run,name='execution-recorder-io',daemon=True)
        self.worker.start()

    def _run(self):
        try:
            while not self.stop.is_set():
                if self.clock()-self.last_health>=self.HEALTH_PERIOD_S:
                    # Age the result from request dispatch, not response arrival.
                    sent=self.clock();self.call('status')
                    with self.lock:self.last_health=sent
                try:payload=self.events.get(timeout=.01)
                except queue.Empty:continue
                try:self.call('event',**payload)
                finally:self.events.task_done()
        except BaseException as exc:
            with self.lock:self.error=str(exc) or type(exc).__name__

    def check(self):
        with self.lock:
            error=self.error;age=self.clock()-self.last_health
        if error:raise RuntimeError('Runtime recorder failed: '+error)
        if age>self.MAX_HEALTH_AGE_S:
            raise RuntimeError(f'Runtime recorder health stalled: age_s={age:.3f}')

    def event(self,phase,**detail):
        self.check()
        if self.closed:raise RuntimeError('Runtime recorder is closed')
        # Preserve observation time even when HTTP delivery is delayed. Server
        # timestamp remains arrival time for compatibility with old readers.
        payload=dict(phase=phase,detail=dict(detail,source_monotonic_s=self.clock()))
        try:self.events.put_nowait(payload)
        except queue.Full:raise RuntimeError('Runtime recorder event queue full')

    def close(self,*,drain=False):
        if self.closed:return
        try:
            if drain:
                deadline=self.clock()+2.
                while self.events.unfinished_tasks:
                    self.check()
                    if self.clock()>deadline:raise RuntimeError('Runtime recorder event drain timed out')
                    self.stop.wait(.005)
                self.check()
        finally:
            self.closed=True;self.stop.set();self.worker.join(timeout=2.1)
            if self.worker.is_alive():raise RuntimeError('Runtime recorder I/O did not terminate')
