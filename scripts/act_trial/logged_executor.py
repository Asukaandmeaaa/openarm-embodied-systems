#!/usr/bin/env python3
"""Log ACT execution with the ten-step completion barrier.

Adds source-observation timing and chunk correlation to output metadata. This
is a Dora position-output node, never a CAN driver or motor activation path.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
import sys
import time

from logged_policy_server import AuditLog, ROOT, chunk_sha256, load_source, sha_file
from queued_executor import queued_async_main

ORIGINAL = ROOT / "official/dora-openarm-evaluation/nodes/dora-openarm-actions-executor/src/dora_openarm_actions_executor/main.py"


class LoggedQueue(asyncio.Queue):
    def __init__(self, log, context):
        super().__init__()
        self.log = log
        self.context = context


    async def get(self):
        event = await super().get()
        self.context.clear()
        metadata = dict(event.get("metadata", {}))
        self.context.update(metadata=metadata, input_id=event.get("id"))
        if event.get("id") != "timing_fault":
            positions = event["value"].to_pylist()
            self.context["trial_chunk_sha256"] = chunk_sha256(positions)
            self.log.write("executor_raw_chunk", **self.context, full_positions_rad=positions)
        else:
            self.log.write("executor_input_fault", **self.context, fault=event["value"].to_pylist())
        return event


class LoggedNode:
    def __init__(self, node, log, context):
        self.node = node
        self.log = log
        self.context = context
        self.bridge_stopped = False

    def __getattr__(self, name):
        return getattr(self.node, name)

    def next(self):
        event = self.node.next()
        if event.get('type') == 'INPUT' and event.get('id') == 'execution_stop':
            self.bridge_stopped = True
            self.log.write('executor_stopped_by_bridge', reason=event['value'].to_pylist())
            # The original Dora receive loop cancels the executor on STOP.
            # Never enqueue this lifecycle event as an action chunk.
            return {'type': 'STOP'}
        return event

    def send_output(self, output_id, value, metadata=None):
        if self.bridge_stopped:
            return  # Latched: a stale chunk can never resume after bridge stop.
        enriched = dict(metadata or {})
        source = self.context.get("metadata", {})
        for key in ("observation_monotonic_ns", "source_age_ns", "source_skew_ns", "reset"):
            if key in source:
                enriched[key] = source[key]
        if "trial_chunk_sha256" in self.context:
            enriched["trial_chunk_sha256"] = self.context["trial_chunk_sha256"]
        payload = value.to_pylist()
        # Keep the full original payload and type, including other-arm context.
        self.log.write("executor_requested_output", output_id=output_id, metadata=enriched,
                       payload=payload, payload_sha256=chunk_sha256(payload),
                       arrow_type=str(value.type), requested_monotonic_ns=time.monotonic_ns())
        return self.node.send_output(output_id, value, enriched)


def instrument_classes(original, log, context):
    original_upsampler = original.HermiteUpsampler
    original_lowpass = original.BiquadLowpass

    class LoggedHermite(original_upsampler):
        def upsample(self, y_chunk, t_eval):
            result = super().upsample(y_chunk, t_eval)
            log.write("executor_blended_and_hermite", **context,
                      blended_positions_rad=y_chunk, blended_sha256=chunk_sha256(y_chunk),
                      evaluation_times_s=t_eval, hermite_positions_rad=result,
                      hermite_sha256=chunk_sha256(result))
            return result

    class LoggedLowpass(original_lowpass):
        def reset_state(self, initial_x):
            result = super().reset_state(initial_x)
            log.write("executor_filter_reset", **context, initial_position=initial_x)
            return result

        def step(self, x):
            result = super().step(x)
            log.write("executor_filter_step", **context, input_position=x,
                      filtered_position=result, filtered_sha256=chunk_sha256(result))
            return result

    original.HermiteUpsampler = LoggedHermite
    original.BiquadLowpass = LoggedLowpass


def logged_async_main(original, log, context):
    async def run(arms, use_upsample, use_filter, control_hz):
        log.write("executor_settings", arms=arms, use_upsample=use_upsample,
                  use_filter=use_filter, control_hz=control_hz)
        node = LoggedNode(original.dora.Node(), log, context)
        events = LoggedQueue(log, context)
        executor_task = asyncio.create_task(original._main_executor(node, events, arms, use_upsample, use_filter, control_hz))
        dora_task = asyncio.create_task(original._main_dora(node, events, executor_task))
        try:
            try:
                await executor_task
            except asyncio.CancelledError:
                pass
            await dora_task
        finally:
            # Cleanup on a logging failure cannot leave the counterpart task
            # running; no control algorithm or successful-path timing changed.
            for task in (executor_task, dora_task):
                if not task.done():
                    task.cancel()
            await asyncio.gather(executor_task, dora_task, return_exceptions=True)
    return run


def main():
    log = AuditLog("executor")
    try:
        original = load_source(ORIGINAL, "_act_trial_original_executor")
        log.write("executor_wrapper_started", argv=sys.argv,
                  source_sha256={str(ORIGINAL): sha_file(ORIGINAL), str(Path(__file__)): sha_file(__file__),
                                 str(Path(__file__).with_name("logged_policy_server.py")): sha_file(Path(__file__).with_name("logged_policy_server.py")),
                                 str(ORIGINAL.parent.parent / "observation_timing.py"): sha_file(ORIGINAL.parent.parent / "observation_timing.py")},
                  algorithms="queued_prefix_v1: execute first 10; fresh inference after completion; original Hermite and biquad; no old-tail blend",
                  timing_note="logging adds overhead; no freshness limit widened or observation timestamp refreshed")
        context = {}
        instrument_classes(original, log, context)
        original._main_async = queued_async_main(original, log, context, LoggedNode, LoggedQueue)
        original.main()
    except BaseException as exc:
        log.write("executor_wrapper_exit", exception_type=type(exc).__name__, detail=str(exc))
        raise
    finally:
        log.close()


if __name__ == "__main__":
    main()
