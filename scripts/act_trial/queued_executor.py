"""Execute an entire ten-step prefix before acknowledging the next inference."""
from __future__ import annotations

import asyncio
import time

import numpy as np
import pyarrow as pa

from action_queue import ACTION_STEPS, INTERVAL_NS, SCHEDULE, sample_times


def queued_async_main(original, log, context, node_class, queue_class):
    async def run(arms, use_upsample, use_filter, control_hz):
        if arms != ["right"] or not use_upsample:
            raise ValueError("queued ACT trial requires right-arm 100Hz interpolation")
        log.write("executor_settings", arms=arms, use_upsample=use_upsample,
                  use_filter=use_filter, control_hz=control_hz, action_schedule=SCHEDULE,
                  executed_steps=ACTION_STEPS, old_tail_blend=False)
        node = node_class(original.dora.Node(), log, context)
        events = queue_class(log, context)
        running = False
        faulted = False
        lowpass = None

        def fault(reason):
            nonlocal faulted
            if not faulted:
                faulted = True
                log.write("queue_executor_fault", reason=str(reason))
                original._emit_timing_fault(node, reason)

        async def execute():
            nonlocal running, lowpass
            while not faulted:
                event = await events.get()
                running = True
                metadata = event["metadata"]
                try:
                    timing = original._validate_action_timing(metadata)
                    positions = np.asarray(event["value"].to_pylist(), dtype=np.float32)
                    if (positions.shape != (ACTION_STEPS, 16) or not np.isfinite(positions).all()
                            or metadata.get("action_schedule") != SCHEDULE
                            or int(metadata["queue_action_steps"]) != ACTION_STEPS
                            or int(metadata["interval"]) != INTERVAL_NS
                            or not metadata.get("queue_chunk_id")):
                        raise ValueError("invalid queued ACT prefix")
                    times, evaluation, duration = sample_times(control_hz)
                    upsampler = original.HermiteUpsampler(1e9 / INTERVAL_NS,
                                                        (ACTION_STEPS - 1) * INTERVAL_NS / 1e9)
                    values = upsampler.upsample(positions, evaluation)
                    if use_filter and lowpass is None:
                        lowpass = original.BiquadLowpass(control_hz, metadata.get("cutoff_hz", 15))
                    if lowpass is not None and metadata.get("reset"):
                        lowpass.reset_state(positions[0])
                    started_ns = time.monotonic_ns()
                    deadline = original.observation_deadline_ns(
                        timing, original._timing_limit_ns("MAX_OBSERVATION_AGE_MS", 500))
                    if started_ns + round(duration * 1e9) >= deadline:
                        raise ValueError("observation budget cannot cover the ten-step prefix")
                    log.write("queue_prefix_started", **context, planned_raw_steps=ACTION_STEPS,
                              started_monotonic_ns=started_ns, duration_s=duration)
                    for offset_s, position in zip(times, values):
                        remaining = (started_ns + round(offset_s * 1e9) - time.monotonic_ns()) / 1e9
                        if remaining > 0:
                            await asyncio.sleep(remaining)
                        if time.monotonic_ns() >= deadline:
                            raise ValueError("running action chunk exceeded observation age limit")
                        if faulted or node.bridge_stopped:
                            return
                        if lowpass is not None:
                            position = lowpass.step(position)
                        node.send_output("move_position_right", pa.array(position[:8], type=pa.float32()),
                                         {"timestamp": time.time_ns(), "action_schedule": SCHEDULE,
                                          "queue_chunk_id": metadata["queue_chunk_id"],
                                          "queue_raw_step": min(ACTION_STEPS - 1, int(round(offset_s * 1e9) // INTERVAL_NS))})
                    # The last action occupies its full slot, not just its first instant.
                    remaining = (started_ns + round(duration * 1e9) - time.monotonic_ns()) / 1e9
                    if remaining > 0:
                        await asyncio.sleep(remaining)
                    if time.monotonic_ns() >= deadline:
                        raise ValueError("running action chunk exceeded observation age limit")
                    if faulted or node.bridge_stopped:
                        return
                    running = False
                    completed = {"queue_chunk_id": metadata["queue_chunk_id"],
                                 "executed_raw_steps": ACTION_STEPS,
                                 "completed_monotonic_ns": time.monotonic_ns(),
                                 "action_schedule": SCHEDULE}
                    log.write("queue_prefix_completed", **completed,
                              elapsed_s=(completed["completed_monotonic_ns"] - started_ns) / 1e9)
                    print(f"[ACT QUEUE] completed {ACTION_STEPS}/{ACTION_STEPS} raw steps; next fresh observation", flush=True)
                    node.send_output("chunk_complete", pa.array([metadata["queue_chunk_id"]]), completed)
                except (ValueError, KeyError, TypeError) as exc:
                    fault(exc)
                    return

        async def receive(executor_task):
            while not faulted:
                if node.is_empty():
                    await asyncio.sleep(.001)
                    continue
                event = node.next()
                if event["type"] != "INPUT":
                    break
                if event["id"] == "timing_fault":
                    fault(event["value"][0].as_py())
                    break
                if event["id"] != "actions":
                    continue
                if running or not events.empty():
                    fault("new ACT prefix arrived before completion of the previous ten steps")
                    break
                await events.put(event)
            executor_task.cancel()

        executor_task = asyncio.create_task(execute())
        receiver_task = asyncio.create_task(receive(executor_task))
        try:
            done, _ = await asyncio.wait((executor_task, receiver_task), return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                if not task.cancelled():
                    task.result()
        finally:
            for task in (executor_task, receiver_task):
                task.cancel()
            await asyncio.gather(executor_task, receiver_task, return_exceptions=True)
    return run
