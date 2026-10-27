"""Dora client: predict 50, execute 10, then infer from a fresh observation."""
from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import tempfile
import time

import dora
import pyarrow as pa

from action_queue import ACTION_STEPS, SCHEDULE, InferenceGate, prefix
from logged_policy_server import AuditLog, chunk_sha256
from observation_timing import validate_timing


def checked_timing(metadata):
    return validate_timing(metadata, time.monotonic_ns(),
                           max_source_age_ns=int(float(os.getenv("MAX_SOURCE_AGE_MS", "200"))*1e6),
                           max_source_skew_ns=int(float(os.getenv("MAX_SOURCE_SKEW_MS", "100"))*1e6),
                           max_observation_age_ns=int(float(os.getenv("MAX_OBSERVATION_AGE_MS", "500"))*1e6))


def run(node, io, shared_dir, log, *, rolling=False):
    from scripts.act_trial.rolling_actions import full_chunk, QUERY_PERIOD_NS, SCHEDULE as ROLLING
    gate = InferenceGate()
    next_observation_ns = 0
    previous_observation_id = None
    sequence = 0
    files = []
    faulted = False
    for event in node:
        if event["type"] == "STOP":
            break
        if event["type"] != "INPUT":
            continue
        if event["id"] == "execution_stop":
            break
        if faulted:
            continue
        try:
            if event["id"] == "chunk_complete":
                if rolling:
                    raise ValueError('rolling inference must not receive prefix acknowledgements')
                gate.complete(event["metadata"], time.monotonic_ns())
                log.write("queue_completion_received", metadata=event["metadata"])
                continue
            if event["id"] != "observation":
                continue
            if rolling:
                if int(event.get('metadata', {}).get('observation_monotonic_ns', -1)) < next_observation_ns:
                    continue
            elif not gate.accepts(event.get("metadata", {})):
                continue
            timing = checked_timing(event["metadata"])
            request_started_ns = time.monotonic_ns()
            observation = event["value"]
            observation_id = observation.field("id")[0].as_py()
            reset = previous_observation_id is None or observation_id < previous_observation_id
            previous_observation_id = observation_id
            sequence += 1
            chunk_id = f"{log.run_id}_{sequence}"
            path = Path(shared_dir) / f"observation_{sequence}.arrow"
            batch = pa.RecordBatch.from_struct_array(observation)
            with pa.OSFile(str(path), "wb") as stream:
                with pa.ipc.new_file(stream, batch.schema) as writer:
                    writer.write(batch)
            files.append(path)
            if len(files) > 5:
                files.pop(0).unlink()
            request = {"name": "inference", "data_path": str(path), "reset": reset,
                       # Dora also supplies a datetime/HLC timestamp that is
                       # not JSON serializable. Preserve the validated source
                       # timing integers needed by the policy, without that
                       # transport-only metadata.
                       "metadata": dict(timing)}
            io.write(json.dumps(request) + "\n")
            io.flush()
            response = io.readline()
            if not response:
                raise ValueError("policy server disconnected")
            actions = json.loads(response)
            if actions.get("error"):
                raise ValueError(actions.get("error_detail") or actions["error"])
            positions = full_chunk(actions) if rolling else prefix(actions)
            checked_timing(timing)  # Original timestamp; inference cannot refresh it.
            metadata = {**timing, "interval": int(actions["interval"]), "reset": reset,
                        "cutoff_hz": actions.get("cutoff_hz", 15),
                        "action_schedule": ROLLING if rolling else SCHEDULE, "queue_chunk_id": chunk_id,
                        "queue_action_steps": len(positions),
                        "predicted_steps": len(actions["positions"]),
                        "policy_chunk_sha256": chunk_sha256(actions["positions"])}
            if rolling:
                # Never infer on an observation buffered before this inference
                # completed. Execution continues independently in the bridge.
                next_observation_ns = max(time.monotonic_ns(), request_started_ns+QUERY_PERIOD_NS)
            else:
                gate.dispatch(chunk_id)
            log.write("rolling_chunk_dispatched" if rolling else "queue_prefix_dispatched", observation_id=observation_id,
                      metadata=metadata, full_prefix_positions_rad=positions)
            node.send_output("actions", pa.array(positions.tolist(), type=pa.list_(pa.float32())), metadata)
        except (ValueError, KeyError, TypeError, OSError) as exc:
            faulted = True
            log.write("queue_client_fault", reason=str(exc))
            print(f"ACT_QUEUE_FAULT: {exc}", flush=True)
            node.send_output("timing_fault", pa.array([str(exc)]), {"fault": "action_queue"})


def main():
    log = AuditLog("queue_client")
    try:
        from scripts.act_trial.profiles import profile_from_config
        from scripts.act_trial.action_queue import contract
        config = json.loads((Path(os.environ['ACT_TRIAL_DIR'])/'trial_config.json').read_text())
        profile = profile_from_config(config)['name']
        if config.get('action_schedule') != contract(profile):
            raise ValueError('action schedule changed; prepare a new trial')
        rolling = profile == 'visual32'
        print('ACT_ACTION_SCHEDULE: '+json.dumps(contract(profile)), flush=True)
        with tempfile.TemporaryDirectory(prefix="act-queue-", dir="/dev/shm") as shared_dir:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
                sock.connect(os.environ["SOCKET"])
                with sock.makefile("rw") as io:
                    run(dora.Node(), io, shared_dir, log, rolling=rolling)
    finally:
        log.close()


if __name__ == "__main__":
    main()
