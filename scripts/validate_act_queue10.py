#!/usr/bin/env python3
"""Run the real Dora client/executor against synthetic observations and policy.

No ROS node, CAN socket, camera, checkpoint or hardware driver is created.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import tempfile
import threading
import time

import yaml

ROOT = Path(__file__).resolve().parents[1]


def observer():
    import dora
    import pyarrow as pa
    node = dora.Node()
    index = 0
    for event in node:
        if event["type"] == "STOP":
            break
        if event["type"] == "INPUT":
            index += 1
            node.send_output("observation", pa.array([{"id": index, "position": [0.] * 16}]),
                             {"observation_monotonic_ns": time.monotonic_ns(),
                              "source_age_ns": 0, "source_skew_ns": 0})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observer", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.observer:
        observer()
        return
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    trial = output / "act"
    trial.mkdir()
    errors = []
    requests = []
    with tempfile.TemporaryDirectory(prefix="act-q10-", dir="/dev/shm") as directory:
        address = str(Path(directory) / "policy.sock")
        with socket.socket(socket.AF_UNIX) as server:
            server.bind(address)
            server.listen(1)
            def peer():
                try:
                    with server.accept()[0] as sock:
                        with sock.makefile("rw") as io:
                            for line in io:
                                request = json.loads(line)
                                requests.append(request)
                                response = {"interval": 33_333_333, "cutoff_hz": 15,
                                            "positions": [[.001 * (i + 50*len(requests))] * 16 for i in range(50)]}
                                io.write(json.dumps(response) + "\n")
                                io.flush()
                except Exception as exc:
                    errors.append(repr(exc))
            thread = threading.Thread(target=peer, daemon=True)
            thread.start()
            python = str(ROOT / "official/dora-openarm-evaluation/.venv/bin/python")
            # Dora canonicalizes executable symlinks; a shell wrapper retains
            # the venv interpreter path and its site-packages.
            wrapper = output / "python_node.sh"
            wrapper.write_text('#!/usr/bin/env bash\nexec ' + shlex.quote(python) + ' "$@"\n')
            wrapper.chmod(0o755)
            graph = {"nodes": [
                {"id": "synthetic-observer", "path": str(wrapper),
                 "args": f"{Path(__file__).resolve()} --observer",
                 "inputs": {"tick": "dora/timer/millis/33"}, "outputs": ["observation"]},
                {"id": "policy-server", "path": str(wrapper),
                 "args": str(ROOT / "scripts/act_trial/queued_policy_client.py"),
                 "inputs": {"observation": "synthetic-observer/observation",
                            "chunk_complete": "actions-executor/chunk_complete"},
                 "outputs": ["actions", "timing_fault"]},
                {"id": "actions-executor", "path": str(wrapper),
                 "args": f"{ROOT}/scripts/act_trial/logged_executor.py --arms right --upsample --filter --control-hz 100",
                 "inputs": {"actions": "policy-server/actions", "timing_fault": "policy-server/timing_fault"},
                 "outputs": ["move_position_right", "timing_fault", "chunk_complete"]}]}
            graph_path = output / "offline_graph.yaml"
            graph_path.write_text(yaml.safe_dump(graph, sort_keys=False))
            env = {**os.environ, "ACT_TRIAL_DIR": str(trial), "SOCKET": address,
                   "PYTHONPATH": str(ROOT / "official/dora-openarm-evaluation/src"),
                   "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
            with (output / "dora.log").open("w") as stream:
                process = subprocess.run([str(ROOT / "official/dora-openarm-evaluation/.venv/bin/dora"),
                                          "run", str(graph_path), "--stop-after", "4s"],
                                         env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=20)
            thread.join(timeout=2)
    rows = [json.loads(line) for path in trial.glob("*.jsonl") for line in path.read_text().splitlines()]
    completed = {row["queue_chunk_id"]: row for row in rows if row["event"] == "queue_prefix_completed"}
    faults = [row for row in rows if row["event"] in ("queue_executor_fault", "queue_client_fault")]
    dispatches = [row for row in rows if row["event"] == "queue_prefix_dispatched"]
    raw_inputs = {row["metadata"]["queue_chunk_id"]: row for row in rows if row["event"] == "executor_raw_chunk"}
    checks = []
    previous_completion = 0
    for dispatch in sorted(dispatches, key=lambda row: row["monotonic_ns"]):
        meta = dispatch["metadata"]
        identifier = meta["queue_chunk_id"]
        if identifier not in completed:
            continue  # The timer can stop an in-progress final prefix.
        outputs = [row for row in rows if row["event"] == "executor_requested_output"
                   and row["output_id"] == "move_position_right"
                   and row["metadata"].get("queue_chunk_id") == identifier]
        completion = completed[identifier]
        checks.append({"queue_chunk_id": identifier, "output_count": len(outputs),
                       "raw_steps_seen": sorted({row["metadata"]["queue_raw_step"] for row in outputs}),
                       "elapsed_s": completion["elapsed_s"],
                       "fresh_after_previous_completion": meta["observation_monotonic_ns"] >= previous_completion,
                       "original_observation_timestamp_preserved": all(
                           row["metadata"]["observation_monotonic_ns"] == meta["observation_monotonic_ns"] for row in outputs),
                       "executed_only_ten_predictions": len(raw_inputs[identifier]["full_positions_rad"]) == 10
                       and len(dispatch["full_prefix_positions_rad"]) == 10})
        previous_completion = completion["completed_monotonic_ns"]
    passed = (process.returncode == 0 and not errors and not faults and len(checks) >= 3
              and all(row["output_count"] == 34 and row["raw_steps_seen"] == list(range(10))
                      and row["elapsed_s"] >= .33333333 and row["fresh_after_previous_completion"]
                      and row["original_observation_timestamp_preserved"]
                      and row["executed_only_ten_predictions"] for row in checks))
    report = {"passed": passed, "hardware_io": False, "real_model_inference": False,
              "dora_returncode": process.returncode, "peer_errors": errors, "faults": faults,
              "completed_prefixes": len(checks), "prefixes": checks,
              "scope": "Real Dora processes and socket, synthetic input and predictions; no physical outcome claim"}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
