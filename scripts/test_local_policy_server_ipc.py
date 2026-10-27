#!/usr/bin/env python3
"""Exercise the local ACT server's Arrow IPC socket with one synthetic observation."""

import argparse
import json
import socket
import tempfile
import time
from pathlib import Path

import numpy as np
import pyarrow as pa


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("socket", type=Path)
    args = parser.parse_args()
    height, width = 360, 640
    ceiling = np.zeros((height, width, 3), dtype=np.uint8)
    wrist = np.full((height, width, 3), 64, dtype=np.uint8)
    batch = pa.RecordBatch.from_arrays(
        [
            pa.array([[0.0] * 16], type=pa.list_(pa.float32())),
            pa.array([ceiling.reshape(-1).tolist()], type=pa.list_(pa.uint8())),
            pa.array([wrist.reshape(-1).tolist()], type=pa.list_(pa.uint8())),
        ],
        ["position", "camera_ceiling", "camera_wrist_right"],
    )
    path = Path(tempfile.mktemp(suffix=".arrow", dir="/dev/shm"))
    try:
        with pa.OSFile(str(path), "wb") as output:
            with pa.ipc.new_file(output, batch.schema) as writer:
                writer.write(batch)
        request = {
            "name": "inference",
            "data_path": str(path),
            "reset": True,
            "metadata": {
                "timestamp": time.time_ns(),
                "observation_monotonic_ns": time.monotonic_ns(),
                "source_age_ns": 10_000_000,
                "source_skew_ns": 5_000_000,
            },
        }
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.connect(str(args.socket))
            with connection.makefile("rw") as io:
                io.write(json.dumps(request) + "\n")
                io.flush()
                response = json.loads(io.readline())
        positions = np.asarray(response["positions"], dtype=np.float32)
        if positions.shape != (50, 16) or not np.isfinite(positions).all():
            raise RuntimeError(f"unexpected action response shape: {positions.shape}")
        if response["interval"] != 33_333_333 or response["cutoff_hz"] != 15:
            raise RuntimeError(f"unexpected timing metadata: {response}")
        print(
            json.dumps(
                {
                    "positions_shape": list(positions.shape),
                    "finite": True,
                    "interval_ns": response["interval"],
                    "cutoff_hz": response["cutoff_hz"],
                },
                indent=2,
            )
        )
    finally:
        path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
