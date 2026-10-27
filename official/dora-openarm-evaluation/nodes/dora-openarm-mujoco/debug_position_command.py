#!/usr/bin/env python3
"""Send right joint1=current+0.2 rad once, using the dora input/output path."""

import time

import dora
import numpy as np
import pyarrow as pa


def main() -> None:
    node = dora.Node()
    sent = False
    verified = False
    target = None
    sent_at = None
    for event in node:
        if event["type"] != "INPUT":
            continue
        if event["id"] == "arm_right_observation" and not sent:
            current = np.asarray(event["value"], dtype=np.float32)
            target = current.copy()
            target[0] += 0.2
            print(
                f"[DBG SENDER] before={current.tolist()} target={target.tolist()}",
                flush=True,
            )
            node.send_output("position_right", pa.array(target, type=pa.float32()))
            sent = True
            sent_at = time.monotonic()
        elif event["id"] == "tick" and sent:
            if not verified and time.monotonic() - sent_at >= 0.5:
                # Re-send after one second so the node logs qpos after the
                # background physics loop has had time to converge.
                node.send_output("position_right", pa.array(target, type=pa.float32()))
                verified = True
            elif verified:
                return


if __name__ == "__main__":
    main()
