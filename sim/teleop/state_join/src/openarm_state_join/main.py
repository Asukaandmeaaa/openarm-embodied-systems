"""Dora node joining right[8] + left[8] arm observations."""

import dora
import numpy as np
import pyarrow as pa


def join_state(right: np.ndarray, left: np.ndarray) -> np.ndarray:
    """Return the contract-ordered float32 state."""
    right = np.asarray(right, dtype=np.float32)
    left = np.asarray(left, dtype=np.float32)
    if right.shape != (8,) or left.shape != (8,):
        raise ValueError(f"expected right[8] and left[8], got {right.shape}, {left.shape}")
    return np.concatenate((right, left)).astype(np.float32, copy=False)


def main() -> None:
    """Run the Dora join node."""
    node = dora.Node()
    latest = {}
    for event in node:
        if event["type"] != "INPUT" or event["id"] not in ("arm_right", "arm_left"):
            continue
        latest[event["id"]] = np.asarray(event["value"], dtype=np.float32)
        if len(latest) == 2:
            state = join_state(latest["arm_right"], latest["arm_left"])
            node.send_output("state", pa.array(state, type=pa.float32()))


if __name__ == "__main__":
    main()
