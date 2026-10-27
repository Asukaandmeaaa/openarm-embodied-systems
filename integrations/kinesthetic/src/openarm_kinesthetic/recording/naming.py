"""Stable identifiers shared by a taught motion and its captured replay."""

from __future__ import annotations

from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import re
import time


TRIAL_ID_PATTERN = re.compile(r"^[0-9]{8}_[0-9]{6}_[0-9]{3}$")


def valid_trial_id(value: str) -> bool:
    return bool(TRIAL_ID_PATTERN.fullmatch(value))


def allocate_trial_id(collection_root: Path, *, now: float | None = None) -> str:
    """Reserve a daily sequence without reusing numbers after failed trials."""
    root = Path(collection_root).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    moment = time.time() if now is None else float(now)
    day = datetime.fromtimestamp(moment).strftime("%Y%m%d")
    clock = datetime.fromtimestamp(moment).strftime("%H%M%S")
    path = root / ".trial_counter.json"
    with path.open("a+", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        stream.seek(0)
        try:
            counters = json.load(stream)
        except json.JSONDecodeError:
            counters = {}
        if not isinstance(counters, dict):
            raise ValueError("trial counter file must contain an object")
        sequence = int(counters.get(day, 0)) + 1
        if sequence > 999:
            raise ValueError(f"daily trial sequence exhausted for {day}")
        counters[day] = sequence
        stream.seek(0)
        stream.truncate()
        json.dump(counters, stream, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    return f"{day}_{clock}_{sequence:03d}"
