"""Cooperative per-interface writer exclusion, shared with ROS hardware.

This module performs file I/O only. Persistent files must never be removed to
"release" ownership; the kernel releases flock when the final descriptor closes.
An old metadata record alone does not mean that a live process owns the lock.
"""

from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import re
import stat
import time


LOCK_ROOT = Path("/run/lock/openarm-can")


def reject_existing_receivers(interface, root, *, require_files=True):
    """Read procfs only; a registered receiver may also transmit CAN frames."""
    files = sorted(path for path in Path(root).iterdir() if path.name.startswith("rcvlist_"))
    if require_files and not files:
        raise RuntimeError("Cannot verify legacy CAN writers: receiver registry unavailable")
    for path in files:
        for line in path.read_text().splitlines():
            fields = line.split()
            if fields and fields[0] in (interface, "(any)", "any"):
                raise RuntimeError(f"Existing CAN receiver may be an unpatched writer on {interface}; "
                                   f"refusing hardware initialization. Stop legacy writers/recorders "
                                   f"before startup. Source: {path} {line.strip()}")


class CanOwnership:
    """Fail closed before a writer constructs a CAN socket or motor driver."""

    def __init__(self, interface: str, backend: str, *, _test_root=None):
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,15}", interface) or interface in (".", ".."):
            raise ValueError("Invalid CAN interface name for ownership lock")
        # Deliberately no environment or command-line production bypass.
        root = LOCK_ROOT if _test_root is None else Path(_test_root)
        root.mkdir(mode=0o755, exist_ok=True)
        directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW)
        self._fd = None
        try:
            info = os.fstat(directory)
            if (not stat.S_ISDIR(info.st_mode) or info.st_mode & 0o022
                    or info.st_uid not in (os.geteuid(), 0)):
                raise RuntimeError(f"Unsafe CAN ownership directory permissions/owner: {root}")
            self.path = root / f"{interface}.lock"
            self._fd = os.open(f"{interface}.lock", os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW,
                               0o644, dir_fd=directory)
        finally:
            os.close(directory)
        try:
            info = os.fstat(self._fd)
            if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_mode & 0o022
                    or info.st_uid not in (os.geteuid(), 0)):
                raise RuntimeError(f"Unsafe CAN ownership file: {self.path}")
            try:
                fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                owner = os.pread(self._fd, 4096, 0).decode(errors="replace")
                raise RuntimeError(f"CAN interface {interface} is already owned; refusing hardware initialization. "
                                   f"Owner: {owner or 'unavailable'}") from exc
            # A pre-patch ROS process has no lock. Check registrations after
            # locking but before the caller can create its SDK/CAN socket.
            # Tests inspect their supplied private root, never the real buses.
            reject_existing_receivers(interface, "/proc/net/can" if _test_root is None else root,
                                      require_files=_test_root is None)
            self.record = dict(schema=1, pid=os.getpid(), uid=os.geteuid(), backend=backend,
                               interface=interface, acquired_unix_ns=time.time_ns())
            payload = (json.dumps(self.record, sort_keys=True) + "\n").encode()
            os.ftruncate(self._fd, 0)
            if os.pwrite(self._fd, payload, 0) != len(payload):
                raise OSError("short ownership metadata write")
            os.fsync(self._fd)
        except BaseException:
            self.close()
            raise

    def close(self):
        """Release only after the corresponding writer has finished cleanup."""
        fd, self._fd = self._fd, None
        if fd is not None:
            # No LOCK_UN: a forked child's cleanup must not unlock its parent.
            os.close(fd)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        if getattr(self, "_fd", None) is not None:
            self.close()
