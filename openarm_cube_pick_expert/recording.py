"""Crash-safe OpenArm Dataset writer.

Frames are written below ``.transactions``.  A successful episode is renamed
into ``episodes`` before metadata is atomically replaced. Failed/interrupted
attempts never become dataset episodes.
"""

from __future__ import annotations

import copy
import io
import os
from pathlib import Path
import shutil
import uuid

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
import yaml


class ExpertRecorder:
    VERSION = "0.4.0"

    def __init__(self, output: Path, metadata: dict, *, frame_shape=(600, 960, 3)):
        self.frame_shape = tuple(frame_shape)
        if len(self.frame_shape) != 3 or self.frame_shape[2] != 3 or min(self.frame_shape) <= 0:
            raise ValueError("frame_shape must be positive height,width,3")
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.transactions = self.output / ".transactions"
        self.transactions.mkdir(exist_ok=True)
        self.metadata_template = copy.deepcopy(metadata)
        self.metadata_path = self.output / "metadata.yaml"
        self.entries: list[dict] = []
        if self.metadata_path.exists():
            existing = yaml.safe_load(self.metadata_path.read_text(encoding="utf-8")) or {}
            self.entries = existing.get("episodes", [])
            episodes_dir = self.output / "episodes"
            has_episode_dirs = episodes_dir.exists() and any(episodes_dir.iterdir())
            if self.entries and not has_episode_dirs:
                quarantine = self.output / "orphaned"
                quarantine.mkdir(exist_ok=True)
                preserved = quarantine / f"pre-audit-invalid-metadata-{uuid.uuid4().hex}.yaml"
                os.replace(self.metadata_path, preserved)
                self.entries = []
                existing = None
            if existing is not None:
                for key, value in self.metadata_template.items():
                    if key in existing and existing[key] != value:
                        raise ValueError(f"existing dataset metadata mismatch for {key!r}")
        self.recover()
        self.cancel()

    def recover(self) -> list[str]:
        recovered = []
        for path in self.transactions.iterdir():
            if path.is_dir():
                recovered.append(path.name)
                shutil.rmtree(path)
        episode_ids = {str(e["id"]) for e in self.entries}
        disk_ids = {p.name for p in (self.output / "episodes").iterdir()} if (self.output / "episodes").exists() else set()
        missing_disk = episode_ids - disk_ids
        if missing_disk and not disk_ids:
            quarantine = self.output / "orphaned"
            quarantine.mkdir(exist_ok=True)
            preserved = quarantine / f"pre-audit-invalid-metadata-{uuid.uuid4().hex}.yaml"
            os.replace(self.metadata_path, preserved)
            self.entries = []
            episode_ids = set()
        orphan_disk = disk_ids - episode_ids
        if orphan_disk:
            quarantine = self.output / "orphaned"
            quarantine.mkdir(exist_ok=True)
            for episode_id in orphan_disk:
                os.replace(self.output / "episodes" / episode_id, quarantine / f"episode-{episode_id}-{uuid.uuid4().hex}")
            disk_ids -= orphan_disk
        if episode_ids != disk_ids:
            raise ValueError(
                f"dataset metadata/directory mismatch: metadata={sorted(episode_ids)} disk={sorted(disk_ids)}"
            )
        return recovered

    @property
    def next_episode_id(self) -> int:
        used = {int(e["id"]) for e in self.entries}
        value = 0
        while value in used:
            value += 1
        return value

    def start(self, attempt_id: str | int):
        if self.stage is not None:
            raise RuntimeError("an episode transaction is already active")
        self.episode_id = self.next_episode_id
        self.stage = self.transactions / f"attempt-{attempt_id}-{uuid.uuid4().hex}"
        self.stage.mkdir()
        self.buffers = {f"{kind}_{side}": ([], []) for kind in ("action", "obs") for side in ("right", "left")}
        return self.episode_id

    def sample(self, timestamp_ns: int, action, observation, frames: dict[str, np.ndarray]):
        if self.stage is None:
            raise RuntimeError("no active episode transaction")
        for kind, values in (("action", action), ("obs", observation)):
            value = np.asarray(values, dtype=np.float32)
            if value.shape != (16,):
                raise ValueError(f"{kind} must be float32[16]")
            for side, part in (("right", value[:8]), ("left", value[8:])):
                timestamps, samples = self.buffers[f"{kind}_{side}"]
                timestamps.append(int(timestamp_ns)); samples.append(part.copy())
        for name in ("ceiling", "wrist_right"):
            rgb = np.asarray(frames[name], dtype=np.uint8)
            if rgb.shape != self.frame_shape:
                raise ValueError(f"camera {name} must have shape {self.frame_shape}")
            directory = self.stage / "cameras" / name
            directory.mkdir(parents=True, exist_ok=True)
            stream = io.BytesIO()
            Image.fromarray(rgb, mode="RGB").save(stream, format="JPEG", quality=90)
            (directory / f"{timestamp_ns}.jpeg").write_bytes(stream.getvalue())

    def _write_streams(self) -> None:
        for key, (timestamps, values) in self.buffers.items():
            if len(values) < 2:
                raise ValueError(f"insufficient samples in {key}")
            kind, side = key.split("_")
            path = self.stage / kind / "arms" / side / "state.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            table = pa.table({
                "timestamp": pa.array(timestamps, type=pa.timestamp("ns")),
                "qpos": pa.array(values, type=pa.list_(pa.float32(), 8)),
            })
            pq.write_table(table, path)

    def finish(self, success: bool, details: dict):
        if not success:
            raise ValueError("only successful expert episodes may be committed")
        self._write_streams()
        destination = self.output / "episodes" / str(self.episode_id)
        destination.parent.mkdir(exist_ok=True)
        if destination.exists():
            raise FileExistsError(destination)
        os.replace(self.stage, destination)
        entry = {"id": str(self.episode_id), "success": True, "task_index": 0, **copy.deepcopy(details)}
        self.entries.append(entry)
        self._write_metadata_atomic()
        self.stage = None
        self.buffers = {}
        return self.episode_id

    def _write_metadata_atomic(self) -> None:
        metadata = copy.deepcopy(self.metadata_template)
        metadata.update(version=self.VERSION, episodes=self.entries)
        temporary = self.metadata_path.with_suffix(".yaml.tmp")
        temporary.write_text(yaml.safe_dump(metadata, sort_keys=False, allow_unicode=True), encoding="utf-8")
        os.replace(temporary, self.metadata_path)

    def cancel(self):
        stage = getattr(self, "stage", None)
        if stage is not None:
            shutil.rmtree(stage, ignore_errors=True)
        self.stage = None
        self.episode_id = None
        self.buffers = {}
