# Copyright 2026 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Evaluation session metadata: checkpoint info, task list, per-episode results."""

from __future__ import annotations

import datetime
import pathlib

import yaml


class EvalMetadata:
    """Tracks one evaluation session: checkpoint, tasks, and per-episode results.

    Saved as ``eval_metadata.yaml`` alongside the dataset directory. Separate from
    the dataset's own ``metadata.yaml`` so it can carry eval-specific fields (notes,
    capture timestamp) without polluting the dataset schema.
    """

    VERSION = "0.1.0"

    def __init__(self) -> None:
        """Initialize with empty data."""
        self.data: dict = {"version": self.VERSION}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, path: pathlib.Path) -> EvalMetadata:
        """Load from YAML file."""
        instance = cls()
        with open(path) as f:
            instance.data = yaml.safe_load(f) or {}
        return instance

    def save(self, path: pathlib.Path) -> None:
        """Atomically save to YAML file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".yaml.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            yaml.dump(self.data, f, default_flow_style=False, allow_unicode=True)
        tmp.replace(path)

    # ------------------------------------------------------------------
    # Checkpoint auto-info
    # ------------------------------------------------------------------

    def auto_populate(self, checkpoint_path: str | None) -> None:
        """Populate checkpoint info with its path and a capture timestamp.

        Only fills fields that are not already present so that repeated calls
        on resume do not overwrite the original captured values.
        """
        checkpoint = self.data.setdefault("checkpoint", {})
        if checkpoint_path and "path" not in checkpoint:
            checkpoint["path"] = checkpoint_path
        auto_info = checkpoint.setdefault("auto_info", {})
        if "captured_at" not in auto_info:
            auto_info["captured_at"] = datetime.datetime.now().isoformat(
                timespec="seconds"
            )

    # ------------------------------------------------------------------
    # Episode recording
    # ------------------------------------------------------------------

    def add_episode(
        self,
        *,
        id: str,
        task_index: int,
        success: bool,
        note: str,
    ) -> None:
        """Append a completed episode record."""
        self.data.setdefault("episodes", []).append(
            {
                "id": id,
                "task_index": task_index,
                "success": success,
                "note": note,
                "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            }
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def episodes(self) -> list[dict]:
        """All recorded episodes."""
        return self.data.get("episodes", [])

    @property
    def tasks(self) -> list[dict]:
        """Task list (prompt, description, ...)."""
        return self.data.get("tasks", [])

    @tasks.setter
    def tasks(self, value: list[dict]) -> None:
        self.data["tasks"] = value

    @property
    def checkpoint_path(self) -> str:
        """Filesystem path to the model checkpoint."""
        return self.data.get("checkpoint", {}).get("path", "")

    @property
    def checkpoint_description(self) -> str:
        """Human-readable description of the checkpoint."""
        return self.data.get("checkpoint", {}).get("description", "")

    @checkpoint_description.setter
    def checkpoint_description(self, value: str) -> None:
        self.data.setdefault("checkpoint", {})["description"] = value

    @property
    def auto_info(self) -> dict:
        """Auto-captured checkpoint info (capture timestamp)."""
        return self.data.get("checkpoint", {}).get("auto_info", {})
