from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/filter_replay_validated_openarm.py"


def test_filter_keeps_only_replay_valid_episodes(tmp_path: Path) -> None:
    source = tmp_path / "raw"
    for episode_id in ("0", "1"):
        episode = source / "episodes" / episode_id
        episode.mkdir(parents=True)
        (episode / "sentinel").write_text(episode_id, encoding="utf-8")
    (source / "metadata.yaml").write_text(
        yaml.safe_dump(
            {
                "version": "0.4.0",
                "tasks": [{"prompt": "pick"}],
                "episodes": [
                    {"id": "0", "success": True},
                    {"id": "1", "success": True},
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    report = tmp_path / "replay.json"
    report.write_text(
        json.dumps(
            {
                "dataset": str(source.resolve()),
                "episodes": [
                    {"episode": 0, "valid": True},
                    {"episode": 1, "valid": False},
                ],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "clean"

    subprocess.run(
        [sys.executable, str(SCRIPT), str(source), str(report), str(output)],
        check=True,
        text=True,
        capture_output=True,
    )

    metadata = yaml.safe_load((output / "metadata.yaml").read_text(encoding="utf-8"))
    assert [entry["id"] for entry in metadata["episodes"]] == ["0"]
    assert (output / "episodes/0").is_symlink()
    assert (output / "episodes/0/sentinel").read_text(encoding="utf-8") == "0"
    assert not (output / "episodes/1").exists()

