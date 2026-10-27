#!/usr/bin/env python3
"""Build a clean OpenArmDataset view containing only strict-replay-valid episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("replay_report", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--copy",
        action="store_true",
        help="copy episode directories instead of creating a lightweight symlink view",
    )
    args = parser.parse_args()

    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    metadata_path = args.source / "metadata.yaml"
    if not metadata_path.exists():
        parser.error(f"missing source metadata: {metadata_path}")

    metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8")) or {}
    report = json.loads(args.replay_report.read_text(encoding="utf-8"))
    if Path(report["dataset"]).resolve() != args.source.resolve():
        parser.error("replay report was produced from a different source dataset")

    valid_ids = {
        str(item["episode"])
        for item in report.get("episodes", [])
        if item.get("valid") is True
    }
    source_entries = metadata.get("episodes", [])
    selected = [entry for entry in source_entries if str(entry["id"]) in valid_ids]
    if not selected:
        parser.error("replay report contains no valid episodes")

    args.output.mkdir(parents=True)
    episodes_output = args.output / "episodes"
    episodes_output.mkdir()
    try:
        for entry in selected:
            episode_id = str(entry["id"])
            source_episode = (args.source / "episodes" / episode_id).resolve()
            destination = episodes_output / episode_id
            if not source_episode.is_dir():
                raise FileNotFoundError(source_episode)
            if args.copy:
                shutil.copytree(source_episode, destination)
            else:
                destination.symlink_to(source_episode, target_is_directory=True)

        clean_metadata = dict(metadata)
        clean_metadata["episodes"] = selected
        (args.output / "metadata.yaml").write_text(
            yaml.safe_dump(clean_metadata, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
    except BaseException:
        shutil.rmtree(args.output, ignore_errors=True)
        raise

    rejected = sorted({str(entry["id"]) for entry in source_entries} - valid_ids)
    print(
        json.dumps(
            {
                "source": str(args.source.resolve()),
                "output": str(args.output.resolve()),
                "mode": "copy" if args.copy else "symlink",
                "accepted_episode_ids": [str(entry["id"]) for entry in selected],
                "rejected_episode_ids": rejected,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
