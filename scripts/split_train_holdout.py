#!/usr/bin/env python3
"""Split a raw OpenArm dataset into deterministic train/holdout symlink views."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import yaml
from tqdm.auto import tqdm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("train_output", type=Path)
    parser.add_argument("holdout_output", type=Path)
    parser.add_argument("--holdout-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--replay-report", type=Path, default=None,
                        help="restrict to episodes marked valid in this replay report")
    args = parser.parse_args()

    for path in (args.train_output, args.holdout_output):
        if path.exists():
            parser.error(f"output already exists: {path}")

    metadata = yaml.safe_load((args.source / "metadata.yaml").read_text(encoding="utf-8"))
    entries = metadata.get("episodes", [])
    episode_ids = [str(entry["id"]) for entry in entries]

    if args.replay_report is not None:
        report = json.loads(args.replay_report.read_text(encoding="utf-8"))
        if Path(report["dataset"]).resolve() != args.source.resolve():
            parser.error("replay report was produced from a different source dataset")
        valid_ids = {str(item["episode"]) for item in report.get("episodes", [])
                     if item.get("valid") is True}
        missing = sorted(set(episode_ids) - valid_ids, key=int)
        if missing:
            parser.error(f"episodes failed strict replay (cannot split): {missing}")
        episode_ids = [eid for eid in episode_ids if eid in valid_ids]

    if len(episode_ids) <= args.holdout_size:
        parser.error(f"only {len(episode_ids)} episodes; need more than holdout size")

    shuffled = sorted(episode_ids, key=int)
    random.Random(args.seed).shuffle(shuffled)
    holdout_ids = set(shuffled[: args.holdout_size])
    train_ids = [eid for eid in sorted(episode_ids, key=int) if eid not in holdout_ids]
    holdout_ids = sorted(holdout_ids, key=int)

    entry_by_id = {str(entry["id"]): entry for entry in entries}
    for output, selected in ((args.train_output, train_ids), (args.holdout_output, holdout_ids)):
        output.mkdir(parents=True)
        episodes_dir = output / "episodes"
        episodes_dir.mkdir()
        try:
            role = "train" if output == args.train_output else "holdout"
            for eid in tqdm(
                selected,
                desc=f"Creating {role} split",
                unit="episode",
                dynamic_ncols=True,
            ):
                source_episode = (args.source / "episodes" / eid).resolve()
                if not source_episode.is_dir():
                    raise FileNotFoundError(source_episode)
                (episodes_dir / eid).symlink_to(source_episode, target_is_directory=True)
            split_metadata = dict(metadata)
            split_metadata["episodes"] = [entry_by_id[eid] for eid in selected]
            split_metadata["split"] = {
                "source": str(args.source.resolve()),
                "split_seed": args.seed,
                "role": role,
            }
            (output / "metadata.yaml").write_text(
                yaml.safe_dump(split_metadata, sort_keys=False, allow_unicode=True),
                encoding="utf-8",
            )
        except BaseException:
            shutil.rmtree(output, ignore_errors=True)
            raise

    print(json.dumps({
        "source": str(args.source.resolve()),
        "train_output": str(args.train_output.resolve()),
        "holdout_output": str(args.holdout_output.resolve()),
        "train_episode_ids": train_ids,
        "holdout_episode_ids": holdout_ids,
        "split_seed": args.seed,
    }, indent=2))


if __name__ == "__main__":
    main()
