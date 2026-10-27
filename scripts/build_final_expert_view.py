#!/usr/bin/env python3
"""Build the final replay-valid expert view from a main run plus a top-up run.

Selects every strictly replay-valid episode from the main dataset, then adds
replay-valid episodes from the top-up dataset (renumbered past the main ids)
until the target size is reached. Produces a symlink view; nothing is copied
and no source dataset is modified.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import yaml
from tqdm.auto import tqdm


def load_valid_ids(report_path: Path, source: Path) -> list[str]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if Path(report["dataset"]).resolve() != source.resolve():
        raise SystemExit(f"replay report {report_path} was produced from a different dataset")
    return sorted(
        (str(item["episode"]) for item in report.get("episodes", []) if item.get("valid") is True),
        key=int,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main", type=Path, required=True)
    parser.add_argument(
        "--main-report",
        type=Path,
        help=(
            "strict replay report for --main; may be omitted only when the "
            "main dataset is itself a frozen final_view"
        ),
    )
    parser.add_argument("--topup", type=Path, required=True)
    parser.add_argument("--topup-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-size", type=int, default=100)
    args = parser.parse_args()

    if args.output.exists():
        parser.error(f"output already exists: {args.output}")

    main_metadata = yaml.safe_load((args.main / "metadata.yaml").read_text(encoding="utf-8"))
    topup_metadata = yaml.safe_load((args.topup / "metadata.yaml").read_text(encoding="utf-8"))
    if args.main_report is None:
        if "final_view" not in main_metadata:
            parser.error(
                "--main-report is required unless --main metadata contains final_view"
            )
        main_ids = sorted(
            (str(entry["id"]) for entry in main_metadata.get("episodes", [])),
            key=int,
        )
    else:
        main_ids = load_valid_ids(args.main_report, args.main)
    topup_ids = load_valid_ids(args.topup_report, args.topup)

    needed = args.target_size - len(main_ids)
    if needed < 0:
        parser.error(
            f"main run already has {len(main_ids)} valid episodes (> target {args.target_size})"
        )
    if len(topup_ids) < needed:
        parser.error(
            f"top-up has only {len(topup_ids)} valid episodes; need {needed} more "
            f"to reach target {args.target_size}"
        )
    selected_topup = topup_ids[:needed]

    main_entry = {str(e["id"]): e for e in main_metadata["episodes"]}
    topup_entry = {str(e["id"]): e for e in topup_metadata["episodes"]}
    next_id = max((int(i) for i in main_entry), default=-1) + 1

    combined_entries = [main_entry[eid] for eid in main_ids]
    topup_map = {}
    for eid in selected_topup:
        entry = dict(topup_entry[eid])
        entry["id"] = str(next_id)
        topup_map[eid] = str(next_id)
        combined_entries.append(entry)
        next_id += 1
    combined_entries.sort(key=lambda e: int(e["id"]))

    args.output.mkdir(parents=True)
    episodes_dir = args.output / "episodes"
    episodes_dir.mkdir()
    try:
        for eid in tqdm(
            main_ids,
            desc="Linking existing episodes",
            unit="episode",
            dynamic_ncols=True,
        ):
            source_episode = (args.main / "episodes" / eid).resolve()
            if not source_episode.is_dir():
                raise FileNotFoundError(source_episode)
            (episodes_dir / eid).symlink_to(source_episode, target_is_directory=True)
        for eid, new_id in tqdm(
            topup_map.items(),
            total=len(topup_map),
            desc="Linking new episodes",
            unit="episode",
            dynamic_ncols=True,
        ):
            source_episode = (args.topup / "episodes" / eid).resolve()
            if not source_episode.is_dir():
                raise FileNotFoundError(source_episode)
            (episodes_dir / new_id).symlink_to(source_episode, target_is_directory=True)

        combined_metadata = dict(main_metadata)
        combined_metadata["episodes"] = combined_entries
        combined_metadata["final_view"] = {
            "main_source": str(args.main.resolve()),
            "main_replay_report": (
                str(args.main_report.resolve()) if args.main_report else None
            ),
            "main_is_frozen_final_view": args.main_report is None,
            "topup_source": str(args.topup.resolve()),
            "topup_replay_report": str(args.topup_report.resolve()),
            "rejected_main_episode_ids": sorted(
                {str(e["id"]) for e in main_metadata["episodes"]} - set(main_ids), key=int
            ),
            "topup_original_to_final_id": topup_map,
            "unused_valid_topup_episode_ids": sorted(
                set(topup_ids) - set(selected_topup), key=int
            ),
        }
        (args.output / "metadata.yaml").write_text(
            yaml.safe_dump(combined_metadata, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
    except BaseException:
        shutil.rmtree(args.output, ignore_errors=True)
        raise

    print(json.dumps({
        "output": str(args.output.resolve()),
        "episodes": len(combined_entries),
        "from_main": len(main_ids),
        "from_topup": len(topup_map),
        "rejected_main_episode_ids": sorted(
            {str(e["id"]) for e in main_metadata["episodes"]} - set(main_ids), key=int
        ),
        "unused_valid_topup_episode_ids": sorted(set(topup_ids) - set(selected_topup), key=int),
    }, indent=2))


if __name__ == "__main__":
    main()
