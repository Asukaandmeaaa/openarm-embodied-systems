"""Gate F readiness gate that refuses to fabricate human demonstrations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..paths import workspace_path


def inspect_human_dataset(root: Path | None = None) -> dict:
    root=root or workspace_path("data","pilot");episodes=root/"episodes";valid=[];rejections={}
    if episodes.is_dir():
        for episode in sorted(path for path in episodes.iterdir() if path.is_dir()):
            metadata_path=episode/"metadata.json"
            if not metadata_path.is_file():rejections[episode.name]="missing_metadata";continue
            metadata=json.loads(metadata_path.read_text())
            if metadata.get("source")!="human_virtual_handle":rejections[episode.name]="not_human_virtual_handle";continue
            if not metadata.get("strict_replay",{}).get("valid"):rejections[episode.name]="strict_replay_not_valid";continue
            valid.append(episode.name)
    count=len(valid)
    return {
        "gate":"F","status":"READY_FOR_MEMORIZATION" if count>=4 else "BLOCKED_USER_DATA",
        "valid_human_replay_episodes":valid,"valid_count":count,"required_for_memorization":4,
        "required_for_closed_loop":[30,50],"rejections":rejections,
        "act_training_executed":False,"closed_loop_executed":False,
        "reason":None if count>=4 else "Codex must not fabricate human virtual-handle demonstrations",
    }


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--dataset",type=Path);args=parser.parse_args();report=inspect_human_dataset(args.dataset)
    output=workspace_path("results","act_smoke_gate.json");output.write_text(json.dumps(report,indent=2)+"\n");print(json.dumps(report,indent=2))
    if report["status"]!="READY_FOR_MEMORIZATION":raise SystemExit(2)


if __name__=="__main__":main()
