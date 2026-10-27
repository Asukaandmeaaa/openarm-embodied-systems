"""Strict-replay a committed human episode and atomically update its metadata."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from ..paths import workspace_path
from .strict import strict_replay


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def replay_human_episode(episode: Path) -> dict:
    metadata_path=episode/"metadata.json";metadata=json.loads(metadata_path.read_text())
    if metadata.get("source")!="human_virtual_handle":raise ValueError("episode is not labeled human_virtual_handle")
    observations=_jsonl(episode/"joint_observation.jsonl");actions_raw=_jsonl(episode/"demonstration_action.jsonl")
    state_times=np.asarray([row["timestamp_s"] for row in observations]);states=np.asarray([row["state"] for row in observations],dtype=np.float32)
    action_times=np.asarray([row["timestamp_s"] for row in actions_raw]);actions=np.asarray([row["action"] for row in actions_raw],dtype=np.float32)
    interpolated_states=np.column_stack([np.interp(action_times,state_times,states[:,i]) for i in range(16)]).astype(np.float32)
    raw_result=metadata.get("task_result",{})
    run=SimpleNamespace(
        times=action_times,states=interpolated_states,actions=actions,
        result=SimpleNamespace(evidence=tuple(raw_result.get("evidence",[])),failures=tuple(raw_result.get("failures",[]))),
    )
    report=strict_replay(run,input_origin="human_virtual_handle")
    right_p95=max(value["p95"] for name,value in report["per_joint_tracking_rad"].items() if name.startswith("right_joint"))
    report["valid"]=bool(report["full_sequence_success"] and not report["collision_differences"]["replay_counts"] and report["return_ready_error_rad"]<=0.03 and right_p95<=0.10)
    report["right_joint_tracking_p95_max_rad"]=right_p95
    metadata["strict_replay"]={"valid":report["valid"],"report":str(workspace_path("results","replay",f"{episode.name}.json"))}
    temporary=metadata_path.with_suffix(".json.tmp");temporary.write_text(json.dumps(metadata,indent=2)+"\n");os.replace(temporary,metadata_path)
    output=workspace_path("results","replay",f"{episode.name}.json");output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2)+"\n")
    clean_catalog=workspace_path("data","pilot","clean_catalog.json")
    valid=[]
    episodes=workspace_path("data","pilot","episodes")
    if episodes.is_dir():
        for path in episodes.iterdir():
            meta=path/"metadata.json"
            if meta.is_file() and json.loads(meta.read_text()).get("strict_replay",{}).get("valid"):valid.append(path.name)
    clean_catalog.write_text(json.dumps({"strict_replay_valid_episodes":sorted(valid)},indent=2)+"\n")
    return report


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("episode",type=Path);args=parser.parse_args();report=replay_human_episode(args.episode);print(json.dumps(report,indent=2))
    if not report["valid"]:raise SystemExit(2)


if __name__=="__main__":main()
