#!/usr/bin/env python3
"""CPU-only scene/phase comparison from stored ACT observations and TRAIN/VAL.

Never initializes ROS, Dora, cameras, sockets or robot control. No training.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/lerobot/act_mixed_20261016"
ROLES = ("ceiling", "wrist_right")


def sustained_event(values, closed=True, start=0):
    mask = values > -.225 if closed else values < -.30
    for i in range(start, len(mask) - 5):
        if mask[i:i+6].all():
            return i
    return None


def trial_inputs(trial):
    started = json.loads((trial / "started.json").read_text())["monotonic_ns"]
    result = json.loads((trial / "result.json").read_text())
    end = result["monotonic_ns"]
    predictions = []
    for path in trial.glob("policy_*.jsonl"):
        for line in path.open():
            row = json.loads(line)
            if row["event"] == "raw_policy_response" and started <= row["monotonic_ns"] <= end:
                predictions.append(row)
    predictions.sort(key=lambda row: row["monotonic_ns"])
    selected = [predictions[0], min(predictions, key=lambda row: abs(row["monotonic_ns"]-(end-2_000_000_000)))]
    samples = []
    for label, row in zip(("home", "stall"), selected):
        path = trial / Path(row["observation_arrow_path"]).name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["observation_arrow_sha256"]
        with pa.memory_map(str(path), "r") as stream:
            obs = pa.ipc.open_file(stream).get_batch(0).to_struct_array()[0]
        images = {role: np.asarray(obs["camera_"+role].as_py(), dtype=np.uint8).reshape(360, 640, 3)
                  for role in ROLES}
        samples.append({"label": label, "state": np.asarray(obs["position"].as_py(), dtype=np.float32),
                        "images": images, "row": row, "path": str(path),
                        "time_s": (row["monotonic_ns"]-started)/1e9})
    summary = {"trial": str(trial), "duration_s": (end-started)/1e9, "stop_reason": result["reason"],
               "predictions": len(predictions),
               "first10_contains_close": sum(any(a[7] > -.225 for a in row["full_postprocessed_actions"][:10]) for row in predictions),
               "future50_contains_close": sum(any(a[7] > -.225 for a in row["full_postprocessed_actions"]) for row in predictions)}
    pre = json.loads((trial / "offline_preflight.json").read_text())
    summary["checkpoint"] = pre["checkpoint"]["directory"]
    summary["weight_sha256"] = next(value for path, value in pre["checkpoint"]["file_sha256"].items()
                                   if path.endswith("/model.safetensors"))
    return samples, summary


def numeric_episodes(split):
    root = DATA / split
    manifest = json.loads((root / "meta/mixed_manifest.json").read_text())
    eligible = {row["episode_index"]: row for row in manifest["episodes"] if row["domain"] != "simulation"}
    episodes = defaultdict(list)
    for path in sorted((root / "data").rglob("*.parquet")):
        ids = np.asarray(pq.read_table(path, columns=["episode_index"])["episode_index"])
        if not np.isin(ids, list(eligible)).any():
            continue
        table = pq.read_table(path, columns=["episode_index", "frame_index", "observation.state", "action"])
        for index, row in enumerate(table.to_pylist()):
            if row["episode_index"] in eligible:
                episodes[row["episode_index"]].append((row["frame_index"], path, index, row))
    result = []
    for index, rows in sorted(episodes.items()):
        rows.sort()
        q = np.array([row[3]["observation.state"] for row in rows], dtype=np.float32)
        actions = np.array([row[3]["action"] for row in rows], dtype=np.float32)
        close = sustained_event(actions[:, 7])
        release = None if close is None else sustained_event(actions[:, 7], False, close+15)
        assert close is not None and release is not None
        result.append({"meta": eligible[index], "rows": rows, "state": q, "action": actions,
                       "close": close, "release": release, "split": split})
    return result


def demo_sample(episode, frame):
    _, path, file_index, _ = episode["rows"][frame]
    images = {}
    table = pq.read_table(path, columns=["observation.images."+role for role in ROLES])
    for role in ROLES:
        value = table["observation.images."+role][file_index].as_py()
        images[role] = np.asarray(Image.open(io.BytesIO(value["bytes"])).convert("RGB")).copy()
    return {"state": episode["state"][frame], "images": images,
            "episode_id": episode["meta"]["episode_id"], "domain": episode["meta"]["domain"],
            "frame": frame, "truth": episode["action"][frame:frame+50]}


def nearest(episodes, state, domain=None, home_only=False):
    best = None
    for episode in episodes:
        if domain and episode["meta"]["domain"] != domain:
            continue
        q = episode["state"][:15] if home_only else episode["state"][:-49]
        distances = np.sqrt(np.mean(np.square(q[:, :7]-state[:7]), axis=1))
        index = int(np.argmin(distances))
        if best is None or distances[index] < best[0]:
            best = (float(distances[index]), episode, index)
    return best


def tensor_observation(sample):
    obs = {"observation.state": torch.from_numpy(sample["state"].copy())}
    obs.update({"observation.images."+role: torch.from_numpy(sample["images"][role].copy()).permute(2, 0, 1).float()/255.
                for role in ROLES})
    return obs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", type=Path, default=ROOT/"results/act_manual_21c7jla7/act")
    parser.add_argument("--previous-trial", type=Path,
                        default=ROOT/"results/act_manual_jiazbclr/act_completed_attempt_1792549874362058617")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(1000)
    current, current_summary = trial_inputs(args.trial)
    previous, previous_summary = trial_inputs(args.previous_trial)
    train, validation = numeric_episodes("train"), numeric_episodes("validation")
    report = {"hardware_io": False, "gpu_inference": False, "training_started": False,
              "scope": "Stored inputs; TRAIN and validation phase diagnostics, no test images or closed-loop success inference",
              "current_trial": current_summary, "previous_trial": previous_summary,
              "same_model_weights": current_summary["weight_sha256"] == previous_summary["weight_sha256"],
              "train_recordings": len(train), "train_unique_sources": len({ep["meta"]["group"] for ep in train}),
              "nearest_train": [], "phase_queries": []}
    print("INPUT_AND_NUMERIC_AUDIT_READY", json.dumps(current_summary), flush=True)
    panels = []
    nearest_stall = None
    for live in current:
        distance, episode, frame = nearest(train, live["state"])
        demo = demo_sample(episode, frame)
        report["nearest_train"].append({"live_label": live["label"], "right7_rms_rad": distance,
                                        "episode_id": demo["episode_id"], "frame": frame,
                                        "domain": demo["domain"], "live_state": live["state"].tolist(),
                                        "demo_state": demo["state"].tolist()})
        if live["label"] == "stall":
            nearest_stall = episode
    for title, sample in (("previous real50K HOME", previous[0]), ("current queue10 HOME", current[0]),
                          ("previous real50K stall", previous[1]), ("current queue10 stall", current[1])):
        panels.append((title, sample))
    for domain in ("latest_real", "old_real"):
        distance, episode, frame = nearest(train, current[0]["state"], domain, True)
        sample = demo_sample(episode, frame)
        panels.append((f"TRAIN {domain} HOME {sample['episode_id']}", sample))
    _, episode, frame = nearest(train, current[1]["state"], "latest_real")
    panels.append((f"TRAIN nearest stalled pose {episode['meta']['episode_id']} f{frame}", demo_sample(episode, frame)))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(panels), 2, figsize=(12, 3*len(panels)))
    for axs, (title, sample) in zip(axes, panels):
        for ax, role in zip(axs, ROLES):
            ax.imshow(sample["images"][role]); ax.set_title(title+" "+role, fontsize=9); ax.axis("off")
    fig.tight_layout()
    fig.savefig(args.output/"scene_comparison.png", dpi=110)
    plt.close(fig)
    (args.output/"report.json").write_text(json.dumps(report, indent=2)+"\n")
    print("SCENE_COMPARISON_READY", flush=True)

    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors
    cp = current_summary["checkpoint"]
    cfg = ACTConfig.from_pretrained(cp); cfg.device = "cpu"
    policy = ACTPolicy.from_pretrained(cp, config=cfg).to("cpu").eval()
    pre, post = make_pre_post_processors(cfg, pretrained_path=cp,
                                        preprocessor_overrides={"device_processor": {"device": "cpu"}})
    def predict(sample):
        policy.reset(); pre.reset(); post.reset()
        with torch.inference_mode():
            prediction = policy.predict_action_chunk(pre(tensor_observation(sample)))
            return post(prediction.reshape(-1, 16)).reshape(50, 16).cpu().numpy()

    repeated = [predict(current[1]) for _ in range(3)]
    report["same_input_repeat_max_diff_rad"] = float(max(np.max(np.abs(p-repeated[0])) for p in repeated))
    report["cpu_vs_recorded_gpu_max_diff_rad"] = float(np.max(np.abs(repeated[0]-np.array(current[1]["row"]["full_postprocessed_actions"]))))
    report["natural_home_input_comparison"] = []
    for name, sample in (("previous", previous[0]), ("current", current[0])):
        pred = predict(sample)
        report["natural_home_input_comparison"].append({"input": name, "state": sample["state"].tolist(),
                                                       "first_prediction": pred[0].tolist(),
                                                       "first10_arm_motion_rad": float(np.abs(pred[:10, :7]-sample["state"][:7]).mean())})

    selected_train = []
    for episode in [nearest_stall, *[ep for ep in train if ep["meta"]["domain"] == "latest_real"]]:
        if episode["meta"]["group"] not in {ep["meta"]["group"] for ep in selected_train}:
            selected_train.append(episode)
        if len(selected_train) == 3:
            break
    jobs = selected_train + validation
    for episode in jobs:
        close, release = episode["close"], episode["release"]
        phases = {"initial": 0, "approach": close-30, "preclose": close-10, "close": close,
                  "after_close": close+15, "prerelease": release-15, "release": release, "after_release": release+15}
        for phase, frame in phases.items():
            frame = int(np.clip(frame, 0, len(episode["state"])-50))
            sample = demo_sample(episode, frame)
            pred = predict(sample)
            truth = sample["truth"]
            report["phase_queries"].append({"split": episode["split"], "domain": sample["domain"],
                                             "episode_id": sample["episode_id"], "phase": phase, "frame": frame,
                                             "arm_first_mae_rad": float(np.abs(pred[0, :7]-truth[0, :7]).mean()),
                                             "gripper_target": float(truth[0, 7]), "gripper_prediction": float(pred[0, 7]),
                                             "target_close_first10": bool(np.any(truth[:10, 7] > -.225)),
                                             "prediction_close_first10": bool(np.any(pred[:10, 7] > -.225)),
                                             "prediction_close_future50": bool(np.any(pred[:, 7] > -.225)),
                                             "target_gripper": truth[:, 7].tolist(), "predicted_gripper": pred[:, 7].tolist()})
        (args.output/"report.json").write_text(json.dumps(report, indent=2)+"\n")
        print("PHASE_EPISODE_DONE", episode["split"], episode["meta"]["episode_id"], flush=True)
    summary = {}
    for split in ("train", "validation"):
        rows = [row for row in report["phase_queries"] if row["split"] == split]
        summary[split] = {phase: {"queries": len(rs),
                                 "arm_first_mae_rad": float(np.mean([r["arm_first_mae_rad"] for r in rs])),
                                 "gripper_target_median": float(np.median([r["gripper_target"] for r in rs])),
                                 "gripper_prediction_median": float(np.median([r["gripper_prediction"] for r in rs])),
                                 "gripper_first_mae_rad": float(np.mean([abs(r["gripper_target"]-r["gripper_prediction"]) for r in rs]))}
                          for phase in {row["phase"] for row in rows}
                          for rs in [[row for row in rows if row["phase"] == phase]]}
    report["phase_summary"] = summary
    report["limitations"] = ["Only three TRAIN and six validation episodes, eight phase queries per recording",
                              "Nearest by joint pose is not an identical image or object-relative pose",
                              "No physical camera calibration or causal attribution from image differences",
                              "Repeated identical input determinism does not imply repeated physical rollout determinism"]
    (args.output/"report.json").write_text(json.dumps(report, indent=2)+"\n")
    print("DIAGNOSIS_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
