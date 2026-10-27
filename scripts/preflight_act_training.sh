#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RAW_DATASET="${RAW_DATASET:-data/raw/openarm_cube_pick/expert_pilot_g3_frozen_20260903}"
LEROBOT_DATASET="${LEROBOT_DATASET:-data/lerobot/openarm_cube_pick_expert_pilot_v3_nosmoothing}"
REPO_ID="${REPO_ID:-local/openarm_cube_pick_expert_pilot_v3_nosmoothing}"
MIN_ROBUST_EPISODES="${MIN_ROBUST_EPISODES:-4}"
PYTHON_BIN="${PYTHON_BIN:-/home/lushilin/miniconda3/envs/vla/bin/python}"
REPORT="${REPORT:-}"
SKIP_LEROBOT="${SKIP_LEROBOT:-0}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python environment not found: $PYTHON_BIN" >&2
  exit 2
fi
if [[ ! -f "$RAW_DATASET/metadata.yaml" ]]; then
  echo "Raw OpenArm dataset not found: $RAW_DATASET" >&2
  exit 2
fi
if [[ "$SKIP_LEROBOT" != "1" && ! -f "$LEROBOT_DATASET/meta/info.json" ]]; then
  echo "LeRobot dataset not found: $LEROBOT_DATASET" >&2
  exit 2
fi
if ! [[ "$MIN_ROBUST_EPISODES" =~ ^[1-9][0-9]*$ ]]; then
  echo "MIN_ROBUST_EPISODES must be a positive integer" >&2
  exit 2
fi

if [[ -n "$REPORT" ]]; then
  REPLAY_REPORT="$REPORT"
  mkdir -p "$(dirname "$REPLAY_REPORT")"
else
  REPLAY_REPORT="$(mktemp --tmpdir openarm-act-replay.XXXXXX.json)"
  trap 'rm -f "$REPLAY_REPORT"' EXIT
fi

export PYTHONPATH="$ROOT:$ROOT/official/openarm_dataset/src:$ROOT/official/openarm_mujoco/src${PYTHONPATH:+:$PYTHONPATH}"
# Keep the validation cache outside read-only home directories used by managed
# runners. Override OPENARM_CACHE_DIR to retain it elsewhere.
OPENARM_CACHE_DIR="${OPENARM_CACHE_DIR:-/tmp/openarm_vla_hf_cache}"
export HF_HOME="${HF_HOME:-$OPENARM_CACHE_DIR}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"

TASK_PROMPT="$($PYTHON_BIN - "$RAW_DATASET/metadata.yaml" <<'PY'
import sys
import yaml

with open(sys.argv[1], encoding="utf-8") as stream:
    metadata = yaml.safe_load(stream) or {}
tasks = metadata.get("tasks", [])
if len(tasks) != 1 or not tasks[0].get("prompt"):
    raise SystemExit("raw dataset metadata must contain exactly one task prompt")
print(tasks[0]["prompt"])
PY
)"

echo "[1/4] Validate the raw OpenArmDataset structure"
"$PYTHON_BIN" -c 'from openarm_dataset.validate import main; main()' \
  --no-update-metadata "$RAW_DATASET"

echo "[2/4] Validate the frozen feature/camera contract"
"$PYTHON_BIN" sim/validation/validate_pilot.py \
  "$RAW_DATASET" --min-success "$MIN_ROBUST_EPISODES"

echo "[3/4] Strictly replay every expert episode"
replay_status=0
"$PYTHON_BIN" sim/validation/expert_action_replay.py \
  --dataset "$RAW_DATASET" --episode all --output "$REPLAY_REPORT" \
  > /dev/null || replay_status=$?
if (( replay_status != 0 && replay_status != 2 )); then
  echo "Expert replay failed to run (exit $replay_status)" >&2
  exit "$replay_status"
fi

accepted="$($PYTHON_BIN - "$REPLAY_REPORT" "$MIN_ROBUST_EPISODES" <<'PY'
import json
import sys

report_path, minimum = sys.argv[1], int(sys.argv[2])
with open(report_path, encoding="utf-8") as stream:
    report = json.load(stream)
accepted = [int(item["episode"]) for item in report["episodes"] if item["valid"]]
rejected = [
    {
        "episode": int(item["episode"]),
        "seed": int(item["seed"]),
        "success": bool(item["success"]),
        "max_lift_m": float(item["max_lift_height_m"]),
        "hold_s": float(item["hold_duration_s"]),
    }
    for item in report["episodes"]
    if not item["valid"]
]
print(",".join(map(str, accepted)))
print(f"Replay-valid episodes: {accepted}", file=sys.stderr)
if rejected:
    print(f"Rejected episodes: {json.dumps(rejected)}", file=sys.stderr)
if len(accepted) < minimum:
    print(
        f"Only {len(accepted)} replay-valid episodes; require at least {minimum}",
        file=sys.stderr,
    )
    raise SystemExit(2)
PY
)"

if [[ "$SKIP_LEROBOT" == "1" ]]; then
  echo "[4/4] Skip LeRobot validation (SKIP_LEROBOT=1)"
else
  echo "[4/4] Traverse the converted LeRobot dataset and audit normalization"
  "$PYTHON_BIN" sim/validation/validate_lerobot.py \
    "$LEROBOT_DATASET" \
    --repo-id "$REPO_ID" \
    --prompt "$TASK_PROMPT" \
    --min-episodes "$MIN_ROBUST_EPISODES" \
    --state-normalization identity
fi

echo
echo "ACT training preflight passed."
echo "Replay-validated episode IDs: [$accepted]"
echo "Use only these IDs for the small-data memorization gate."
echo "Keep ACT STATE normalization at IDENTITY."
