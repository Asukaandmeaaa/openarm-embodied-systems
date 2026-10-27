#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DATASET="${1:?usage: $0 LEROBOT_DATASET [OUTPUT_DIR]}"
OUTPUT_DIR="${2:-outputs/act_openarm_cube_pick_20k}"
PYTHON_BIN="${PYTHON_BIN:-python}"
STEPS="${STEPS:-20000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MIN_EPISODES="${MIN_EPISODES:-20}"
EVAL_SPLIT="${EVAL_SPLIT:-0.1}"
EVAL_STEPS="${EVAL_STEPS:-1000}"
SKIP_VALIDATION="${SKIP_VALIDATION:-0}"
PROMPT="${PROMPT:-Pick up the orange cube with the right arm.}"
LOG_PATH="${OPENARM_TRAIN_LOG:-results/act_training/$(basename "$OUTPUT_DIR").log}"

# LeRobot rejects periodic offline evaluation when no episodes are held out.
# Keep all explicitly separated training episodes when EVAL_SPLIT is zero.
case "$EVAL_SPLIT" in
  0|0.0|0.00|0.000) EVAL_STEPS=0 ;;
esac

if [[ "$SKIP_VALIDATION" != "1" ]]; then
  "$PYTHON_BIN" sim/validation/validate_lerobot.py \
    "$DATASET" \
    --repo-id local/openarm_cube_pick_act_train \
    --prompt "$PROMPT" \
    --min-episodes "$MIN_EPISODES" \
    --state-normalization identity
else
  echo "Skipping LeRobot traversal validation (SKIP_VALIDATION=1)."
fi

OPENARM_TRAIN_LOG="$LOG_PATH" exec "$PYTHON_BIN" scripts/train_act_smoke_with_log.py \
  --dataset.repo_id=local/openarm_cube_pick_act_train \
  --dataset.root="$DATASET" \
  --dataset.eval_split="$EVAL_SPLIT" \
  --policy.type=act \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.normalization_mapping='{"VISUAL":"MEAN_STD","STATE":"IDENTITY","ACTION":"MEAN_STD"}' \
  --output_dir="$OUTPUT_DIR" \
  --job_name="$(basename "$OUTPUT_DIR")" \
  --steps="$STEPS" \
  --batch_size="$BATCH_SIZE" \
  --num_workers="$NUM_WORKERS" \
  --eval_steps="$EVAL_STEPS" \
  --max_eval_samples=256 \
  --env_eval_freq=0 \
  --save_freq=5000 \
  --log_freq=50 \
  --seed=1000 \
  --cudnn_deterministic=true \
  --wandb.enable=false
