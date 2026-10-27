#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENARM_TRAIN_LOG="$ROOT/results/act_real_fixed13/train.log"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
PYTHON="$ROOT/.venv-real-act/bin/python"
"$PYTHON" scripts/validate_real_act_dataset.py
set +e
"$PYTHON" -u scripts/train_act_smoke_with_log.py \
  --config_path=configs/training/act_real_fixed13_10000.yaml
TRAIN_STATUS=$?
printf '%s\n' "$TRAIN_STATUS" > "$ROOT/results/act_real_fixed13/train_exit_code.txt"
if [[ "$TRAIN_STATUS" -eq 0 ]]; then
  "$PYTHON" scripts/evaluate_real_act_offline.py \
    "$ROOT/outputs/act_real_fixed13_10000_20260930/checkpoints/last/pretrained_model"
  printf '%s\n' "$?" > "$ROOT/results/act_real_fixed13/offline_eval_exit_code.txt"
fi
exit "$TRAIN_STATUS"
