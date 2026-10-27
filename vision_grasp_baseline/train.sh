#!/usr/bin/env bash
set -euo pipefail
task_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$task_root"
task_python="$task_root/.venv-real-act/bin/python"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=4
mode=${1:---help}
if (( $# )); then shift; fi
case "$mode" in
  prepare) exec "$task_python" -m vision_grasp_baseline.prepare_training "$@" ;;
  smoke)
    smoke_stamp=$(date +%Y%m%d_%H%M%S)
    export OPENARM_TRAIN_LOG="$task_root/results/act_training/act_visual32_smoke_$smoke_stamp.log"
    exec "$task_python" scripts/train_act_visual32.py --config_path=configs/training/act_visual32_20261022.json \
      --steps=2 --batch_size=8 --num_workers=0 --eval_steps=1 --max_eval_samples=1 --save_freq=2 --log_freq=1 \
      --output_dir="$task_root/outputs/act_visual32_smoke_$smoke_stamp" "$@" ;;
  train) exec "$task_python" scripts/train_act_visual32.py --config_path=configs/training/act_visual32_20261022.json "$@" ;;
  *) echo 'Usage: ./vision_grasp_baseline/train.sh prepare [--audit-only] | smoke | train [LeRobot overrides]' ;;
esac
