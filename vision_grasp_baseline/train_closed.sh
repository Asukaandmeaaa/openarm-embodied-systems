#!/usr/bin/env bash
set -euo pipefail
task_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$task_root"
task_python="$task_root/.venv-real-act/bin/python"
task_config=${OPENARM_CLOSED_CONFIG:-"$task_root/configs/training/act_closed_v2_20261023a_50k.json"}
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
mode=${1:---help}
if (( $# )); then shift; fi
case "$mode" in
  prepare) exec nice -n 10 ionice -c 3 "$task_python" -m vision_grasp_baseline.prepare_closed_training "$@" ;;
  smoke)
    smoke_stamp=$(date +%Y%m%d_%H%M%S)
    export OPENARM_TRAIN_LOG="$task_root/results/act_training/act_closed_smoke_$smoke_stamp.log"
    exec nice -n 10 ionice -c 3 "$task_python" scripts/train_act_closed.py --config_path="$task_config" \
      --steps=2 --num_workers=0 --eval_steps=2 --max_eval_samples=0 --save_freq=2 --log_freq=1 \
      --output_dir="$task_root/outputs/act_closed_smoke_$smoke_stamp" "$@" ;;
  train) exec nice -n 10 ionice -c 3 "$task_python" scripts/train_act_closed.py --config_path="$task_config" "$@" ;;
  evaluate) exec nice -n 10 ionice -c 3 "$task_python" scripts/evaluate_act_closed.py "$@" ;;
  *) echo 'Usage: bash vision_grasp_baseline/train_closed.sh prepare [--audit-only] | smoke | train [LeRobot overrides] | evaluate --checkpoint DIR --output JSON' ;;
esac
