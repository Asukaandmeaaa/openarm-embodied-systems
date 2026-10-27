#!/usr/bin/env bash
set -euo pipefail
cd /home/lushilin/openarm_vla_ws
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=4
if [[ $# -gt 0 ]]; then
  echo '训练：bash scripts/train_act_mixed_50k.sh；恢复请使用文档中的train_act_mixed.py命令。' >&2
  exit 2
fi
exec .venv-real-act/bin/python scripts/train_act_mixed.py --config_path=configs/training/act_mixed_50k_20261016.json
