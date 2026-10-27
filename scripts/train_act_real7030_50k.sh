#!/usr/bin/env bash
set -euo pipefail
cd /home/lushilin/openarm_vla_ws
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=4
if [[ ${1:-} == --audit && $# == 1 ]]; then
  exec .venv-real-act/bin/python scripts/train_act_real7030.py --audit --steps 50000
fi
if [[ $# != 0 ]]; then echo 'Usage: bash scripts/train_act_real7030_50k.sh [--audit]' >&2; exit 2; fi
if [[ -e outputs/act_real7030_50k_20261020 ]]; then
  echo 'Output already exists; refusing overwrite. Preserve it for explicit resume setup.' >&2
  exit 2
fi
exec .venv-real-act/bin/python scripts/train_act_real7030.py \
  --config_path=configs/training/act_real7030_50k_20261020.json
