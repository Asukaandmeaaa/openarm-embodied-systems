#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PATH="$ROOT/.venv/bin:$PATH"

pytest -q official/dora-openarm-keyboard/tests
pytest -q official/dora-openarm-evaluation/nodes/dora-openarm-dataset-recorder/tests
pytest -q official/dora-openarm-evaluation/nodes/dora-openarm-mujoco/tests
pytest -q sim/teleop/state_join/tests
python sim/validation/dump_feature_contract.py
dora graph configs/collection/openarm_cube_pick.yaml --mermaid >/dev/null
dora build configs/collection/openarm_cube_pick.yaml --uv

echo "P1 preflight passed"
