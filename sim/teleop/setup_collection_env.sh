#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

uv pip install --python .venv/bin/python \
  -e official/openarm_mujoco \
  -e official/openarm_dataset \
  -e official/dora-openarm-keyboard \
  -e official/dora-openarm-data-collection/nodes/dora-openarm-data-collection-ui \
  -e official/dora-openarm-data-collection/nodes/dora-openarm-kinematics \
  -e official/dora-openarm-data-collection/nodes/dora-openarm-quitter \
  -e official/dora-openarm-evaluation/nodes/dora-openarm-dataset-recorder \
  -e official/dora-openarm-evaluation/nodes/dora-openarm-mujoco \
  -e sim/teleop/state_join \
  dora-rs-cli==0.5.0 pytest pillow pyyaml

echo "Collection environment ready: $ROOT/.venv"
