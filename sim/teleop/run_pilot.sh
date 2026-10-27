#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PATH="$ROOT/.venv/bin:$PATH"

if [[ ! -x .venv/bin/dora ]]; then
  echo "Missing .venv/bin/dora. Run: sim/teleop/setup_collection_env.sh" >&2
  exit 1
fi

mkdir -p data/raw/openarm_cube_pick results/p1_openarm_cube_pick
if [[ -e data/raw/openarm_cube_pick/pilot/metadata.yaml ]]; then
  echo "Resuming existing pilot dataset; recorder will choose the next episode id." >&2
fi

echo "Collection UI: http://127.0.0.1:8000"
echo "Keyboard teleop (click to focus): http://127.0.0.1:8080"
echo "Frozen prompt: Pick up the orange cube with the right arm."
exec dora run configs/collection/openarm_cube_pick.yaml
