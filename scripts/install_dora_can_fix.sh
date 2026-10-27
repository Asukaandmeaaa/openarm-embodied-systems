#!/usr/bin/env bash
# Build for the selected interpreter/architecture; never access robot hardware.
set -euo pipefail
workspace=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
evaluation="$workspace/official/dora-openarm-evaluation"
target_python=${1:-"$evaluation/.venv/bin/python"}
if [[ ! -x "$target_python" ]]; then
  echo "Python not found: $target_python. Pass the Dora environment's Python path." >&2
  exit 1
fi
source_dir="$evaluation/vendor/openarm_can/python"
node_dir="$evaluation/nodes/dora-openarm"
if command -v uv >/dev/null 2>&1; then
  uv pip install --python "$target_python" --no-deps --reinstall "$source_dir"
  uv pip install --python "$target_python" -e "$node_dir"
else
  "$target_python" -m pip install --no-deps --force-reinstall "$source_dir"
  "$target_python" -m pip install -e "$node_dir"
fi
"$target_python" -c 'from dora_openarm.can_runtime import require_paced_can; print("Dora CAN ready:", require_paced_can())'
