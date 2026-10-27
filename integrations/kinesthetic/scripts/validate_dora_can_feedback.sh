#!/usr/bin/env bash
# Feedback only: does not start a Dora dataflow or enable/disable motors.
set -euo pipefail
workspace=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
vla_workspace=${OPENARM_VLA_WS:-"$(dirname -- "$workspace")/openarm_vla_ws"}
target_python=${DORA_PYTHON:-"$vla_workspace/official/dora-openarm-evaluation/.venv/bin/python"}
exec "$target_python" "$vla_workspace/scripts/validate_openarm_feedback.py" "$@"
