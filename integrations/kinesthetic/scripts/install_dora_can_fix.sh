#!/usr/bin/env bash
# Dora uses the shared VLA driver, separate from the ROS teaching controller.
set -euo pipefail
workspace=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
vla_workspace=${OPENARM_VLA_WS:-"$(dirname -- "$workspace")/openarm_vla_ws"}
exec bash "$vla_workspace/scripts/install_dora_can_fix.sh" "$@"
