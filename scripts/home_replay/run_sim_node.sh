#!/usr/bin/env bash
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
workspace_dir=$(cd -- "$script_dir/../.." && pwd)
# Dora canonicalizes executable symlinks; a wrapper preserves venv discovery.
exec "$workspace_dir/official/dora-openarm-evaluation/.venv/bin/python" \
  "$script_dir/simulate.py" --dora "$@"
