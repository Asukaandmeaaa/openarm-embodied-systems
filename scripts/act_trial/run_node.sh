#!/usr/bin/env bash
# Launch only trial software; ROS is started separately and retains CAN ownership.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
workspace_dir=$(cd -- "$script_dir/../.." && pwd)

if [[ $# -lt 1 ]]; then
  echo 'Usage: run_node.sh {bridge|executor|policy-server|policy-client|camera} [arguments...]' >&2
  exit 2
fi
role=$1
shift

: "${ACT_TRIAL_DIR:?ACT_TRIAL_DIR must name the prepared trial logging directory}"
: "${SOCKET:?SOCKET must name the dedicated policy server Unix socket}"
case "$ACT_TRIAL_DIR" in
  /*) ;;
  *) echo 'ACT_TRIAL_DIR must be an absolute path' >&2; exit 2 ;;
esac
case "$SOCKET" in
  /*) ;;
  *) echo 'SOCKET must be an absolute path' >&2; exit 2 ;;
esac
if [[ ! -d "$ACT_TRIAL_DIR" || ! -w "$ACT_TRIAL_DIR" ]]; then
  echo 'ACT_TRIAL_DIR must already exist and be writable' >&2
  exit 2
fi

# ROS setup scripts may refer to unset variables. This only configures imports;
# sourcing workstation_env.sh does not create ROS nodes or touch either CAN bus.
set +u
ownership_domain=${ROS_DOMAIN_ID:-}
source "$workspace_dir/scripts/home_replay/workstation_env.sh"
if [[ ${ROS_SECURITY_ENABLE:-} == true && ${ROS_SECURITY_STRATEGY:-} == Enforce ]]; then
  : "${ownership_domain:?Secure trial requires an explicit ROS_DOMAIN_ID}"
  export ROS_DOMAIN_ID="$ownership_domain"
fi
set -u
export PYTHONPATH="$workspace_dir/official/dora-openarm-evaluation/src:$workspace_dir:${PYTHONPATH:-}"
export OPENARM_PRETRAINED_PATH="${OPENARM_PRETRAINED_PATH:-$workspace_dir/outputs/act_real_combined_unreviewed_30000_20261008/checkpoints/030000/pretrained_model}"
export REQUIRE_SOURCE_TIMING="${REQUIRE_SOURCE_TIMING:-yes}"
export MAX_SOURCE_AGE_MS="${MAX_SOURCE_AGE_MS:-200}"
export MAX_SOURCE_SKEW_MS="${MAX_SOURCE_SKEW_MS:-100}"
export MAX_OBSERVATION_AGE_MS="${MAX_OBSERVATION_AGE_MS:-500}"

case "$role" in
  camera)
    interpreter="$workspace_dir/official/dora-openarm-evaluation/.venv/bin/python"
    entrypoint="$script_dir/configured_camera.py"
    ;;
  bridge)
    interpreter="$workspace_dir/official/dora-openarm-evaluation/.venv/bin/python"
    entrypoint="$script_dir/ros_bridge.py"
    ;;
  executor)
    interpreter="$workspace_dir/official/dora-openarm-evaluation/.venv/bin/python"
    entrypoint="$script_dir/logged_executor.py"
    ;;
  policy-server)
    interpreter="$workspace_dir/.venv-real-act/bin/python"
    entrypoint="$script_dir/logged_policy_server.py"
    ;;
  policy-client)
    interpreter="$workspace_dir/official/dora-openarm-evaluation/.venv/bin/python"
    entrypoint="$script_dir/queued_policy_client.py"
    ;;
  *)
    echo "Unknown trial software role: $role" >&2
    exit 2
    ;;
esac
if [[ ! -x "$interpreter" || ! -f "$entrypoint" ]]; then
  echo "Trial interpreter or entrypoint is unavailable for role: $role" >&2
  exit 2
fi

# Use the venv interpreter path directly: Dora canonicalizes executable symlinks.
# Terminal commands use tee: keep startup/readiness/errors visible on pipes.
exec "$interpreter" -u "$entrypoint" "$@"
