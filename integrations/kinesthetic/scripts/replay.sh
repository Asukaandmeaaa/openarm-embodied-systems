#!/usr/bin/env bash
# Original H1/H2/player commands, with automatic dated terminal logs.
set -eo pipefail
REPLAY_SCRIPT_ROOT=/home/lushilin/openarm_kinesthetic_ws
# Start a child shell with only login essentials. Inherited ACT overlays, DDS
# profiles, Python environments and exported shell functions must not leak in.
# This does not modify the caller's shell or stop any running controller.
if [[ "${OPENARM_REPLAY_CLEAN_ENV:-}" != 1 ]]; then
  exec /usr/bin/env -i \
    HOME="$HOME" USER="$(id -un)" LOGNAME="$(id -un)" \
    PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    LANG="${LANG:-C.UTF-8}" TERM="${TERM:-dumb}" \
    XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}" \
    OPENARM_REPLAY_CLEAN_ENV=1 \
    /bin/bash --noprofile --norc "$REPLAY_SCRIPT_ROOT/scripts/replay.sh" "$@"
fi
REPLAY_STAGE=${1:-}
case "$REPLAY_STAGE" in
  h1|h2|play|list|plan) ;;
  *) echo 'Usage: replay.sh {h1|h2|play|list|plan} [--playlist FILE] [--only N]' >&2; exit 2 ;;
esac
shift
# Forward only selection arguments; never allow overriding motion/plan mode.
REPLAY_SELECTION=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --playlist|--only)
      [[ $# -ge 2 && "$REPLAY_STAGE" != h1 && "$REPLAY_STAGE" != h2 ]] || exit 2
      REPLAY_SELECTION+=("$1" "$2"); shift 2 ;;
    *) echo "Unknown selection argument: $1" >&2; exit 2 ;;
  esac
done
if [[ "$REPLAY_STAGE" == list ]]; then
  exec /usr/bin/python3 "$REPLAY_SCRIPT_ROOT/scripts/button_play_recorded_motion.py" --list "${REPLAY_SELECTION[@]}"
fi
source "$REPLAY_SCRIPT_ROOT/scripts/original_replay_env.sh"
# Do not mix the original command path with the secured ACT session.
/usr/bin/python3 - <<'PY'
import sys
sys.path.insert(0, '/home/lushilin/openarm_kinesthetic_ws/scripts')
from button_play_recorded_motion import competing_processes
conflicts = competing_processes()
if conflicts:
    raise SystemExit('Conflicting command/input processes still running; stop them manually first: '+str(conflicts))
PY
if [[ "$REPLAY_STAGE" == h1 ]]; then
  # Hold this lock for the entire H1 lifetime, including CAN initialization.
  exec 9>"$REPLAY_LOG_DIR/.h1.lock"
  flock -n 9 || { echo 'H1 is already starting/running; refusing duplicate startup.' >&2; exit 1; }
  # H1 is the sole hardware backend. Never start a second ROS motor process.
  /usr/bin/python3 - <<'PY'
from pathlib import Path
for proc in Path('/proc').glob('[0-9]*'):
    try:
        args = (proc/'cmdline').read_bytes().split(b'\0')
    except OSError:
        continue
    if any(Path(a.decode(errors='replace')).name == 'ros2_control_node' for a in args if a):
        raise SystemExit('ROS hardware is already running (PID '+proc.name+'); do not launch a second H1.')
PY
fi
REPLAY_STAGE_LOG=$(mktemp -d "$REPLAY_LOG_DIR/$(date +%Y%m%d_%H%M%S)_${REPLAY_STAGE}_XXXXXX")
echo "REPLAY_STAGE_LOG: $REPLAY_STAGE_LOG"
case "$REPLAY_STAGE" in
  h1)
    /usr/bin/python3 "$REPLAY_SCRIPT_ROOT/scripts/replay_can_preflight.py" \
      2>&1 | tee "$REPLAY_STAGE_LOG/can_preflight.log"
    ros2 launch "$REPLAY_SCRIPT_ROOT/scripts/replay_h1.launch.py" \
      arm_type:=openarm_v2.0 use_fake_hardware:=false launch_rviz:=false \
      robot_controller:=joint_trajectory_controller \
      right_can_interface:=can0 left_can_interface:=can1 \
      gravity_compensation:=true gravity_scale:=1.0 \
      gripper_contact_mode:=monitor gripper_torque_threshold:=0.0 \
      gripper_torque_release_threshold:=0.0 gripper_hard_torque_limit:=0.0 \
      2>&1 | tee "$REPLAY_STAGE_LOG/terminal.log"
    ;;
  h2)
    ros2 run openarm_kinesthetic_recorder home_guarded_executor \
      --config "$REPLAY_SCRIPT_ROOT/ros2_ws/src/openarm_kinesthetic_recorder/config/home_initialization.yaml" \
      --allow-real-execution 2>&1 | tee "$REPLAY_STAGE_LOG/terminal.log"
    cp "$HOME/.ros/openarm_kinesthetic/home_initialization.execution.json" "$REPLAY_STAGE_LOG/h2_execution.json"
    ;;
  play)
    /usr/bin/python3 "$REPLAY_SCRIPT_ROOT/scripts/button_play_recorded_motion.py" \
      --allow-real-execution "${REPLAY_SELECTION[@]}" 2>&1 | tee "$REPLAY_STAGE_LOG/terminal.log"
    ;;
  plan)
    /usr/bin/python3 "$REPLAY_SCRIPT_ROOT/scripts/button_play_recorded_motion.py" \
      --plan-only "${REPLAY_SELECTION[@]}" 2>&1 | tee "$REPLAY_STAGE_LOG/terminal.log"
    ;;
esac
