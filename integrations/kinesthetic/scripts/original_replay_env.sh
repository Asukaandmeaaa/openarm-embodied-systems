# Source in fresh terminals for the original domain-97 ROS workflow.
# This file launches no nodes and sends no robot commands.
if [[ "${AMENT_PREFIX_PATH:-}:${LD_LIBRARY_PATH:-}" == *'.control_ownership_overlay'* ]]; then
  echo 'Open a fresh terminal; do not source the ACT environment before this file.' >&2
  return 1
fi
unset ROS_SECURITY_ENABLE ROS_SECURITY_STRATEGY ROS_SECURITY_KEYSTORE ROS_SECURITY_ENCLAVE_OVERRIDE
unset OPENARM_AUTHORITY_TOKEN ROS_LOCALHOST_ONLY
source /opt/ros/jazzy/setup.bash || return
source /home/lushilin/openarm_ws/install/setup.bash || return
source /home/lushilin/openarm_kinesthetic_ws/ros2_ws/install-h1h2-capture/setup.bash || return
export KIN_REPLAY_ROOT=/home/lushilin/openarm_kinesthetic_ws
export PYTHONPATH="$KIN_REPLAY_ROOT/src:$KIN_REPLAY_ROOT/ros2_ws/src/openarm_kinesthetic_recorder:${PYTHONPATH:-}"
export ROS_DOMAIN_ID=97
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export REPLAY_LOG_DIR="$KIN_REPLAY_ROOT/results/manual_replay"
mkdir -p "$REPLAY_LOG_DIR"
set -o pipefail
