#!/usr/bin/env bash
set -eo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 /absolute/path/to/right_motion_YYYYMMDD_HHMMSS.json" >&2
  exit 2
fi

ROOT=/home/ubuntu/openarm_kinesthetic_ws
TRAJECTORY_FILE=$1

source /opt/ros/jazzy/setup.bash
source /home/ubuntu/openarm_ws/install/setup.bash
source "$ROOT/ros2_ws/install/setup.bash"
set -u
export PYTHONPATH="$ROOT/src:$ROOT/ros2_ws/src/openarm_kinesthetic_recorder${PYTHONPATH:+:$PYTHONPATH}"
unset ROS_LOCALHOST_ONLY
export ROS_DOMAIN_ID=97
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST

HOME_CONFIG="$ROOT/ros2_ws/install/openarm_kinesthetic_recorder/share/openarm_kinesthetic_recorder/config/home_initialization.yaml"

ros2 run openarm_kinesthetic_recorder home_guarded_executor \
  --config "$HOME_CONFIG" \
  --allow-real-execution

ros2 run openarm_kinesthetic_recorder recorded_motion_player \
  --trajectory "$TRAJECTORY_FILE" \
  --allow-real-execution
