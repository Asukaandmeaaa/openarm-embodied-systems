# Source this file in each capture terminal. It starts no nodes and touches no CAN bus.
source /opt/ros/jazzy/setup.bash || return
source /home/lushilin/openarm_ws/install/setup.bash || return
source /home/lushilin/openarm_kinesthetic_ws/ros2_ws/install-h1h2-capture/setup.bash || return
export PYTHONPATH="/home/lushilin/openarm_kinesthetic_ws/src:${PYTHONPATH:-}"
unset ROS_LOCALHOST_ONLY
export ROS_DOMAIN_ID=97
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
export VLA_CAPTURE_ROOT=/home/lushilin/openarm_vla_ws
export KIN_CAPTURE_ROOT=/home/lushilin/openarm_kinesthetic_ws
export H2_CAPTURE_CONFIG="$KIN_CAPTURE_ROOT/ros2_ws/install-h1h2-capture/openarm_kinesthetic_recorder/share/openarm_kinesthetic_recorder/config/home_initialization.yaml"
