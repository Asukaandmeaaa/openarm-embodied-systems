# Source only. Align recorder output with the selected processed trajectory root.
# Starts no camera, recorder, controller or motion.
source /home/lushilin/openarm_vla_ws/scripts/real14_smooth_env.sh || return
export COLLECT_ROOT="$SMOOTH_ROOT"
export COLLECT_BATCH=adjusted_scene_real14_20261016_smooth_v1
