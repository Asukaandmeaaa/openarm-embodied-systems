# Source only; independent historical batch. No nodes or robot commands.
source /home/lushilin/openarm_kinesthetic_ws/scripts/original_replay_env.sh || return
export CFG=/home/lushilin/openarm_kinesthetic_ws/ros2_ws/src/openarm_kinesthetic_recorder/config/real_kinesthetic.yaml
export SMOOTH_ROOT=/home/lushilin/openarm_vla_ws/openarm_kinesthetic_data/collections_legacy25_20261016_smooth_v1
sha256sum --check --status "$SMOOTH_ROOT/notes/replay14_files.sha256" || return 1
mapfile -t LEGACY_SMOOTH_FILES < "$SMOOTH_ROOT/notes/replay14_files.txt"
[[ ${#LEGACY_SMOOTH_FILES[@]} -eq 23 ]] || return 1
export COLLECT_ROOT="$SMOOTH_ROOT"
export COLLECT_BATCH=legacy25_20261016_smooth_v1
export REPLAY_LOG_DIR="$SMOOTH_ROOT/replay_validation"
mkdir -p "$REPLAY_LOG_DIR"
echo '已加载旧专家轨迹平滑批次：23条；10月8日008、012已隔离。'
