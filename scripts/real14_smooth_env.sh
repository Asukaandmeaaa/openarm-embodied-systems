# Source only; never starts nodes or publishes robot commands.
source /home/lushilin/openarm_vla_ws/scripts/real14_env.sh || return
export SMOOTH_ROOT=/home/lushilin/openarm_vla_ws/openarm_kinesthetic_data/collections_real14_20261016_smooth_v1
sha256sum --check --status "$SMOOTH_ROOT/notes/replay14_files.sha256" || return 1
mapfile -t REAL14_SMOOTH_FILES < "$SMOOTH_ROOT/notes/replay14_files.txt"
[[ ${#REAL14_SMOOTH_FILES[@]} -eq 12 ]] || return 1
export REPLAY_LOG_DIR="$SMOOTH_ROOT/replay_validation"
mkdir -p "$REPLAY_LOG_DIR"
real14_smooth_select() {
    unset FILE TRIAL
    if [[ $# -ne 1 || ! "$1" =~ ^([1-9]|1[0-2])$ ]]; then
        echo '用法：real14_smooth_select 1～12（006、012已隔离）' >&2
        return 1
    fi
    sha256sum --check --status "$SMOOTH_ROOT/notes/replay14_files.sha256" || return 1
    export FILE="${REAL14_SMOOTH_FILES[$1-1]}"
    export TRIAL="$(basename "$FILE" .json)"
    echo "平滑候选 $1/12：$FILE；仍需本条计划检查和人工真实抓放验证。"
}
echo '已加载12条平滑候选；原006、012超出当前J6规划范围，未加入。'
