# Source only: loads the fixed, audited 14-file selection. Starts no nodes.
source /home/lushilin/openarm_vla_ws/scripts/real14_env.sh || return
sha256sum --check --status "$COLLECT_ROOT/notes/replay14_files.sha256" || {
  echo '轨迹内容已改变，停止使用旧的14条清单。' >&2
  return 1
}
mapfile -t REAL14_FILES < "$COLLECT_ROOT/notes/replay14_files.txt"
[[ ${#REAL14_FILES[@]} -eq 14 ]] || return 1
export REPLAY_LOG_DIR="$COLLECT_ROOT/replay_validation"
mkdir -p "$REPLAY_LOG_DIR"
real14_select() {
  unset FILE TRIAL
  if [[ $# -ne 1 || ! "$1" =~ ^([1-9]|1[0-4])$ ]]; then
    echo '用法：real14_select 1 到 real14_select 14' >&2
    return 1
  fi
  sha256sum --check --status "$COLLECT_ROOT/notes/replay14_files.sha256" || return 1
  export FILE="${REAL14_FILES[$1-1]}"
  export TRIAL="$(basename "$FILE" .json)"
  echo "已选择第 $1/14 条：$FILE"
}
