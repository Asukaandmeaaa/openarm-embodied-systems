#!/usr/bin/env bash
# Operator-run launcher: one B press per episode; never starts H1 or cameras.
set -eo pipefail
source /home/lushilin/openarm_vla_ws/scripts/legacy_smooth_capture_env.sh
if [[ ${1:-} == --list && $# -eq 1 ]]; then
  for index in "${!LEGACY_SMOOTH_FILES[@]}"; do
    printf '%2d  %s\n' "$((index+1))" "$(basename "${LEGACY_SMOOTH_FILES[$index]}")"
  done
  exit 0
fi
CAPTURE_START_NUMBER=${1:-1}
if [[ $# -gt 1 || ! "$CAPTURE_START_NUMBER" =~ ^([1-9]|1[0-9]|2[0-3])$ ]]; then
  echo '用法：bash scripts/capture_legacy_smooth_batch.sh [1～23，默认1]；--list查看顺序' >&2
  exit 2
fi
# The directory entry is an existing batch feature. Refuse a directory that
# differs from the audited list instead of silently capturing extra files.
/usr/bin/python3 - "$SMOOTH_ROOT" <<'PY'
from pathlib import Path
import sys
root=Path(sys.argv[1])
expected=[Path(s).resolve() for s in (root/'notes/replay14_files.txt').read_text().splitlines()]
actual=sorted(p.resolve() for p in (root/'trajectories').glob('*.json'))
if actual!=expected:raise SystemExit('轨迹目录与已检查清单不一致，停止启动。')
PY
echo "从清单第 ${CAPTURE_START_NUMBER}/23 条开始；每次按B只采一条，保存后复位物体再按B。"
ros2 run openarm_kinesthetic_recorder episode_capture_button --ros-args \
  -p trajectory_directory:="$SMOOTH_ROOT/trajectories" \
  -p trajectory_start_number:="$CAPTURE_START_NUMBER" \
  -p validation_mode:=inline \
  -p plan_only_passed:=false -p dry_replay_passed:=false \
  -p position_id_from_source:=true \
  -p batch_id:="$COLLECT_BATCH" \
  -p cube_forward_m:=0.0 -p cube_lateral_m:=0.0 \
  -p placement_measurement_method:=per_episode_marked_origin \
  -p replay_button_index:=1 -p keyframe_settle_tolerance_rad:=0.04 \
  -p allocate_new_episode_ids:=true \
  2>&1 | tee -a "$REPLAY_LOG_DIR/batch_from_${CAPTURE_START_NUMBER}_capture.log"
