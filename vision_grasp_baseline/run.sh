#!/usr/bin/env bash
set -eo pipefail
baseline_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
workspace_dir=$(cd -- "$baseline_dir/.." && pwd)
cd "$workspace_dir"
stage=${1:---help}
shift || true
case "$stage" in
  install-h1-fix) exec /usr/bin/python3 -m scripts.control_ownership.h1_gripper_fix --closed-transit "$@" ;;
  stop) touch "$baseline_dir/STOP"; echo 'Visual baseline stop requested; ROS remains enabled.'; exit 0 ;;
  cycle) exec /usr/bin/python3 -m vision_grasp_baseline.cycle "$@" ;;
  collect) exec /usr/bin/python3 -m vision_grasp_baseline.collect "$@" ;;
  camera-check) exec /usr/bin/python3 -m vision_grasp_baseline.collect --camera-check "$@" ;;
  recording-check) exec /usr/bin/python3 -m vision_grasp_baseline.recording_check "$@" ;;
  audit-act) exec "$baseline_dir/.venv/bin/python" -m vision_grasp_baseline.training_contract "$@" ;;
  reset-collection) exec /usr/bin/python3 -m vision_grasp_baseline.collection_layout "$@" ;;
  capture) exec "$baseline_dir/.venv/bin/python" "$baseline_dir/capture.py" "$@" ;;
  detect) exec "$baseline_dir/.venv/bin/python" "$baseline_dir/perception.py" "$@" ;;
  gateway) exec ./scripts/actctl "$stage" "$@" ;;
esac
source scripts/home_replay/workstation_env.sh
source .control_ownership_overlay/install/setup.bash
unset ROS_SECURITY_ENCLAVE_OVERRIDE OPENARM_AUTHORITY_TOKEN
export OPENARM_KEYSTORE="$workspace_dir/.control_ownership_keys/physical117"
export OPENARM_AUTHORITY_SOCKET="/run/user/$(id -u)/openarm-control.sock"
security_exports=$(/usr/bin/python3 -m scripts.control_ownership.security env --keystore "$OPENARM_KEYSTORE" --domain 117)
eval "$security_exports"
case "$stage" in
  h1)
    exec /usr/bin/python3 -m vision_grasp_baseline.collection_h1 --keystore "$OPENARM_KEYSTORE" --physical "$@" ;;
  collection-service)
    exec "$baseline_dir/.venv/bin/python" -m vision_grasp_baseline.collection_service "$@" --ros-args --enclave /openarm/observer ;;
  recovery-h1)
    source "$baseline_dir/.recovery_overlay/install/setup.bash"
    exec /usr/bin/python3 -m vision_grasp_baseline.recovery_launch --keystore "$OPENARM_KEYSTORE" --physical "$@" ;;
  status|synchronize-hold|reset-fault)
    exec /usr/bin/python3 -m scripts.control_ownership.client "$stage" "$@" ;;
  hover-plan)
    exec /usr/bin/python3 -m vision_grasp_baseline.hover --mode plan "$@" --ros-args --enclave /openarm/h2 ;;
  observe)
    exec /usr/bin/python3 -m vision_grasp_baseline.observe "$@" --ros-args --enclave /openarm/h2 ;;
  reachability)
    exec /usr/bin/python3 -m vision_grasp_baseline.reachability "$@" --ros-args --enclave /openarm/h2 ;;
  grasp-plan)
    exec /usr/bin/python3 -m vision_grasp_baseline.grasp "$@" --ros-args --enclave /openarm/h2 ;;
  continuous-plan)
    exec /usr/bin/python3 -m vision_grasp_baseline.continuous_plan "$@" --ros-args --enclave /openarm/h2 ;;
  continuous-check)
    exec /usr/bin/python3 -m vision_grasp_baseline.continuous_execute --check-only "$@" --ros-args --enclave /openarm/h2 ;;
  continuous-execute)
    exec /usr/bin/python3 -m scripts.control_ownership.client run --owner h2 -- \
      /usr/bin/python3 -m vision_grasp_baseline.continuous_execute "$@" --ros-args --enclave /openarm/h2 ;;
  grasp-test)
    exec /usr/bin/python3 -m scripts.control_ownership.client run --owner h2 -- \
      /usr/bin/python3 -m vision_grasp_baseline.grasp_test "$@" --ros-args --enclave /openarm/h2 ;;
  place-test)
    exec /usr/bin/python3 -m scripts.control_ownership.client run --owner h2 -- \
      /usr/bin/python3 -m vision_grasp_baseline.place_test "$@" --ros-args --enclave /openarm/h2 ;;
  waypoint-place-test)
    exec /usr/bin/python3 -m scripts.control_ownership.client run --owner h2 -- \
      /usr/bin/python3 -m vision_grasp_baseline.waypoint_place_test "$@" --ros-args --enclave /openarm/h2 ;;
  waypoint-place-plan)
    exec /usr/bin/python3 -m vision_grasp_baseline.waypoint_place_test --plan-only "$@" --ros-args --enclave /openarm/h2 ;;
  clear-view)
    exec /usr/bin/python3 -m scripts.control_ownership.client run --owner h2 -- \
      /usr/bin/python3 -m vision_grasp_baseline.clear_view "$@" --ros-args --enclave /openarm/h2 ;;
  zero-plan)
    exec /usr/bin/python3 -m vision_grasp_baseline.zero "$@" --ros-args --enclave /openarm/h2 ;;
  zero-test)
    exec /usr/bin/python3 -m scripts.control_ownership.client run --owner h2 -- \
      /usr/bin/python3 -m vision_grasp_baseline.zero --execute "$@" --ros-args --enclave /openarm/h2 ;;
  hover-execute)
    exec /usr/bin/python3 -m scripts.control_ownership.client run --owner h2 -- \
      /usr/bin/python3 -m vision_grasp_baseline.hover --mode execute "$@" --ros-args --enclave /openarm/h2 ;;
  *)
    echo 'Usage: ./vision_grasp_baseline/run.sh <stage> [arguments]'
    echo 'Main: collect [--episodes 30] [--tag nearby|spread|rotation] [--replay-plan NEW_POLICY_FILE] [--output DIR] | camera-check | recording-check [--seconds 60] [--plans 5] | cycle --continuous [--output DIR] | status | stop'
    echo 'Startup: install-h1-fix (power OFF, H1/gateway stopped) | h1 (closed-transit collection zero) | recovery-h1 (hold measured pose) | gateway | synchronize-hold'
    echo 'Inspect: capture | detect | observe | reachability | zero-plan | hover-plan | grasp-plan | continuous-plan | continuous-check'
    echo 'Offline data audit (no hardware): audit-act EPISODE_DIR [EPISODE_DIR ...]'
    echo 'Storage reset (no hardware): reset-collection --tag all [--archive-existing]; default is preview only'
    echo 'Attended physical stages: zero-test | continuous-execute | hover-execute | grasp-test | place-test | waypoint-place-test | clear-view'
    ;;
esac
