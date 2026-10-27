from copy import deepcopy

import pytest
import yaml

from openarm_kinesthetic.config import (
    ConfigError,
    load_yaml,
    reject_non_finite,
    validate_gamepad_control,
    validate_poses,
    validate_project_configs,
    validate_real_ready_search,
    validate_ready_trajectory,
    validate_scene,
    validate_task_contract,
)
from openarm_kinesthetic.real.ready_candidates import RealReadySearchPolicy


def test_project_configs_are_valid() -> None:
    # 19 since ready_search_real.yaml joined the validated project configurations alongside
    # initialization.yaml, ready_search.yaml and ready_trajectory.yaml.
    assert len(validate_project_configs()) == 19


def test_replay_signature_cannot_omit_registered_table_yaw() -> None:
    config = load_yaml("ready_trajectory.yaml")
    config["scene_signature"]["compared_fields"].remove("table_forward_world")
    with pytest.raises(ConfigError, match="replay comparison contract"):
        validate_ready_trajectory(config)


def test_missing_coordinate_frame_is_rejected() -> None:
    config = deepcopy(load_yaml("scene_nominal.yaml"))
    del config["frames"]["world"]
    with pytest.raises(ConfigError, match="coordinate frame"):
        validate_scene(config)


def test_wrong_7d_pose_is_rejected() -> None:
    config = deepcopy(load_yaml("poses.yaml"))
    config["right_task_ready"]["candidate_arm"] = [0.0] * 6
    with pytest.raises(ConfigError, match="length 7"):
        validate_poses(config)


def test_wrong_8d_arm_with_gripper_is_rejected() -> None:
    config = deepcopy(load_yaml("task_contract.yaml"))
    config["ordering"]["right_arm_with_gripper"] = ["x"] * 7
    with pytest.raises(ConfigError, match="8 fields"):
        validate_task_contract(config)


def test_wrong_16d_feature_is_rejected() -> None:
    config = deepcopy(load_yaml("task_contract.yaml"))
    config["features"]["action"]["shape"] = [15]
    with pytest.raises(ConfigError, match=r"\[16\]"):
        validate_task_contract(config)


def test_non_finite_number_is_rejected() -> None:
    with pytest.raises(ConfigError, match="non-finite"):
        reject_non_finite({"bad": float("nan")})


def test_uncalibrated_real_scene_cannot_be_ready() -> None:
    config = deepcopy(load_yaml("scene_real_v1.yaml"))
    assert config["ready"] is False
    config["ready"] = True
    with pytest.raises(ConfigError, match="cannot be ready"):
        validate_scene(config)


def test_motor_zero_cannot_be_act_home_in_pose_contract() -> None:
    config = deepcopy(load_yaml("poses.yaml"))
    config["episode_home"] = "motor_zero"
    with pytest.raises(ConfigError, match="must not be ACT episode Home"):
        validate_poses(config)


def test_motor_zero_cannot_be_act_home_in_data_contract() -> None:
    config = deepcopy(load_yaml("task_contract.yaml"))
    config["episode"]["home_pose"] = "motor_zero"
    with pytest.raises(ConfigError, match="must not be ACT episode Home"):
        validate_task_contract(config)


def test_gamepad_cannot_disable_deadman_or_enable_hardware_access() -> None:
    config = deepcopy(load_yaml("gamepad_control.yaml"))
    config["control"]["activation"] = "toggle"
    with pytest.raises(ConfigError, match="hold_lt_deadman"):
        validate_gamepad_control(config)
    config = deepcopy(load_yaml("gamepad_control.yaml"))
    config["safety"]["robot_hardware_access"] = "allowed"
    with pytest.raises(ConfigError, match="forbid hardware access"):
        validate_gamepad_control(config)


def _real_search_config() -> dict:
    return deepcopy(load_yaml("ready_search_real.yaml"))


def test_config_file_cannot_authorize_real_motion() -> None:
    config = _real_search_config()
    config["scope"]["default_action"] = "execute"
    with pytest.raises(ConfigError, match="plan_only"):
        validate_real_ready_search(config)

    config = _real_search_config()
    config["scope"]["execution_authorization"] = "config_value"
    with pytest.raises(ConfigError, match="allow_real_execution"):
        validate_real_ready_search(config)

    config = _real_search_config()
    config["scope"]["joint_values_source"] = "simulation_quaternion"
    with pytest.raises(ConfigError, match="live_moveit_ik_on_registered_table"):
        validate_real_ready_search(config)

    config = _real_search_config()
    config["scope"]["simulation_prior_role"] = "result"
    with pytest.raises(ConfigError, match="search_order_only"):
        validate_real_ready_search(config)


def test_legacy_pose_cannot_be_promoted_out_of_comparison_only() -> None:
    config = _real_search_config()
    config["commissioning_guards"]["legacy_is_comparison_only"] = False
    with pytest.raises(ConfigError, match="never be the execution target"):
        validate_real_ready_search(config)


def test_search_may_keep_the_operator_selected_position_fixed() -> None:
    config = _real_search_config()
    config["real_ready_search"]["position_offsets_m"]["forward"] = [0.0]
    config["real_ready_search"]["position_offsets_m"]["lateral"] = [0.0]
    config["real_ready_search"]["position_offsets_m"]["normal"] = [0.0]
    validate_real_ready_search(config)


def test_runtime_policy_loader_accepts_a_fixed_registered_point(tmp_path) -> None:
    config = _real_search_config()
    config["real_ready_search"]["position_offsets_m"]["forward"] = [0.0]
    config["real_ready_search"]["position_offsets_m"]["lateral"] = [0.0]
    config["real_ready_search"]["position_offsets_m"]["normal"] = [0.0]
    path = tmp_path / "fixed_ready_search_real.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    policy = RealReadySearchPolicy.load(path)
    assert policy.forward_offsets_m() == (0.0,)
    assert policy.lateral_offsets_m() == (0.0,)
    assert policy.normal_offsets_m() == (0.0,)


def test_search_cannot_aim_below_the_registered_clearance_plane() -> None:
    config = _real_search_config()
    config["real_ready_search"]["position_offsets_m"]["normal"] = [-0.02, 0.0]
    with pytest.raises(ConfigError, match="below the registered clearance plane"):
        validate_real_ready_search(config)


def test_search_offsets_stay_a_local_neighbourhood() -> None:
    config = _real_search_config()
    config["real_ready_search"]["position_offsets_m"]["forward"] = [0.60]
    with pytest.raises(ConfigError, match="maximum_offset_magnitude_m"):
        validate_real_ready_search(config)


def test_profile_verification_replay_gate_cannot_be_disabled() -> None:
    config = load_yaml("ready_trajectory.yaml")
    config["replay_gate"]["require_profile_verified"] = False
    with pytest.raises(ConfigError, match="must be true"):
        validate_ready_trajectory(config)


def test_j4_baseline_must_be_measured_in_the_same_run() -> None:
    config = _real_search_config()
    config["commissioning_guards"]["baseline_source"] = "stored_constant_1.913"
    with pytest.raises(ConfigError, match="solved_in_the_same_run"):
        validate_real_ready_search(config)


def test_j4_guard_must_be_labelled_a_commissioning_gate() -> None:
    config = _real_search_config()
    config["commissioning_guards"]["basis"] = "measured_during_commissioning_run".replace(
        "commissioning", "factory"
    )
    with pytest.raises(ConfigError, match="commissioning risk gate"):
        validate_real_ready_search(config)

    # A guard renamed after hardware vocabulary is an invented datasheet limit.
    config = _real_search_config()
    config["commissioning_guards"]["motor_thermal_j4_limit_rad"] = 1.60
    with pytest.raises(ConfigError, match="hardware limit"):
        validate_real_ready_search(config)


def test_j4_path_gate_is_required_and_positive() -> None:
    config = _real_search_config()
    del config["commissioning_guards"]["max_path_absolute_j4_rad"]
    with pytest.raises(ConfigError, match="max_path_absolute_j4_rad"):
        validate_real_ready_search(config)

    config = _real_search_config()
    config["commissioning_guards"]["max_path_absolute_j4_rad"] = float("nan")
    with pytest.raises(ConfigError, match="max_path_absolute_j4_rad"):
        validate_real_ready_search(config)


def test_j4_improvement_gate_must_declare_when_it_applies() -> None:
    """A floor read against a baseline already below the ceiling is unreachable, not strict.

    On 2026-09-18 a fixed-TCP orientation search rejected all 125 IK solutions because a
    0.30 rad improvement had to be bought from a 0.8340 rad baseline with orientation
    alone, which cannot move the elbow. A configuration that drops that scope rule, or
    invents a different one, must not load.
    """
    config = _real_search_config()
    del config["commissioning_guards"]["improvement_gate_reference"]
    with pytest.raises(ConfigError, match="improvement_gate_reference"):
        validate_real_ready_search(config)

    # "Always demand the improvement" is the rule that was implicit in the search code, and
    # no configuration file may reassert what produced the failure.
    config = _real_search_config()
    config["commissioning_guards"]["improvement_gate_reference"] = "always"
    with pytest.raises(ConfigError, match="improvement_gate_reference"):
        validate_real_ready_search(config)

    # Nor may the floor be dropped on the case it was actually written for.
    config = _real_search_config()
    config["commissioning_guards"]["improvement_gate_reference"] = (
        "never_applied_without_a_failed_region"
    )
    with pytest.raises(ConfigError, match="improvement_gate_reference"):
        validate_real_ready_search(config)


def test_no_admissible_candidate_must_fail_closed() -> None:
    config = _real_search_config()
    config["commissioning_guards"]["on_no_admissible_candidate"] = "fallback_to_legacy"
    with pytest.raises(ConfigError, match="fail_closed"):
        validate_real_ready_search(config)


def test_simulation_prior_only_orders_the_search() -> None:
    config = _real_search_config()
    config["real_ready_search"]["simulation_prior"] = "oblique_90"
    with pytest.raises(ConfigError, match="simulation_prior"):
        validate_real_ready_search(config)


def test_trajectory_selection_order_names_real_metrics() -> None:
    config = _real_search_config()
    config["trajectory_selection"]["order"] = ["looks_natural"]
    with pytest.raises(ConfigError, match="not a planned-path metric"):
        validate_real_ready_search(config)

    config = _real_search_config()
    config["trajectory_selection"]["planning_candidate_limit"] = 99
    with pytest.raises(ConfigError, match="planning_candidate_limit"):
        validate_real_ready_search(config)
