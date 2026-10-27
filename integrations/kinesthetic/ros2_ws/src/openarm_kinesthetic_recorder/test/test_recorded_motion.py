import json
from types import SimpleNamespace

import pytest
from action_msgs.msg import GoalStatus
from moveit_msgs.msg import MoveItErrorCodes

from openarm_kinesthetic_recorder.home_measurements import Q_LIFTED_HOME_RIGHT
from openarm_kinesthetic_recorder.recorded_motion import (
    GRIPPER_TARGETS,
    COLLECTION_REPLAY_VELOCITY_LIMITS_RAD_S,
    JOINT_NAMES,
    KIND,
    MARKED_KIND,
    MARKED_SCHEMA_VERSION,
    audit_recorded_path,
    analyze_dense_path,
    interpolated_arm_states,
    is_keyframe_motion,
    is_marked_motion,
    marker_segment_ranges,
    validate_collection_trajectory,
    playback_time_scale,
    load_recorded_motion,
    validate_recorded_motion,
)
from openarm_kinesthetic_recorder.recorded_motion_player import (
    RecordedMotionPlayer,
    build_dense_arm_trajectory,
    build_replay_trajectories,
    parse_arguments,
    replay_provenance,
    verify_home_provenance,
)
from openarm_kinesthetic_recorder import recorded_motion_player as player_module


def _payload():
    first = [*Q_LIFTED_HOME_RIGHT, -0.10]
    second = list(first)
    second[0] += 0.05
    second[7] = -0.35
    return {
        "schema_version": 1,
        "kind": KIND,
        "joint_names": JOINT_NAMES,
        "required_start_home_positions_rad": list(Q_LIFTED_HOME_RIGHT),
        "duration_s": 1.0,
        "sample_count": 2,
        "samples": [
            {"time_from_start_s": 0.0, "positions_rad": first},
            {"time_from_start_s": 1.0, "positions_rad": second},
        ],
    }


def _marked_payload():
    positions = [[*Q_LIFTED_HOME_RIGHT, -0.35] for _ in range(4)]
    positions[1][0] += 0.01
    positions[2][0] += 0.02
    positions[3][0] += 0.03
    samples = [
        {
            "sample_index": index,
            "time_from_start_s": 0.02 * index,
            "source_timestamp_s": 1000.0 + 0.02 * index,
            "positions_rad": value,
        }
        for index, value in enumerate(positions)
    ]
    markers = [
        {
            "marker_index": 0,
            "type": "keypoint",
            "sample_index": 1,
            "time_from_start_s": 0.021,
            "source_timestamp_s": 1000.021,
            "source_button": "LB",
            "binding_error_s": 0.001,
        },
        {
            "marker_index": 1,
            "type": "gripper_close",
            "sample_index": 2,
            "time_from_start_s": 0.041,
            "source_timestamp_s": 1000.041,
            "source_button": "Y",
            "binding_error_s": 0.001,
            "target_position_rad": GRIPPER_TARGETS["gripper_close"],
            "result": "requested",
        },
    ]
    return {
        "schema_version": MARKED_SCHEMA_VERSION,
        "kind": MARKED_KIND,
        "mode": "continuous_marked",
        "arm": "right",
        "joint_names": JOINT_NAMES,
        "required_start_home_positions_rad": list(Q_LIFTED_HOME_RIGHT),
        "sample_period_target_s": 0.02,
        "marker_binding_tolerance_s": 0.25,
        "maximum_allowed_sample_gap_s": 0.25,
        "duration_s": 0.06,
        "samples": samples,
        "markers": markers,
        "quality": {
            "accepted_sample_count": 4,
            "rejected_incomplete_count": 0,
            "rejected_nonfinite_count": 0,
            "rejected_invalid_timestamp_count": 0,
            "timestamp_regression_count": 0,
            "estimated_dropped_sample_count": 0,
            "maximum_sample_gap_s": 0.02,
        },
    }


def test_recorded_motion_round_trip_and_dense_validation(tmp_path) -> None:
    path = tmp_path / "motion.json"
    path.write_text(json.dumps(_payload()), encoding="utf-8")
    payload = load_recorded_motion(path)
    dense = list(interpolated_arm_states(payload["samples"], 0.02))
    assert len(dense) == 4
    assert dense[0][1] == 0.0
    assert dense[-1][1] == pytest.approx(1.0)


def test_recorded_motion_rejects_wrong_home_and_non_monotonic_time() -> None:
    payload = _payload()
    payload["required_start_home_positions_rad"][0] += 0.1
    with pytest.raises(ValueError, match="fixed lifted HOME"):
        validate_recorded_motion(payload)


def test_keyframe_motion_reuses_schema_and_validates_semantic_events() -> None:
    payload = _payload()
    payload["mode"] = "keyframe"
    payload["waypoint_count"] = 2
    for index, sample in enumerate(payload["samples"]):
        sample["waypoint_index"] = index
        sample["gripper_event"] = "close" if index == 1 else "none"
        sample["timestamp_s"] = 1000.0 + index
    assert is_keyframe_motion(validate_recorded_motion(payload))
    payload["samples"][1]["gripper_event"] = "invalid"
    with pytest.raises(ValueError, match="invalid gripper_event"):
        validate_recorded_motion(payload)

    payload = _payload()
    payload["samples"][1]["time_from_start_s"] = 0.0
    with pytest.raises(ValueError, match="strictly increasing"):
        validate_recorded_motion(payload)


def test_replay_builds_synchronized_arm_and_gripper_trajectories() -> None:
    arm, gripper = build_replay_trajectories(_payload(), start_delay_s=0.2)
    assert arm.joint_names == JOINT_NAMES[:7]
    assert gripper.joint_names == [JOINT_NAMES[7]]
    assert len(arm.points) == len(gripper.points) == 2
    assert arm.points[0].time_from_start.nanosec == 200_000_000
    assert list(gripper.points[-1].positions) == [-0.35]


def test_replay_stretches_fast_sample_segments_without_speeding_slow_ones() -> None:
    payload = _payload()
    payload["samples"][-1]["time_from_start_s"] = 0.1
    payload["duration_s"] = 0.1
    scale = playback_time_scale(payload["samples"])
    assert scale == pytest.approx(3.125)
    arm, gripper = build_replay_trajectories(payload, time_scale=scale)
    assert arm.points[-1].time_from_start.sec == 0
    assert arm.points[-1].time_from_start.nanosec == 512_500_000
    assert gripper.points[-1].time_from_start == arm.points[-1].time_from_start


def test_playback_requires_successful_fixed_home_report(tmp_path) -> None:
    report = tmp_path / "home.json"
    report.write_text(
        json.dumps(
            {
                "state": "HOME_READY",
                "success": True,
                "home_goal_positions_rad": list(Q_LIFTED_HOME_RIGHT),
            }
        ),
        encoding="utf-8",
    )
    assert verify_home_provenance(report)["success"] is True
    report.write_text(json.dumps({"state": "FAIL_RIGID", "success": False}))
    with pytest.raises(RuntimeError, match="not HOME_READY"):
        verify_home_provenance(report)


def test_schema_v2_round_trip_and_unknown_version_rejected(tmp_path) -> None:
    path = tmp_path / "marked.json"
    path.write_text(json.dumps(_marked_payload()), encoding="utf-8")
    assert is_marked_motion(load_recorded_motion(path))
    invalid = _marked_payload()
    invalid["schema_version"] = 99
    with pytest.raises(ValueError, match="unsupported.*version"):
        validate_recorded_motion(invalid)


def test_schema_v2_accepts_semantic_only_gripper_events_for_replay() -> None:
    payload = _marked_payload()
    payload["markers"][1]["result"] = "semantic_only"
    assert validate_recorded_motion(payload)["markers"][1]["result"] == "semantic_only"


def _collection_payload():
    payload = _marked_payload()
    payload["samples"][-1]["positions_rad"] = [*Q_LIFTED_HOME_RIGHT, -0.35]
    payload["markers"] = [
        {
            "marker_index": 0,
            "type": "gripper_close",
            "sample_index": 1,
            "time_from_start_s": 0.021,
            "source_timestamp_s": 1000.021,
            "source_button": "Y",
            "binding_error_s": 0.001,
            "target_position_rad": GRIPPER_TARGETS["gripper_close"],
            "result": "semantic_only",
        },
        {
            "marker_index": 1,
            "type": "gripper_open",
            "sample_index": 2,
            "time_from_start_s": 0.041,
            "source_timestamp_s": 1000.041,
            "source_button": "A",
            "binding_error_s": 0.001,
            "target_position_rad": GRIPPER_TARGETS["gripper_open"],
            "result": "semantic_only",
        },
    ]
    return payload


def test_collection_trajectory_requires_two_markers_and_terminal_home() -> None:
    payload = _collection_payload()
    summary = validate_collection_trajectory(payload)
    assert summary["marker_count"] == 2
    assert summary["end_home_max_error_rad"] == pytest.approx(0.0)

    payload["samples"][-1]["positions_rad"][3] += 0.031
    with pytest.raises(ValueError, match=r"joint=J4.*error=0.031000"):
        validate_collection_trajectory(payload)

    payload = _collection_payload()
    payload["markers"][1]["source_button"] = "Y"
    with pytest.raises(ValueError, match="exactly"):
        validate_collection_trajectory(payload)


def test_collection_save_and_replay_share_original_rate_velocity_limit() -> None:
    payload = _collection_payload()
    payload["samples"][1]["positions_rad"][6] = Q_LIFTED_HOME_RIGHT[6] + 0.022
    summary = validate_collection_trajectory(payload)
    assert summary["maximum_recorded_velocity_rad_s"][6] == pytest.approx(1.1)
    assert COLLECTION_REPLAY_VELOCITY_LIMITS_RAD_S[6] == pytest.approx(20.943946)

    payload = _collection_payload()
    payload["samples"][1]["positions_rad"][2] = Q_LIFTED_HOME_RIGHT[2] + 0.12
    with pytest.raises(ValueError, match="original-rate replay velocity"):
        validate_collection_trajectory(payload)


def test_collection_replay_corrects_measured_fixed_home_before_completion() -> None:
    target = list(Q_LIFTED_HOME_RIGHT)
    actual = list(target)
    actual[3] -= 0.041
    planned = []
    waited = []
    node = SimpleNamespace(
        all_positions=dict(zip(JOINT_NAMES[:7], actual)),
        wait_for_full_joint_state=lambda: None,
        verify_guard=lambda _report: None,
        plan_segment=lambda start, goal, velocity, acceleration: (
            planned.append((start, goal, velocity, acceleration)) or "trajectory",
            SimpleNamespace(duration_s=2.0, point_count=21),
        ),
        verify_controller_active=lambda *_args: None,
        execute_same_trajectory=lambda *_args, **_kwargs: {
            "action_status": GoalStatus.STATUS_SUCCEEDED,
            "moveit_error_code": MoveItErrorCodes.SUCCESS,
        },
        wait_for_arm_target=lambda goal, **_kwargs: (
            waited.append(goal),
            node.all_positions.update(zip(JOINT_NAMES[:7], goal)),
        ),
        publish_reference=False,
        get_logger=lambda: SimpleNamespace(warn=lambda _message: None),
    )
    RecordedMotionPlayer.ensure_collection_fixed_home(node, {})
    assert len(planned) == 1
    assert planned[0][2:] == (0.10, 0.10)
    assert waited == [target]
    assert node.publish_reference is False


def test_replay_provenance_hashes_exact_trajectory_and_marks_full_range(tmp_path) -> None:
    payload = validate_recorded_motion(_collection_payload())
    path = tmp_path / "motion.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    provenance = replay_provenance(
        path, payload, 1.0, marker_segment_ranges(payload)
    )
    assert provenance["trajectory_path"] == str(path.resolve())
    assert len(provenance["trajectory_sha256"]) == 64
    assert provenance["full_replay"] is True
    assert [item["type"] for item in provenance["marker_summary"]] == [
        "gripper_close",
        "gripper_open",
    ]


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda p: p["samples"][1].update(sample_index=3), "sample_index"),
        (lambda p: p["samples"][1].update(time_from_start_s=0.0), "strictly increasing"),
        (lambda p: p["samples"][1]["positions_rad"].__setitem__(0, float("nan")), "finite"),
        (lambda p: p["samples"][1].update(positions_rad=[0.0]), "dimension"),
        (lambda p: p["markers"][0].update(sample_index=99), "invalid_sample_reference"),
        (lambda p: p["markers"][1].update(sample_index=0), "order is reversed"),
    ],
)
def test_schema_v2_rejects_corrupt_indices_times_values_and_markers(mutate, message) -> None:
    payload = _marked_payload()
    mutate(payload)
    with pytest.raises(ValueError, match=message):
        validate_recorded_motion(payload)


def test_dense_dynamics_stretches_time_without_changing_samples() -> None:
    payload = _marked_payload()
    before = json.dumps(payload["samples"], sort_keys=True)
    dynamics = analyze_dense_path(
        payload["samples"], velocity_limits_rad_s=[0.1] * 7,
        acceleration_limits_rad_s2=[0.5] * 7,
        velocity_scaling=0.5, acceleration_scaling=0.5,
    )
    assert dynamics.time_scale > 1.0
    assert dynamics.duration_s > payload["duration_s"]
    assert json.dumps(payload["samples"], sort_keys=True) == before


def test_recorded_path_audit_preserves_selected_range_timing() -> None:
    payload = _marked_payload()
    audit = audit_recorded_path(payload["samples"][1:3])
    assert audit.duration_s == pytest.approx(0.02)
    assert audit.maximum_velocity_rad_s[0] == pytest.approx(0.5)
    assert audit.maximum_sample_gap_s == pytest.approx(0.02)


def test_marked_validation_checks_only_selected_samples_at_recorded_rate() -> None:
    payload = validate_recorded_motion(_marked_payload())
    checked = []
    node = SimpleNamespace(
        verify_guard=lambda _report: None,
        all_positions=dict(zip(JOINT_NAMES[:7], payload["samples"][0]["positions_rad"][:7])),
        validate_state=lambda state, label: checked.append((state, label)),
        operator_interrupt_requested=False,
        software_stop_latched=False,
    )
    audit, count = RecordedMotionPlayer.validate_marked_path(
        node,
        payload,
        {},
        1.0,
        0.25,
        0.20,
        0,
        1,
    )
    assert audit.duration_s == pytest.approx(0.02)
    assert count == 2
    assert len(checked) == 2


def test_marked_recorded_timing_rejects_velocity_above_hard_guard() -> None:
    payload = validate_recorded_motion(_marked_payload())
    payload["samples"][1]["positions_rad"][2] += 0.12
    node = SimpleNamespace(
        verify_guard=lambda _report: None,
        all_positions=dict(
            zip(JOINT_NAMES[:7], payload["samples"][0]["positions_rad"][:7])
        ),
        validate_state=lambda _state, _label: None,
        operator_interrupt_requested=False,
        software_stop_latched=False,
    )
    with pytest.raises(RuntimeError, match="exceeds velocity guard"):
        RecordedMotionPlayer.validate_marked_path(
            node,
            payload,
            {},
            1.0,
            0.25,
            0.20,
            0,
            1,
        )


def test_marked_playback_rate_defaults_to_recorded_time_and_rejects_speedup() -> None:
    args = parse_arguments(["player", "--trajectory", "motion.json", "--plan-only"])
    assert args.playback_rate == pytest.approx(1.0)
    with pytest.raises(SystemExit):
        parse_arguments(
            [
                "player",
                "--trajectory",
                "motion.json",
                "--plan-only",
                "--playback-rate",
                "1.01",
            ]
        )


def test_dense_path_reports_gap_and_position_step_with_sample_context() -> None:
    payload = _marked_payload()
    with pytest.raises(ValueError, match=r"sample_index=1 reason=sample_gap"):
        analyze_dense_path(payload["samples"], maximum_gap_s=0.01)
    with pytest.raises(ValueError, match=r"sample_index=1.*reason=position_step"):
        analyze_dense_path(payload["samples"], maximum_step_rad=0.005)


def test_marker_segments_and_trajectories_preserve_every_dense_sample() -> None:
    payload = validate_recorded_motion(_marked_payload())
    ranges = marker_segment_ranges(payload)
    assert [(r["start_sample_index"], r["end_sample_index"]) for r in ranges] == [
        (0, 1), (1, 2), (2, 3)
    ]
    trajectories = [
        build_dense_arm_trajectory(
            payload["samples"], r["start_sample_index"], r["end_sample_index"],
            time_scale=2.0,
        )
        for r in ranges
    ]
    reconstructed = [list(trajectories[0].points[0].positions)]
    for trajectory in trajectories:
        reconstructed.extend(list(point.positions) for point in trajectory.points[1:])
    assert reconstructed == [sample["positions_rad"][:7] for sample in payload["samples"]]


def _joint_state(names, positions, stamp):
    seconds = int(stamp)
    return SimpleNamespace(
        name=names,
        position=positions,
        header=SimpleNamespace(
            stamp=SimpleNamespace(sec=seconds, nanosec=int((stamp - seconds) * 1e9))
        ),
    )


def test_player_cache_never_stitches_partial_or_frozen_joint_state_frames() -> None:
    node = SimpleNamespace(
        total_joint_state_messages=0,
        last_message_received_at=None,
        complete_joint_state_messages=0,
        consecutive_complete_samples=0,
        last_missing_joint_names=list(JOINT_NAMES),
        last_complete_source_timestamp_s=None,
        all_positions={},
        all_received_at=None,
        gripper_reference=None,
    )
    complete = _joint_state(JOINT_NAMES, list(range(8)), 100.0)
    RecordedMotionPlayer.on_all_joint_state(node, complete)
    assert node.consecutive_complete_samples == 1
    assert node.all_positions[JOINT_NAMES[-1]] == 7.0

    partial = _joint_state(JOINT_NAMES[:-1], list(range(7)), 100.1)
    RecordedMotionPlayer.on_all_joint_state(node, partial)
    assert node.consecutive_complete_samples == 0
    assert node.complete_joint_state_messages == 1
    assert node.all_positions[JOINT_NAMES[-1]] == 7.0

    frozen = _joint_state(JOINT_NAMES, [10.0] * 8, 100.0)
    RecordedMotionPlayer.on_all_joint_state(node, frozen)
    assert node.consecutive_complete_samples == 0
    assert node.complete_joint_state_messages == 1
    assert node.all_positions[JOINT_NAMES[0]] == 0.0


def test_player_startup_gate_accepts_publisher_delayed_three_seconds(monkeypatch) -> None:
    class _Clock:
        value = 0.0

        def __call__(self):
            self.value += 0.05
            return self.value

    clock = _Clock()
    node = SimpleNamespace(
        joint_state_startup_timeout_s=10.0,
        joint_state_freshness_s=0.5,
        joint_state_required_consecutive_samples=3,
        total_joint_state_messages=0,
        last_message_received_at=None,
        complete_joint_state_messages=0,
        consecutive_complete_samples=0,
        last_missing_joint_names=list(JOINT_NAMES),
        last_complete_source_timestamp_s=None,
        all_positions={},
        all_received_at=None,
        gripper_reference=None,
    )
    delivered = {"count": 0}

    def spin_once(_node, timeout_sec):
        del timeout_sec
        if clock.value >= 3.0 and delivered["count"] < 3:
            delivered["count"] += 1
            RecordedMotionPlayer.on_all_joint_state(
                node,
                _joint_state(
                    JOINT_NAMES, [0.0] * 8, 100.0 + delivered["count"] * 0.02
                ),
            )

    monkeypatch.setattr(player_module.time, "monotonic", clock)
    monkeypatch.setattr(player_module.rclpy, "spin_once", spin_once)
    RecordedMotionPlayer.wait_for_full_joint_state(node)
    assert delivered["count"] == 3
    assert node.consecutive_complete_samples == 3


@pytest.mark.parametrize(
    ("publisher_count", "total", "complete", "consecutive", "reason"),
    [
        (0, 0, 0, 0, "no_publisher"),
        (1, 2, 0, 0, "incomplete_or_invalid_messages"),
        (1, 4, 1, 0, "timestamp_not_increasing_or_incomplete"),
    ],
)
def test_player_startup_diagnostic_distinguishes_failure_modes(
    monkeypatch, publisher_count, total, complete, consecutive, reason
) -> None:
    monkeypatch.setattr(player_module.time, "monotonic", lambda: 10.0)
    node = SimpleNamespace(
        last_message_received_at=None if total == 0 else 9.9,
        all_received_at=None if complete == 0 else 9.8,
        joint_state_freshness_s=0.5,
        joint_state_startup_timeout_s=10.0,
        joint_state_required_consecutive_samples=3,
        total_joint_state_messages=total,
        complete_joint_state_messages=complete,
        consecutive_complete_samples=consecutive,
        last_missing_joint_names=[JOINT_NAMES[-1]] if complete == 0 else [],
        count_publishers=lambda _topic: publisher_count,
    )
    diagnostic = RecordedMotionPlayer.joint_state_gate_diagnostic(node)
    assert f"reason={reason}" in diagnostic
    assert "subscriber_qos=sensor_data" in diagnostic
