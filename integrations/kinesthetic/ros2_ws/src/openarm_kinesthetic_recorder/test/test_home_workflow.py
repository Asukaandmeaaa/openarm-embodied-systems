from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import MethodType, SimpleNamespace
import json
import sys
import time

import pytest
from launch import LaunchContext
from launch.actions import GroupAction, OpaqueFunction, RegisterEventHandler
from launch_ros.actions import Node

from openarm_kinesthetic_recorder import home_guarded_executor as home
from openarm_kinesthetic_recorder.real_episode_recorder import (
    JOINT_NAMES,
    RealEpisodeRecorderNode,
    demonstrated_action,
    jpeg_dimensions,
    load_home_provenance,
    ordered_joint_positions,
    validate_trial_metadata,
)
from openarm_kinesthetic.recording import EpisodeRecorder, audit_real_episode_directory
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from action_msgs.msg import GoalStatus
from moveit_msgs.msg import MoveItErrorCodes, RobotState, RobotTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint


PACKAGE = Path(__file__).resolve().parents[1]


def test_home_config_is_fixed_lifted_home_and_opens_j8_afterward() -> None:
    config = home.load_home_config(PACKAGE / "config/home_initialization.yaml")

    assert config["right_home_joint_positions_rad"] == pytest.approx(
        [
            -0.1450262670,
            -0.0151076284,
            -0.0060887956,
            1.3751977241,
            -0.0031428430,
            -0.1984151663,
            -0.1222898194,
        ]
    )
    assert len(config["right_joint_names"]) == 7
    assert all("finger" not in name for name in config["right_joint_names"])
    assert config["gripper_policy"] == "open_after_arm_home"
    assert config["home_gripper"]["joint_name"] == home.RIGHT_GRIPPER_JOINT_NAME
    assert config["home_gripper"]["open_position_rad"] == pytest.approx(-0.35)
    assert config["planning"]["velocity_scaling"] == pytest.approx(0.70)
    assert config["planning"]["acceleration_scaling"] == pytest.approx(0.50)
    assert config["final_acceptance"]["stable_duration_s"] == pytest.approx(0.20)
    assert config["static_collision"]["dimensions_xyz_m"] == [0.50, 0.70, 0.50]
    assert "plan_only_authorized" not in config["static_collision"]
    assert "plan_only_authorized" not in config["station_frame"]
    assert config["side_prehome"]["path_quality_gates"] == {
        "minimum_clearance_m": 0.002,
        "maximum_backward_excursion_m": 0.20,
        "maximum_lateral_excursion_m": 0.60,
        "maximum_vertical_excursion_m": 0.60,
        "maximum_absolute_j4_rad": 2.20,
    }


def test_stage_a_transit_tolerance_accepts_observed_error_without_relaxing_home(
    monkeypatch,
) -> None:
    config = {
        "planning": {"service_timeout_s": 1.0},
        "final_acceptance": {
            "joint_tolerance_rad": 0.03,
            "settle_timeout_s": 0.6,
            "stable_duration_s": 0.2,
        },
    }
    node = SimpleNamespace(
        refresh_rigid_state=lambda _timeout: [0.03070413609413225] + [0.0] * 6
    )

    ticks = iter(index * 0.1 for index in range(100))
    monkeypatch.setattr(home.time, "monotonic", lambda: next(ticks))
    final_home = home.wait_for_home_settle(node, [0.0] * 7, config)
    assert final_home["accepted"] is False
    assert final_home["joint_tolerance_rad"] == pytest.approx(0.03)

    ticks = iter(index * 0.1 for index in range(100))
    monkeypatch.setattr(home.time, "monotonic", lambda: next(ticks))
    transit = home.wait_for_home_settle(
        node,
        [0.0] * 7,
        config,
        joint_tolerance_rad=home.STAGE_A_TRANSIT_JOINT_TOLERANCE_RAD,
    )
    assert transit["accepted"] is True
    assert transit["joint_tolerance_rad"] == pytest.approx(0.04)


def test_invalid_home_config_rejects_a_gripper_goal(tmp_path) -> None:
    source = (PACKAGE / "config/home_initialization.yaml").read_text()
    source = source.replace(
        "right_home_joint_positions_rad: [-0.1450262670, -0.0151076284, -0.0060887956, 1.3751977241, -0.0031428430, -0.1984151663, -0.1222898194]",
        "right_home_joint_positions_rad: [0, 0, 0, 1.57, 0, 0, 0, -0.35]",
    )
    path = tmp_path / "bad.yaml"
    path.write_text(source)

    with pytest.raises(ValueError, match="seven finite"):
        home.load_home_config(path)


def test_process_arguments_strip_ros_launch_arguments(monkeypatch) -> None:
    monkeypatch.setattr(home, "get_package_share_directory", lambda _name: str(PACKAGE))
    args = home.parse_process_arguments(
        [
            "home_guarded_executor",
            "--allow-real-execution",
            "--ros-args",
            "-r",
            "__node:=home_guarded_executor",
        ]
    )
    assert args.allow_real_execution is True


def test_fixed_collision_snapshot_requires_exact_pose_and_all_dimensions() -> None:
    configured = home.load_home_config(PACKAGE / "config/home_initialization.yaml")[
        "static_collision"
    ]
    snapshot = {
        "id": configured["object_id"],
        "frame_id": configured["frame_id"],
        "dimensions_m": list(configured["dimensions_xyz_m"]),
        "object_pose": {
            "position_m": [0.0, 0.0, 0.0],
            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
        "primitive_pose": {
            "position_m": list(configured["center_xyz_m"]),
            "orientation_xyzw": list(configured["orientation_xyzw"]),
        },
    }
    home.verify_static_collision_snapshot(snapshot, configured)
    snapshot["dimensions_m"][2] -= 0.02
    with pytest.raises(RuntimeError, match="dimensions_m"):
        home.verify_static_collision_snapshot(snapshot, configured)


def test_fixed_collision_accepts_moveit_normalized_object_pose() -> None:
    configured = home.load_home_config(PACKAGE / "config/home_initialization.yaml")[
        "static_collision"
    ]
    snapshot = {
        "id": configured["object_id"],
        "frame_id": configured["frame_id"],
        "dimensions_m": list(configured["dimensions_xyz_m"]),
        "object_pose": {
            "position_m": list(configured["center_xyz_m"]),
            "orientation_xyzw": list(configured["orientation_xyzw"]),
        },
        "primitive_pose": {
            "position_m": [0.0, 0.0, 0.0],
            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
    }

    home.verify_static_collision_snapshot(snapshot, configured)
    assert home.effective_primitive_pose(snapshot)["position_m"] == pytest.approx(
        configured["center_xyz_m"]
    )


def test_real_recorder_contract_keeps_j8_in_both_arms() -> None:
    assert len(JOINT_NAMES) == 16
    assert JOINT_NAMES[7] == "openarm_right_finger_joint1"
    assert JOINT_NAMES[15] == "openarm_left_finger_joint1"


def test_real_recorder_orders_both_j8_and_parks_left_action() -> None:
    message = JointState()
    message.name = list(reversed(JOINT_NAMES))
    positions_by_name = {name: float(index) for index, name in enumerate(JOINT_NAMES)}
    message.position = [positions_by_name[name] for name in message.name]

    measured = ordered_joint_positions(message)
    assert measured == [float(index) for index in range(16)]
    assert demonstrated_action(measured, [80.0 + index for index in range(8)]) == [
        *[float(index) for index in range(8)],
        *[80.0 + index for index in range(8)],
    ]


def test_recorder_requires_a_fresh_successful_home_report(tmp_path, monkeypatch) -> None:
    path = tmp_path / "home.json"
    payload = {
        "workflow": "FIXED_LIFTED_HOME",
        "state": "HOME_READY",
        "success": True,
        "j8_policy": "open_after_arm_home",
        "j8_commanded_by_initialization": True,
        "home_gripper": {
            "accepted": True,
            "target_position_rad": -0.35,
            "measured_position_rad": -0.34,
        },
        "completed_wall_time_s": 1000.0,
        "trajectory_sha256": "trajectory",
        "config_sha256": "config",
        "home_goal_positions_rad": list(home.FIXED_LIFTED_HOME),
    }
    path.write_text(__import__("json").dumps(payload))
    monkeypatch.setattr(
        "openarm_kinesthetic_recorder.real_episode_recorder.time.time",
        lambda: 1001.0,
    )
    provenance = load_home_provenance(path, 300.0)
    assert provenance["trajectory_sha256"] == "trajectory"
    payload["success"] = False
    path.write_text(__import__("json").dumps(payload))
    with pytest.raises(RuntimeError, match="HOME_READY"):
        load_home_provenance(path, 300.0)


def test_episode_start_gate_uses_lifted_target_from_home_report(tmp_path) -> None:
    now = time.time()
    target = list(home.FIXED_LIFTED_HOME)
    node = SimpleNamespace(
        active=False,
        sources_ready=lambda: (True, "sources ready"),
        passive=False,
        latest_passive_wall_s=now,
        count_publishers=lambda _topic: 1,
        maximum_age=1.0,
        latest_state=[*target, -0.2, *([0.0] * 8)],
        home_provenance={"home_goal_positions_rad": target},
        maximum_episode_start_home_error=0.06,
        recorder=EpisodeRecorder(tmp_path),
        parked_left_action=None,
        camera_index={},
        trial_metadata={
            "batch_id": "pilot_01",
            "trial_id": "trial_p04_001",
            "position_id": "P04",
            "cube_forward_m": 0.0,
            "cube_lateral_m": 0.0,
            "placement_measurement_method": "placement_mat",
            "plan_only_passed": True,
            "dry_replay_passed": True,
        },
    )
    response = SimpleNamespace(success=False, message="")

    RealEpisodeRecorderNode.start(node, None, response)

    assert response.success is True
    metadata = json.loads((node.recorder.active / "metadata.json").read_text())
    assert metadata["episode_home"] == "fixed_lifted_home"
    assert metadata["episode_start_home_error_rad"] == pytest.approx(0.0)
    assert metadata["source"] == "real_right_continuous_marked_replay"
    assert metadata["position_id"] == "P04"


def test_episode_directory_uses_numbered_trajectory_trial_id(tmp_path) -> None:
    now = time.time()
    target = list(home.FIXED_LIFTED_HOME)
    trial_id = "20260928_140008_001"
    node = SimpleNamespace(
        active=False,
        sources_ready=lambda: (True, "sources ready"),
        passive=False,
        latest_passive_wall_s=now,
        count_publishers=lambda _topic: 1,
        maximum_age=1.0,
        latest_state=[*target, -0.35, *([0.0] * 8)],
        home_provenance={"home_goal_positions_rad": target},
        maximum_episode_start_home_error=0.06,
        recorder=EpisodeRecorder(tmp_path),
        trial_metadata={
            "batch_id": "fixed_01", "trial_id": trial_id,
            "position_id": "fixed_01", "cube_forward_m": 0.0,
            "cube_lateral_m": 0.0,
            "placement_measurement_method": "fixed_marked_origin",
            "plan_only_passed": True, "dry_replay_passed": True,
        },
    )
    response = SimpleNamespace(success=False, message="")
    RealEpisodeRecorderNode.start(node, None, response)
    assert response.success
    assert node.recorder.episode_id == trial_id
    assert (tmp_path / ".transactions" / trial_id).is_dir()


def test_episode_start_requires_live_hardware_passive_publisher(tmp_path) -> None:
    node = SimpleNamespace(
        active=False,
        sources_ready=lambda: (True, "sources ready"),
        passive=False,
        latest_passive_wall_s=time.time() - 30.0,
        count_publishers=lambda _topic: 0,
        maximum_age=1.0,
        trial_metadata={"trial_id": "test"},
    )
    response = SimpleNamespace(success=False, message="")
    RealEpisodeRecorderNode.start(node, None, response)
    assert not response.success
    assert "live hardware publisher" in response.message


def test_auto_review_service_uses_pending_review_without_operator_success() -> None:
    calls = []
    node = SimpleNamespace(
        _finish_review=lambda response, *, operator_confirmed: (
            calls.append(operator_confirmed) or response
        )
    )
    response = SimpleNamespace(success=False, message="")
    assert RealEpisodeRecorderNode.finish_auto_review(node, None, response) is response
    assert calls == [False]


def test_trial_metadata_requires_position_identity_and_completed_preflight() -> None:
    payload = {
        "batch_id": "pilot_01",
        "trial_id": "trial_p04_001",
        "position_id": "P04",
        "cube_forward_m": 0,
        "cube_lateral_m": 0.0,
        "placement_measurement_method": "placement_mat",
        "plan_only_passed": True,
        "dry_replay_passed": True,
    }
    assert validate_trial_metadata(payload)["cube_forward_m"] == 0.0
    payload["source_trajectory_trial_id"] = "20261008_091207_006"
    assert validate_trial_metadata(payload)["source_trajectory_trial_id"] == "20261008_091207_006"
    payload["source_trajectory_trial_id"] = "invalid"
    with pytest.raises(ValueError, match="source_trajectory_trial_id"):
        validate_trial_metadata(payload)
    payload.pop("source_trajectory_trial_id")
    payload["dry_replay_passed"] = False
    with pytest.raises(ValueError, match="dry_replay_passed"):
        validate_trial_metadata(payload)


def test_jpeg_dimensions_reads_sof_without_decoding() -> None:
    jpeg = (
        b"\xff\xd8"
        b"\xff\xc0\x00\x0b\x08\x02\xd0\x05\x00\x01\x01\x11\x00"
        b"\xff\xd9"
    )
    assert jpeg_dimensions(jpeg) == [1280, 720]
    assert jpeg_dimensions(b"not jpeg") is None


def _replay_status_payload(state="REPLAY_COMPLETE", playback_rate=1.0):
    return {
        "schema_version": 1,
        "state": state,
        "timestamp_s": 1001.0,
        "trajectory_path": "/tmp/p04.json",
        "trajectory_sha256": "a" * 64,
        "trajectory_schema_version": 2,
        "trajectory_mode": "continuous_marked",
        "trajectory_duration_s": 12.0,
        "playback_rate": playback_rate,
        "full_replay": True,
        "marker_summary": [
            {
                "marker_index": 0,
                "type": "gripper_close",
                "sample_index": 10,
                "source_button": "Y",
                "result": "semantic_only",
            },
            {
                "marker_index": 1,
                "type": "gripper_open",
                "sample_index": 20,
                "source_button": "A",
                "result": "semantic_only",
            },
        ],
    }


def test_replay_status_atomically_enriches_active_episode(tmp_path) -> None:
    recorder = EpisodeRecorder(tmp_path)
    recorder.begin({}, "active")
    node = SimpleNamespace(
        active=True,
        recorder=recorder,
        active_replay=None,
        active_replay_complete=False,
        abort_active=lambda reason: pytest.fail(reason),
    )
    RealEpisodeRecorderNode.on_replay_status(
        node, String(data=json.dumps(_replay_status_payload()))
    )
    metadata = json.loads((recorder.active / "metadata.json").read_text())
    assert node.active_replay_complete is True
    assert metadata["trajectory_sha256"] == "a" * 64
    assert metadata["replay_complete"] is True


def test_replay_status_uses_active_source_id_after_trial_metadata_cleared(tmp_path) -> None:
    recorder = EpisodeRecorder(tmp_path)
    recorder.begin({}, "20261008_131712_016")
    failures = []
    node = SimpleNamespace(
        active=True,
        recorder=recorder,
        trial_metadata=None,
        active_trial_id="20261008_131712_016",
        active_source_trajectory_trial_id="20261008_090909_002",
        active_replay=None,
        active_replay_complete=False,
        abort_active=failures.append,
    )
    payload = _replay_status_payload()
    payload["trajectory_trial_id"] = "20261008_090909_002"
    RealEpisodeRecorderNode.on_replay_status(node, String(data=json.dumps(payload)))
    assert failures == []
    assert node.active_replay_complete is True
    payload["trajectory_trial_id"] = "20261008_091000_003"
    RealEpisodeRecorderNode.on_replay_status(node, String(data=json.dumps(payload)))
    assert failures and "trial_id does not match" in failures[0]


def test_replay_status_rejects_non_original_rate(tmp_path) -> None:
    recorder = EpisodeRecorder(tmp_path)
    recorder.begin({}, "active")
    failures = []
    node = SimpleNamespace(
        active=True,
        recorder=recorder,
        active_replay=None,
        active_replay_complete=False,
        abort_active=failures.append,
    )
    RealEpisodeRecorderNode.on_replay_status(
        node,
        String(data=json.dumps(_replay_status_payload(playback_rate=0.5))),
    )
    assert failures and "full-rate" in failures[0]


def test_episode_success_gate_commits_only_complete_evidence(tmp_path) -> None:
    trajectory = tmp_path / "trajectory.json"
    trajectory.write_text("{}", encoding="utf-8")
    digest = __import__("hashlib").sha256(trajectory.read_bytes()).hexdigest()
    recorder = EpisodeRecorder(tmp_path)
    marker_summary = _replay_status_payload()["marker_summary"]
    recorder.begin(
        {
            "source": "real_right_continuous_marked_replay",
            "batch_id": "pilot_01",
            "trial_id": "p04_001",
            "position_id": "P04",
            "cube_forward_m": 0.0,
            "cube_lateral_m": 0.0,
            "placement_measurement_method": "placement_mat",
            "trajectory_path": str(trajectory),
            "trajectory_sha256": digest,
            "marker_summary": marker_summary,
            "playback_rate": 1.0,
            "replay_complete": True,
            "full_replay": True,
            "home_initialization": {"home_goal_positions_rad": list(home.FIXED_LIFTED_HOME)},
        },
        "active",
    )
    target = list(home.FIXED_LIFTED_HOME)
    positions = [*target, -0.35, *([0.0] * 8)]
    recorder.append("joint_observation", 1.0, {"positions": positions, "order": JOINT_NAMES})
    recorder.append("joint_observation", 2.0, {"positions": positions, "order": JOINT_NAMES})
    action = {
        "positions": positions,
        "order": JOINT_NAMES,
        "source": "recorded_motion_player_controller_reference",
    }
    recorder.append("demonstration_action", 1.1, action)
    recorder.append("demonstration_action", 1.9, action)
    recorder.append("gripper_event", 1.2, {"event": "close"})
    recorder.append("gripper_event", 1.8, {"event": "open"})
    for key in ("ceiling", "wrist_right"):
        relative = Path("cameras") / key / "00000000.jpg"
        image = recorder.active / relative
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(b"jpeg")
        recorder.append("camera", 1.0, {"key": key, "path": str(relative)})
        relative = Path("cameras") / key / "00000001.jpg"
        image = recorder.active / relative
        image.write_bytes(b"jpeg")
        recorder.append("camera", 2.0, {"key": key, "path": str(relative)})
    node = SimpleNamespace(
        active=True,
        active_replay_complete=True,
        observation_count=2,
        reference_count=2,
        gripper_events=["close", "open"],
        camera_stats={
            key: {
                "count": 2,
                "first_timestamp_s": 1.0,
                "last_timestamp_s": 2.0,
                "resolutions": {(1280, 720)},
            }
            for key in ("ceiling", "wrist_right")
        },
        latest_state=positions,
        latest_state_wall_s=time.time(),
        maximum_age=1.0,
        home_provenance={"home_goal_positions_rad": target},
        recorder=recorder,
    )
    response = SimpleNamespace(success=False, message="")
    node._finish_review = MethodType(RealEpisodeRecorderNode._finish_review, node)
    RealEpisodeRecorderNode.finish_success(node, None, response)
    assert response.success is True
    destination = tmp_path / "episodes_pending_review" / "active"
    metadata = json.loads((destination / "metadata.json").read_text())
    assert metadata["episode_end_home_max_error_rad"] == pytest.approx(0.0)
    assert metadata["camera_observation"]["ceiling"]["observed_fps"] == 1.0
    assert metadata["automatic_episode_audit"]["passed"] is True
    assert audit_real_episode_directory(destination, JOINT_NAMES)["passed"] is True
    next((destination / "cameras" / "ceiling").glob("*.jpg")).unlink()
    audit = audit_real_episode_directory(destination, JOINT_NAMES)
    assert audit["passed"] is False
    assert any("missing/empty" in failure for failure in audit["failures"])


def _description():
    path = PACKAGE / "launch/nano_home_teaching.launch.py"
    spec = spec_from_file_location("nano_home_teaching_launch", path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    original = module.get_package_share_directory
    module.get_package_share_directory = lambda package: (
        str(PACKAGE) if package == "openarm_kinesthetic_recorder" else original(package)
    )
    return module.generate_launch_description()


def test_new_launch_has_no_table_detector_and_starts_home_only() -> None:
    description = _description()
    nodes = [entity for entity in description.entities if isinstance(entity, Node)]
    assert [node._Node__node_executable for node in nodes] == [
        "home_guarded_executor"
    ]
    assert all(node._Node__node_executable != "table_initializer" for node in nodes)
    camera_group = next(
        entity for entity in description.entities if isinstance(entity, GroupAction)
    )
    assert camera_group._GroupAction__forwarding is False


def test_new_launch_requires_explicit_motion_authorization() -> None:
    gate = next(
        entity for entity in _description().entities if isinstance(entity, OpaqueFunction)
    )
    context = LaunchContext()
    context.launch_configurations["allow_real_execution"] = "true"
    assert gate.execute(context) == []
    context.launch_configurations["allow_real_execution"] = "false"
    with pytest.raises(RuntimeError, match="requires allow_real_execution"):
        gate.execute(context)


def test_home_success_starts_teaching_cameras_and_recorder() -> None:
    handler = next(
        entity._RegisterEventHandler__event_handler
        for entity in _description().entities
        if isinstance(entity, RegisterEventHandler)
    )
    callback = handler._OnActionEventBase__on_event
    success = callback(SimpleNamespace(returncode=0), None)
    failure = callback(SimpleNamespace(returncode=2), None)
    assert [
        action._Node__node_executable
        for action in success
        if isinstance(action, Node)
    ] == [
        "wrist_camera",
        "real_gamepad",
            "right_kinesthetic_bridge",
            "real_episode_recorder",
            "right_motion_recorder",
            "teaching_status",
    ]
    assert not any(isinstance(action, Node) for action in failure)


def _run_home_main(monkeypatch, tmp_path, *, armed=True, failure=None):
    config_path = PACKAGE / "config/home_initialization.yaml"
    config = home.load_home_config(config_path)
    goal = config["right_home_joint_positions_rad"]
    prehome = [0.05, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0]
    configured_collision = config["static_collision"]
    commissioned = tmp_path / "commissioned.json"
    commissioned.write_text(
        json.dumps(
            {
                "approved_candidate_id": "test_side",
                "collision_guard_sha256": home._hash(configured_collision),
                "operator_approved": True,
            }
        )
    )
    config["side_prehome"].update(
        {
            "selected_joint_positions_rad": prehome,
            "selected_candidate_id": "test_side",
            "commissioned_report_path": str(commissioned),
            "commissioned_report_sha256": __import__("hashlib").sha256(
                commissioned.read_bytes()
            ).hexdigest(),
        }
    )
    snapshot = {
        "id": configured_collision["object_id"],
        "frame_id": configured_collision["frame_id"],
        "dimensions_m": list(configured_collision["dimensions_xyz_m"]),
        "object_pose": {
            "position_m": [0.0, 0.0, 0.0],
            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
        "primitive_pose": {
            "position_m": list(configured_collision["center_xyz_m"]),
            "orientation_xyzw": list(configured_collision["orientation_xyzw"]),
        },
    }
    instances = []

    class FakeNode:
        def __init__(self):
            self.execution_goal_sent = False
            self.gripper_goal_sent = False
            self.executed = []
            self.gripper_commands = []
            self.controller_checks = []
            self.guard_calls = 0
            self.refresh_calls = 0
            instances.append(self)

        def verify_controller_active(self, name, _timeout):
            self.controller_checks.append(name)
            return {"name": name, "state": "active", "type": "test"}

        def refresh_rigid_state(self, _timeout):
            self.refresh_calls += 1
            if self.refresh_calls == 1 and failure == "nonzero":
                return [0.1] * 7
            return [0.0] * 7 if self.refresh_calls <= 2 else list(prehome)

        def apply_static_collision(self, _config, _timeout):
            return {
                "id": configured_collision["object_id"],
                "frame_id": configured_collision["frame_id"],
                "center_xyz_m": list(configured_collision["center_xyz_m"]),
                "orientation_xyzw": list(configured_collision["orientation_xyzw"]),
                "dimensions_xyz_m": list(configured_collision["dimensions_xyz_m"]),
                "top_z_m": configured_collision["top_z_m"],
                "source": configured_collision["source"],
            }

        def guard_snapshot(self, *_args):
            self.guard_calls += 1
            result = json.loads(json.dumps(snapshot))
            if failure == "scene_changed" and self.guard_calls == 2:
                result["primitive_pose"]["position_m"][0] += 0.01
            return result

        def plan(self, payload, stage_goal, *_args, **_kwargs):
            trajectory = RobotTrajectory()
            trajectory.joint_trajectory.joint_names = list(config["right_joint_names"])
            start_point = JointTrajectoryPoint()
            start_point.positions = list(payload["start_positions_rad"])
            goal_point = JointTrajectoryPoint()
            goal_point.positions = list(stage_goal)
            goal_point.time_from_start.sec = 2
            trajectory.joint_trajectory.points = [start_point, goal_point]
            return SimpleNamespace(
                error_code=SimpleNamespace(val=MoveItErrorCodes.SUCCESS),
                trajectory=trajectory,
                trajectory_start=RobotState(),
            )

        def publish_display(self, _response):
            pass

        def fk_pose(self, _payload, _positions, _timeout):
            return SimpleNamespace(
                pose=SimpleNamespace(position=SimpleNamespace(x=0.0, y=0.0, z=0.5))
            )

        def begin_tracking_audit(self):
            pass

        def finish_tracking_audit(self):
            return {"sample_count": 0, "source": "test_fixture"}

        def execute_same_trajectory(self, value, *_args, **_kwargs):
            self.execution_goal_sent = True
            self.executed.append(value)
            return {
                "action_status": GoalStatus.STATUS_SUCCEEDED,
                "moveit_error_code": MoveItErrorCodes.SUCCESS,
            }

        def open_home_gripper(self, _config):
            self.gripper_goal_sent = True
            self.gripper_commands.append(-0.35)
            self.current_positions = list(goal)
            self.current_state_monotonic_s = time.monotonic()
            self.gripper_position = -0.34
            if failure == "gripper_not_open":
                raise RuntimeError("right gripper OPEN measured acceptance failed")
            return {
                "accepted": True,
                "joint_name": home.RIGHT_GRIPPER_JOINT_NAME,
                "target_position_rad": -0.35,
                "measured_position_rad": -0.34,
                "position_error_rad": 0.01,
            }

        def destroy_node(self):
            pass

    report = tmp_path / "home-report.json"
    argv = [
        "home_guarded_executor",
        "--config",
        str(config_path),
        "--report",
        str(report),
    ]
    if armed:
        argv.append("--allow-real-execution")
    monkeypatch.setattr(home, "HomeInitializationNode", FakeNode)
    monkeypatch.setattr(home, "load_home_config", lambda _path: config)
    # These tests isolate downstream planning/execution contracts.  Lowered-table
    # fail-closed behavior is covered separately in test_home_measurements.py.
    monkeypatch.setattr(
        home,
        "require_current_table_collision",
        lambda value, **_kwargs: value["static_collision"],
    )
    monkeypatch.setattr(home, "get_package_share_directory", lambda _name: str(PACKAGE))
    monkeypatch.setattr(home.rclpy, "init", lambda: None)
    monkeypatch.setattr(home.rclpy, "ok", lambda: False)
    monkeypatch.setattr(home.rclpy, "shutdown", lambda: None)
    monkeypatch.setattr(
        home,
        "wait_for_home_settle",
        lambda *_args, **_kwargs: {"accepted": True, "sample_count": 2},
    )
    monkeypatch.setattr(sys, "argv", argv)
    result = home.main()
    return result, instances[0], json.loads(report.read_text())


@pytest.mark.parametrize(
    ("armed", "failure", "reason"),
    [
        (False, None, "--allow-real-execution is required"),
        (True, "nonzero", "HOME_START_NOT_ZERO"),
        (True, "scene_changed", "effective_position_m"),
    ],
)
def test_home_fail_closed_before_goal_send(
    monkeypatch, tmp_path, armed, failure, reason
) -> None:
    result, node, report = _run_home_main(
        monkeypatch, tmp_path, armed=armed, failure=failure
    )
    assert result == 2
    assert node.executed == []
    assert report["state"] == "FAIL_RIGID"
    assert reason in report["failure_reason"]


def test_home_success_executes_a_seven_joint_trajectory(monkeypatch, tmp_path) -> None:
    result, node, report = _run_home_main(monkeypatch, tmp_path)
    assert result == 0
    assert len(node.executed) == 2
    assert all(
        item.joint_trajectory.joint_names == home.RIGHT_ARM_JOINT_NAMES
        for item in node.executed
    )
    assert all(
        "finger" not in name
        for item in node.executed
        for name in item.joint_trajectory.joint_names
    )
    assert report["state"] == "HOME_READY"
    assert report["workflow"] == "FIXED_LIFTED_HOME"
    assert report["home_goal_positions_rad"] == pytest.approx(
        home.FIXED_LIFTED_HOME
    )
    assert node.executed[-1].joint_trajectory.points[-1].positions == pytest.approx(
        home.FIXED_LIFTED_HOME
    )
    assert report["j8_commanded_by_initialization"] is True
    assert node.gripper_commands == [-0.35]
    assert report["home_gripper"]["accepted"] is True
    assert report["stages"][0]["acceptance_tolerance_rad"] == pytest.approx(0.04)
    assert report["stages"][1]["acceptance_tolerance_rad"] == pytest.approx(0.03)
    assert set(node.controller_checks) == {
        "right_joint_trajectory_controller",
        "right_gripper_controller",
    }
    assert "joint_trajectory_controller_tolerances" not in report
    for stage, sent in zip(report["stages"], node.executed):
        assert stage["robot_trajectory"] == home.message_to_ordereddict(sent)
        assert stage["execute_returned_monotonic_ns"] >= stage["execute_called_monotonic_ns"]
    assert report["post_h2_measured_arm_positions_rad"] == pytest.approx(goal := home.FIXED_LIFTED_HOME)
    assert report["post_h2_measured_gripper_position_rad"] == -0.34
    assert report["config_snapshot"]["right_home_joint_positions_rad"] == pytest.approx(goal)


def test_home_refuses_ready_when_gripper_does_not_open(monkeypatch, tmp_path) -> None:
    result, node, report = _run_home_main(
        monkeypatch, tmp_path, failure="gripper_not_open"
    )

    assert result == 2
    assert len(node.executed) == 2
    assert node.gripper_commands == [-0.35]
    assert report["state"] == "FAIL_RIGID"
    assert report["success"] is False
    assert report["j8_commanded_by_initialization"] is True
    assert "gripper OPEN measured acceptance failed" in report["failure_reason"]
