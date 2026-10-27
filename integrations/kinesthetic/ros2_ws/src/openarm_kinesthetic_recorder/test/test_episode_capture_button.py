import json
from pathlib import Path
import time
from types import SimpleNamespace

from std_msgs.msg import String

from openarm_kinesthetic_recorder.episode_capture_button import EpisodeCaptureButton


def test_second_capture_allocates_distinct_episode_id_and_preserves_source(tmp_path) -> None:
    source_id = "20261008_091207_006"
    collection_root = tmp_path / "collections"
    trajectory = collection_root / "trajectories" / f"{source_id}.json"
    trajectory.parent.mkdir(parents=True)
    node = SimpleNamespace(
        allocate_new_episode_ids=True,
        collection_root=collection_root,
        trajectory=trajectory,
        trajectories=[trajectory],
        metadata={"trial_id": source_id},
    )
    node._episode_trial_id = lambda: EpisodeCaptureButton._episode_trial_id(node)
    EpisodeCaptureButton.select_trajectory(node, 0)
    assert node.source_trial_id == source_id
    assert node.trial_id != source_id
    assert node.metadata["trial_id"] == node.trial_id
    assert node.metadata["source_trajectory_trial_id"] == source_id


def test_b_press_starts_only_after_fresh_idle_gamepad_and_ready_recorder() -> None:
    messages = []
    node = SimpleNamespace(
        button_index=1,
        stage="READY",
        gamepad_status={"connected": True, "stop_latched": False, "control_mode": "idle"},
        gamepad_status_at=time.monotonic(),
        latest_status={"active": False, "sources_ready": True},
        collection_root=Path("/tmp/collections"),
        latest_status_at=time.monotonic(),
        start_client=SimpleNamespace(service_is_ready=lambda: True),
        get_node_names=lambda: ["episode_capture_button", "real_gamepad", "real_episode_recorder"],
        trial_id="20260928_140008_001",
        get_logger=lambda: SimpleNamespace(warn=messages.append, error=messages.append),
    )
    press = String(data=json.dumps({
        "events": [{"kind": "button", "index": 1, "value": 1, "initial": False}]
    }))
    node.latest_status["output_root"] = str(node.collection_root)
    EpisodeCaptureButton.on_gamepad_events(node, press)
    assert node.stage == "METADATA"
    assert "B_CAPTURE_REQUESTED" in messages[-1]

    node.stage = "READY"
    node.gamepad_status["stop_latched"] = True
    EpisodeCaptureButton.on_gamepad_events(node, press)
    assert node.stage == "READY"
    assert "B_CAPTURE_REJECTED" in messages[-1]

    node.gamepad_status["stop_latched"] = False
    node.get_node_names = lambda: ["episode_capture_button", "right_kinesthetic_bridge"]
    EpisodeCaptureButton.on_gamepad_events(node, press)
    assert node.stage == "READY"


def test_b_metadata_ack_must_match_trajectory_trial() -> None:
    published = []
    calls = []
    node = SimpleNamespace(
        stage="METADATA",
        trial_id="20260928_140008_001",
        latest_status={"pending_trial_id": "other"},
        deadline=time.monotonic() + 5.0,
        last_publish=0.0,
        metadata={"trial_id": "20260928_140008_001"},
        metadata_pub=SimpleNamespace(publish=published.append),
        start_client=SimpleNamespace(call_async=lambda _request: calls.append(True)),
    )
    EpisodeCaptureButton.on_timer(node)
    assert len(published) == 1 and not calls
    node.latest_status["pending_trial_id"] = node.trial_id
    EpisodeCaptureButton.on_timer(node)
    assert node.stage == "STARTING" and calls


def test_b_capture_auto_saves_after_successful_replay() -> None:
    logs = []
    requests = []
    future = SimpleNamespace(
        done=lambda: True,
        result=lambda: SimpleNamespace(
            success=True, message="saved pending review: /tmp/episode"
        ),
    )
    node = SimpleNamespace(
        stage="REPLAYING",
        player=SimpleNamespace(poll=lambda: 0),
        auto_finish_client=SimpleNamespace(
            service_is_ready=lambda: True,
            call_async=lambda _request: requests.append(True) or future,
        ),
        get_logger=lambda: SimpleNamespace(warn=logs.append, error=logs.append),
    )
    EpisodeCaptureButton.on_timer(node)
    assert node.stage == "SAVING" and requests == [True]
    EpisodeCaptureButton.on_timer(node)
    assert node.stage == "SAVED"
    assert "B_CAPTURE_SAVED_PENDING_REVIEW" in logs[-1]
