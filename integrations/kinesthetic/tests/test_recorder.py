from dataclasses import dataclass
import json

import numpy as np
import pytest

from openarm_kinesthetic.recording import EpisodeRecorder, RecorderStateError
from openarm_kinesthetic.recording.synchronization import synchronize_30hz


@dataclass
class Result:
    full_sequence_success: bool


def test_success_failure_cancel_and_residual_isolation(tmp_path) -> None:
    recorder = EpisodeRecorder(tmp_path)
    recorder.begin({"task": "x"}, "ok"); recorder.append("physics", 0.0, {"q": [0]})
    with pytest.raises(RecorderStateError): recorder.finish_success(Result(False))
    recorder.finish_success(Result(True))
    recorder.begin({"task": "x"}, "bad"); recorder.finish_failure("contract")
    recorder.begin({"task": "x"}, "cancel"); recorder.cancel()
    (recorder.transactions / "crash").mkdir(); (recorder.transactions / "crash" / "metadata.json").write_text('{"transaction_state":"ACTIVE"}')
    catalog = recorder.write_catalog()
    assert catalog["episodes"] == ["ok"] and catalog["failed"] == ["bad"]
    assert catalog["episodes_pending_review"] == []
    assert catalog["residual_transactions"] == [{"episode_id": "crash", "state": "ACTIVE"}]


def test_operator_success_is_saved_pending_review(tmp_path):
    recorder = EpisodeRecorder(tmp_path)
    recorder.begin({"source": "real_right_kinesthetic"}, "real_1")
    recorder.update_metadata({"trajectory_sha256": "a" * 64})

    destination = recorder.finish_operator_success("attended")

    metadata = json.loads((destination / "metadata.json").read_text())
    assert destination.parent.name == "episodes_pending_review"
    assert metadata["transaction_state"] == "COMMITTED_PENDING_REVIEW"
    assert metadata["operator_marked_success"] is True
    assert metadata["automatic_task_contract_passed"] is False
    assert metadata["trajectory_sha256"] == "a" * 64
    assert not any(path.name == "cancel" for path in tmp_path.rglob("*"))


def test_automatic_replay_commit_does_not_claim_operator_task_success(tmp_path):
    recorder = EpisodeRecorder(tmp_path)
    recorder.begin({"source": "real_right_continuous_marked_replay"}, "20260928_140008_001")
    destination = recorder.finish_automatic_pending_review("raw audit passed")
    metadata = json.loads((destination / "metadata.json").read_text())
    assert destination.parent.name == "episodes_pending_review"
    assert metadata["transaction_state"] == "COMMITTED_PENDING_REVIEW"
    assert metadata["operator_marked_success"] is False
    assert metadata["review_status"] == "pending"
    assert metadata["terminal_label"] is None


def test_timestamp_synchronization_reports_repeat_and_missing() -> None:
    times = np.arange(0, 0.21, 0.01); values = np.tile(np.arange(16), (len(times), 1))
    _, states, actions, cameras, report = synchronize_30hz(times, values, times, values, {"ceiling": [0, .1, .2], "wrist_right": [0, .033, .066, .1, .133, .166, .2]})
    assert states.shape == actions.shape == (7, 16)
    assert report.repeated_camera_frames["ceiling"] > 0
    assert set(cameras) == {"ceiling", "wrist_right"}


def test_timestamp_synchronization_uses_true_nearest_samples_and_rejects_bad_clocks() -> None:
    times = np.array([0.0, 0.02, 0.2])
    values = np.tile(np.arange(16), (len(times), 1))
    _, _, _, _, report = synchronize_30hz(
        times,
        values,
        times,
        values,
        {"ceiling": [0.0, 0.2], "wrist_right": [0.0, 0.2]},
    )
    assert report.maximum_state_skew_s == pytest.approx(0.08)
    with pytest.raises(ValueError, match="strictly monotonic"):
        synchronize_30hz(
            times,
            values,
            times,
            values,
            {"ceiling": [0.1, 0.0], "wrist_right": [0.0, 0.2]},
        )


def test_recorder_rejects_nonmonotonic_source_clock(tmp_path) -> None:
    recorder=EpisodeRecorder(tmp_path);recorder.begin({},"clock")
    recorder.append("physics",1.0,{})
    with pytest.raises(RecorderStateError,match="strictly monotonic"):recorder.append("physics",1.0,{})


def test_recorder_preserves_gripper_event_as_its_own_timestamped_stream(tmp_path) -> None:
    recorder = EpisodeRecorder(tmp_path)
    recorder.begin({}, "gripper")
    recorder.append("gripper_event", 12.5, {"event": "close"})
    path = recorder.finish_operator_success("review")
    row = json.loads((path / "gripper_event.jsonl").read_text())
    assert row == {"timestamp_s": 12.5, "event": "close"}
