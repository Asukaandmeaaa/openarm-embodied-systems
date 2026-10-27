from datetime import datetime

from openarm_kinesthetic.recording import allocate_trial_id, valid_trial_id


def test_trial_ids_are_numbered_and_persist_across_calls(tmp_path) -> None:
    moment = datetime(2026, 9, 28, 14, 0, 8).timestamp()
    first = allocate_trial_id(tmp_path, now=moment)
    second = allocate_trial_id(tmp_path, now=moment + 1)
    assert first == "20260928_140008_001"
    assert second == "20260928_140009_002"
    assert valid_trial_id(first)
    assert not valid_trial_id("fixed_20260928_140008")
