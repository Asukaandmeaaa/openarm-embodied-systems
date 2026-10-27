import copy
import sys
from pathlib import Path

import pytest


sys.path.insert(
    0,
    str(Path(__file__).parents[1] / "ros2_ws/src/openarm_kinesthetic_recorder"),
)

from openarm_kinesthetic_recorder.act_contract import (  # noqa: E402
    ACTION_ORDER,
    validate_action_chunk,
)


def _payload(*, sequence=1, observation=1, published=10.0, valid_until=10.5):
    actions = [[float(i + joint) for joint in range(16)] for i in range(10)]
    return {
        "schema": "openarm.act_action_chunk.v1",
        "action_sequence": sequence,
        "source_observation_sequence": observation,
        "published_monotonic_s": published,
        "valid_until_monotonic_s": valid_until,
        "inference_api": "predict_action_chunk",
        "action_hz": 30.0,
        "chunk_size": 10,
        "action_order": list(ACTION_ORDER),
        "actions": actions,
        "right_actions": [row[:8] for row in actions],
    }


def test_contract_accepts_one_monotonic_10x8_chunk_and_normalizes_values():
    normalized = validate_action_chunk(_payload(), now=10.1, previous_sequence=0, previous_observation=0)
    assert normalized["action_sequence"] == 1
    assert normalized["right_actions"][0] == [float(i) for i in range(8)]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("inference_api", "select_action", "predict_action_chunk"),
        ("action_sequence", 0, "increase strictly"),
        ("source_observation_sequence", 0, "increase strictly"),
    ],
)
def test_contract_rejects_wrong_api_or_sequence_regressions(field, value, message):
    payload = _payload()
    payload[field] = value
    with pytest.raises(ValueError, match=message):
        validate_action_chunk(payload, now=10.1, previous_sequence=0, previous_observation=0)


def test_contract_rejects_expired_chunk_without_mutating_the_document():
    payload = _payload(valid_until=10.05)
    original = copy.deepcopy(payload)
    with pytest.raises(ValueError, match="expired"):
        validate_action_chunk(payload, now=10.1, previous_sequence=0, previous_observation=0)
    assert payload == original
