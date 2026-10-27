import json

from openarm_kinesthetic.validation.gate_f import inspect_human_dataset


def test_act_gate_refuses_nonhuman_and_requires_four_replay_valid(tmp_path) -> None:
    episodes=tmp_path/"episodes";episodes.mkdir()
    for i in range(4):
        path=episodes/f"e{i}";path.mkdir();(path/"metadata.json").write_text(json.dumps({"source":"human_virtual_handle","strict_replay":{"valid":i<3}}))
    report=inspect_human_dataset(tmp_path);assert report["status"]=="BLOCKED_USER_DATA";assert report["valid_count"]==3
