from openarm_kinesthetic.validation.gate_d import validate_gate_d


def test_automatic_expert_recorder_sync_and_quality_gate() -> None:
    report = validate_gate_d(save_artifact=False)
    assert report["status"] == "PASS", report
    assert report["automatic_expert"]["grasp_count"] == 1
    assert report["checks"]["left_action_fixed_park"]
    assert report["checks"]["box_path_collision_checked"]
    assert report["automatic_expert"]["box_path_plan"]["plan_hash"]
