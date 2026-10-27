from openarm_kinesthetic.planning.initialization import plan_initialization
from openarm_kinesthetic.planning.layout import scan_layouts


def test_nominal_layout_has_full_ik_and_collision_checked_path() -> None:
    results, summary = scan_layouts(max_candidates=1)
    assert len(results) == 1
    assert summary["success_count"] == 1
    assert summary["nominal"]["failure"] == "PASS"
    assert summary["nominal"]["minimum_clearance_m"] > 0.01
    assert summary["nominal_box_path"]["plan_hash"]
    assert tuple(summary["nominal_box_path"]["stages"]) == ("above_box", "place", "retreat")


def test_initialization_selects_safe_path_and_zero_is_discriminating_failure() -> None:
    report = plan_initialization()
    assert report["status"] == "PASS"
    assert report["selected"]["success"]
    assert not report["motor_zero_negative_control"]["success"]
    assert "arm_pedestal_collision" in report["motor_zero_negative_control"]["collision_failures"]
