from openarm_kinesthetic.validation.gate_e import validate_gate_e


def test_strict_position_actuator_replay_passes_shared_contract() -> None:
    report=validate_gate_e()
    assert report["status"]=="PASS",report
    assert not report["replay"]["object_weld_used"]
    assert report["checks"]["box_path_plan_traceable"]
    assert report["replay"]["pipeline"]["upsampler"] == "HermiteUpsampler"
    assert report["replay"]["pipeline"]["lowpass"] == "BiquadLowpass"
    assert report["replay"]["pipeline"]["command_target"] == "data.ctrl"
