"""Offline supervision provenance tests; never connect to a robot."""
import json

import numpy as np
import pytest

from scripts.convert_real_episodes_act import prepare
from vision_grasp_baseline.training_contract import audit_action_contract, INTERFACES
from vision_grasp_baseline.tests.test_collection import recorded, closed_state


def episode(tmp_path, monkeypatch):
    path, _ = recorded(tmp_path, monkeypatch, raw=True)
    meta = json.loads((path/'metadata.json').read_text())
    meta['controller_command_interfaces'] = INTERFACES
    (path/'metadata.json').write_text(json.dumps(meta))
    return path


def mutate(path, function):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    function(rows)
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))


def test_native_conversion_preserves_feedback_input_and_distinct_command_label(tmp_path, monkeypatch):
    path = episode(tmp_path, monkeypatch)
    proof = audit_action_contract(path)
    assert proof['passed'] and proof['checked_actions'] == 151
    assert proof['tracking_diagnostic']['absolute_error_p95_rad'][0] == pytest.approx(.02)
    _, q, ids, _ = prepare(path, 30)
    state = q['joint_observation'][ids['joint_observation']]
    action = q['demonstration_action'][ids['demonstration_action']]
    assert state.shape == action.shape == (91, 16)
    assert np.allclose(state, closed_state())
    assert np.all(action[:, :7] == np.float32(.02))
    assert not np.array_equal(state, action)


def test_feedback_must_not_replace_command_actions(tmp_path, monkeypatch):
    path = episode(tmp_path, monkeypatch)
    mutate(path/'demonstration_action.jsonl', lambda rows: rows[20].update(positions=closed_state()))
    with pytest.raises(ValueError, match='Action labels differ'):
        audit_action_contract(path)


def test_target_must_match_output_not_reference_or_feedback(tmp_path, monkeypatch):
    path = episode(tmp_path, monkeypatch)
    mutate(path/'target_right_arm.jsonl', lambda rows: rows[20].update(positions=[.4]*7))
    with pytest.raises(ValueError, match='not raw JTC output'):
        audit_action_contract(path)


def test_cannot_replace_state_with_controller_reference(tmp_path, monkeypatch):
    path = episode(tmp_path, monkeypatch)
    mutate(path/'joint_observation.jsonl', lambda rows: rows[20].update(positions=[.4]*16))
    with pytest.raises(ValueError, match='not original JointState'):
        audit_action_contract(path)


def test_future_component_target_is_rejected(tmp_path, monkeypatch):
    path = episode(tmp_path, monkeypatch)
    def future(rows):
        rows[20]['component_timestamp_s']['left_arm'] += .1
    mutate(path/'demonstration_action.jsonl', future)
    with pytest.raises(ValueError, match='not causal/fresh'):
        audit_action_contract(path)


def test_missing_or_wrong_controller_profile_fails_closed(tmp_path, monkeypatch):
    path = episode(tmp_path, monkeypatch)
    meta = json.loads((path/'metadata.json').read_text())
    meta['controller_command_interfaces']['right_joint_trajectory_controller'] = ['position']
    (path/'metadata.json').write_text(json.dumps(meta))
    with pytest.raises(ValueError, match='controller interfaces'):
        audit_action_contract(path)


def test_velocity_evidence_must_be_retained_for_position_velocity_deployment(tmp_path, monkeypatch):
    path = episode(tmp_path, monkeypatch)
    def remove(rows):
        next(r for r in rows if r['controller'] == 'right_arm')['message']['output']['velocities'] = []
    mutate(path/'controller_raw.jsonl', remove)
    with pytest.raises(ValueError, match='dimensions'):
        audit_action_contract(path)


def test_new_export_runs_provenance_gate_before_writing_dataset(tmp_path, monkeypatch):
    from vision_grasp_baseline import export_act
    path = episode(tmp_path, monkeypatch)
    meta = json.loads((path/'metadata.json').read_text())
    meta.update(review_status='approved', operator_marked_success=True)
    (path/'metadata.json').write_text(json.dumps(meta))
    mutate(path/'demonstration_action.jsonl', lambda rows: rows[20].update(positions=closed_state()))
    output = tmp_path/'exported'
    monkeypatch.setattr('sys.argv', ['export_act', '--source', str(tmp_path), '--output', str(output)])
    with pytest.raises(ValueError, match='Action labels differ'):
        export_act.main()
    assert not output.exists()


def test_export_audit_report_records_controller_contract_without_hardware(tmp_path, monkeypatch):
    from vision_grasp_baseline import export_act
    path = episode(tmp_path, monkeypatch)
    meta = json.loads((path/'metadata.json').read_text())
    meta.update(review_status='approved', operator_marked_success=True,ros_time_mapping=dict(offset_ns=0))
    (path/'metadata.json').write_text(json.dumps(meta))
    from vision_grasp_baseline.gripper_settings import COLLECTION_EVENTS
    events=[dict(phase='continuous_dispatch',detail=dict(start_ros_ns=100_000_000_000,duration_s=3.))]
    events += [dict(phase='timeline_event',detail=dict(name=name,time_s=i*.3))
               for i,name in enumerate(COLLECTION_EVENTS)]
    (path/'events.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in events))
    output = tmp_path/'exported'
    monkeypatch.setattr('sys.argv', ['export_act', '--source', str(tmp_path), '--output', str(output), '--audit-only'])
    export_act.main()
    report = json.loads((tmp_path/'exported_conversion_report.json').read_text())
    assert report['action_contract']['controller_command_interfaces'] == INTERFACES
    assert report['episodes'][0]['action_contract_audit']['passed']
    assert report['episodes'][0]['phase_coverage']['all_task_phases_present']
    assert not output.exists()
