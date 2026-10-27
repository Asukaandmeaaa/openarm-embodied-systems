"""ACT diagnostics must not inherit a fault limit from a single H2 run."""
import hashlib
import json

import pytest

from scripts.act_trial.tracking_contract import tracking_contract, stationary_recovery_contract


def bundle(tmp_path, peak=.03863330737282597):
    rows=[{'joint_name':f'openarm_right_joint{i+1}',
           'maximum_absolute_tracking_error_rad':peak if i==5 else .02} for i in range(7)]
    report={'success':True,'execution_tracking_audit':{
        'stage_a':{'per_joint':rows},'stage_b':{'per_joint':rows}}}
    raw=json.dumps(report).encode();(tmp_path/'h2_execution.json').write_bytes(raw)
    manifest=json.dumps({'complete':True,'files':{'h2_execution.json':hashlib.sha256(raw).hexdigest()}}).encode()
    (tmp_path/'manifest.json').write_bytes(manifest)
    return {'capture_directory':str(tmp_path),'capture_manifest_sha256':hashlib.sha256(manifest).hexdigest(),
            'h2_execution':report}


def test_stationary_recovery_is_separate_from_task_monitoring(tmp_path):
    b=bundle(tmp_path)
    result=stationary_recovery_contract(b)
    assert result['limits_rad']==[.03]*5+[.04]+[.03]*2
    assert result['home_tolerance_unchanged'] and not result['automatic_threshold_adaptation']
    task = tracking_contract(b, 'task')
    assert task['schema'] == 'act_tracking_policy_v3'
    assert task['active_stop_limits_rad'] == [None]*8
    assert task['holding_stop_limits_rad'] == [.03]*8
    assert task['warning_thresholds_rad'] == [.03]*8
    assert not task['H2_peaks_used_for_ACT_stop']
    assert tracking_contract(b)['active_stop_limits_rad'] == [.03]*8


@pytest.mark.parametrize('peak',[.02,.041,float('nan')])
def test_unreviewed_history_never_widens_limit(tmp_path,peak):
    with pytest.raises(ValueError):stationary_recovery_contract(bundle(tmp_path,peak))


@pytest.mark.parametrize('file',['manifest.json','h2_execution.json'])
def test_changed_evidence_is_rejected(tmp_path,file):
    b=bundle(tmp_path);(tmp_path/file).write_text('{}')
    with pytest.raises(ValueError,match='hash mismatch'):stationary_recovery_contract(b)


def test_dynamic_policy_does_not_derive_limits_from_h2(tmp_path):
    expected = tracking_contract({}, 'task')
    for peak in (.001, .03863, .5):
        assert tracking_contract(bundle(tmp_path, peak), 'task') == expected
    # JSON null describes absent task tracking stops, never nonfinite infinity.
    json.dumps(expected, allow_nan=False)
    with pytest.raises(ValueError): tracking_contract({}, 'unknown')
