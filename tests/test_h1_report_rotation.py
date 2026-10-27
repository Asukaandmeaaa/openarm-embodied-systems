"""Report allocation only: no ROS launch or hardware I/O."""
from scripts.act_trial.manual_experiment import startup_report_path


def test_repeated_h1_allocates_new_reports_without_changing_old_evidence(tmp_path):
    previous = tmp_path/'motor_startup.json'
    previous.write_text('old report')
    first = startup_report_path(tmp_path)
    first.write_text('first new report')
    second = startup_report_path(tmp_path)
    assert first != second
    assert second.parent.is_dir() and not second.exists()
    assert previous.read_text() == 'old report'
    assert first.read_text() == 'first new report'
    assert second.parent.parent == tmp_path/'h1_startups'
