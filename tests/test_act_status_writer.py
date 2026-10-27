import json
import threading
from pathlib import Path

from scripts.act_trial.status_writer import StatusWriter


def test_slow_write_cannot_overwrite_terminal(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    original = Path.write_text
    def delayed(path, data, *args, **kwargs):
        if path.name.endswith('.worker.tmp'):
            entered.set()
            assert release.wait(2)
        return original(path, data, *args, **kwargs)
    monkeypatch.setattr(Path, 'write_text', delayed)
    path = tmp_path/'ready.json'
    writer = StatusWriter(path)
    try:
        writer.submit({'stage': 'active'})
        assert entered.wait(2)
        for i in range(100):
            writer.submit({'stage': 'active', 'i': i})
        assert json.loads(writer.pending)['i'] == 99
        writer.close()
        path.write_text('{"stage":"stopped"}')
    finally:
        release.set()
        writer.close()
        writer.thread.join(2)
    assert not writer.thread.is_alive()
    assert json.loads(path.read_text())['stage'] == 'stopped'


def test_writer_failure_reported(tmp_path, monkeypatch):
    import pytest
    failed = threading.Event()
    def fail(*args, **kwargs):
        failed.set()
        raise OSError('disk error')
    monkeypatch.setattr(Path, 'write_text', fail)
    writer = StatusWriter(tmp_path/'ready.json')
    writer.submit({'stage': 'active'})
    assert failed.wait(2)
    writer.thread.join(2)
    with pytest.raises(RuntimeError, match='disk error'):
        writer.submit({})
    writer.close()
