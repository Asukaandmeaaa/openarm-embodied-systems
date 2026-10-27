"""Latest-only status file writer; never publishes robot commands."""
import json
from pathlib import Path
import threading


class StatusWriter:
    def __init__(self, path):
        self.path = Path(path)
        self.condition = threading.Condition()
        self.pending = None
        self.closed = False
        self.error = None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def submit(self, value):
        # Serialize before handing ownership to the worker. Queue never grows.
        payload = json.dumps(value, allow_nan=False) + '\n'
        with self.condition:
            if self.error:
                raise RuntimeError('status writer failed: '+str(self.error))
            if self.closed:
                raise RuntimeError('status writer closed')
            self.pending = payload
            self.condition.notify()

    def close(self):
        # Prevent any late worker rename from overwriting a terminal report.
        with self.condition:
            self.closed = True
            self.pending = None
            self.condition.notify()

    def _run(self):
        temporary = self.path.with_suffix('.worker.tmp')
        try:
            while True:
                with self.condition:
                    self.condition.wait_for(lambda: self.closed or self.pending is not None)
                    if self.closed:
                        return
                    payload, self.pending = self.pending, None
                temporary.write_text(payload)
                with self.condition:
                    if self.closed:
                        return
                    temporary.replace(self.path)
        except Exception as exc:
            with self.condition:
                self.error = exc
        finally:
            temporary.unlink(missing_ok=True)
