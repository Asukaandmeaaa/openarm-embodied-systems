# Copyright 2026 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""dora-rs node providing a web UI to orchestrate inference evaluation runs."""

import argparse
import asyncio
import collections
from contextlib import asynccontextmanager
import dataclasses
import datetime
import dora
from collections.abc import AsyncIterable
from fastapi import FastAPI, Form, Request
from fastapi.responses import JSONResponse
from fastapi.sse import EventSourceResponse, ServerSentEvent
from fastapi.templating import Jinja2Templates
import os
import pathlib
import pyarrow as pa
import time
import uvicorn
import yaml

from .eval_metadata import EvalMetadata

base_dir = os.path.dirname(__file__)
templates = Jinja2Templates(directory=f"{base_dir}/templates")

node = None
tasks: list[dict] = []
eval_metadata: EvalMetadata | None = None
eval_metadata_path: pathlib.Path | None = None

auto_open = False
port = None


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Open a web browser automatically if requested."""
    if auto_open:
        url = f"http://127.0.0.1:{port}"
        await asyncio.create_subprocess_exec("open", url)
    yield


app = FastAPI(lifespan=_lifespan)


# ------------------------------------------------------------------
# State
# ------------------------------------------------------------------


@dataclasses.dataclass
class State:
    """Full UI state — serialised to JSON and pushed via SSE on every change."""

    collecting: bool = False
    running: bool = True
    episode_number: int = 0
    task_index: int = 0
    task_title: str = ""

    # Latest status reported by each arm. Field name matches the dora input id.
    arm_status_right: str = "stopped"
    arm_status_left: str = "stopped"
    require_home: bool = os.getenv("REQUIRE_HOME", "no") == "yes"
    home_status_right: str = "unknown"
    home_status_at: float = 0.0
    home_detail: str = ""

    # Blocks START after each episode until operator confirms scene is ready.
    reset_needed: bool = False

    # True between "initialize" (button press) and both arms emitting "started".
    initializing: bool = False

    # Session-level flags.
    resuming: bool = False
    resume_count: int = 0  # episodes already in metadata at resume time


state = State()
_state_changed = asyncio.Condition()
_arms_started: set[str] = set()  # tracks which arms have reported "started" during init


# ------------------------------------------------------------------
# Camera stats (mirrors data-collection-ui implementation)
# ------------------------------------------------------------------

CAMERA_INPUTS = (
    "camera_wrist_right",
    "camera_wrist_left",
    "camera_head_left",
    "camera_head_right",
    "camera_ceiling",
)

CAMERA_TIMESTAMP_WINDOW = 60
CAMERA_STALE_AFTER_S = 1.0

# dora-openarm status inputs, one per arm. The input id matches the State field.
ARM_STATUS_INPUTS = ("arm_status_right", "arm_status_left")


@dataclasses.dataclass
class CameraStats:
    """Rolling FPS/jitter stats for one camera stream."""

    fps: float = 0.0
    jitter_ms: float = 0.0


camera_stats: dict[str, CameraStats] = {n: CameraStats() for n in CAMERA_INPUTS}
camera_timestamps: dict[str, collections.deque] = {
    n: collections.deque(maxlen=CAMERA_TIMESTAMP_WINDOW) for n in CAMERA_INPUTS
}


def _event_ts_to_seconds(ts) -> float:
    if isinstance(ts, datetime.datetime):
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=datetime.timezone.utc)
        return ts.timestamp()
    if isinstance(ts, (int, float)):
        return float(ts) / 1e9
    return time.time()


def _update_camera_stats(event_id: str, ts_s: float) -> None:
    series = camera_timestamps[event_id]
    if series and ts_s - series[-1] > CAMERA_STALE_AFTER_S:
        series.clear()
    series.append(ts_s)
    if len(series) < 2:
        return
    span = series[-1] - series[0]
    if span <= 0:
        return
    fps = (len(series) - 1) / span
    diffs = [series[i] - series[i - 1] for i in range(1, len(series))]
    jitter_ms = (max(diffs) - min(diffs)) * 1e3
    camera_stats[event_id].fps = fps
    camera_stats[event_id].jitter_ms = jitter_ms


# ------------------------------------------------------------------
# Per-task / session statistics derived from eval_metadata
# ------------------------------------------------------------------


def _task_stats(task_index: int) -> dict:
    eps = eval_metadata.episodes if eval_metadata else []
    task_eps = [e for e in eps if e.get("task_index") == task_index]
    n = len(task_eps)
    s = sum(1 for e in task_eps if e.get("success"))
    return {"episodes": n, "success": s, "pct": round(100 * s / n) if n else 0}


def _session_stats() -> dict:
    eps = eval_metadata.episodes if eval_metadata else []
    n = len(eps)
    s = sum(1 for e in eps if e.get("success"))
    return {"total": n, "success": s, "pct": round(100 * s / n) if n else 0}


def _episode_history(limit: int = 10) -> list[dict]:
    if not eval_metadata:
        return []
    result = []
    for ep in eval_metadata.episodes[-limit:]:
        ti = ep.get("task_index", 0)
        prompt = tasks[ti]["prompt"] if ti < len(tasks) else "?"
        result.append(
            {
                "id": ep["id"],
                "task_index": ti,
                "task_title": prompt,
                "success": ep.get("success", False),
                "note": ep.get("note", ""),
            }
        )
    return result


# ------------------------------------------------------------------
# State serialisation for SSE
# ------------------------------------------------------------------


def _state_payload() -> dict:
    return {
        "collecting": state.collecting,
        "initializing": state.initializing,
        "reset_needed": state.reset_needed,
        "episode_number": state.episode_number,
        "task_index": state.task_index,
        "task_count": len(tasks),
        "task_title": state.task_title,
        "arm_status_right": state.arm_status_right,
        "arm_status_left": state.arm_status_left,
        "home_status_right": state.home_status_right,
        "home_detail": state.home_detail,
        "require_home": state.require_home,
        "home_can_start": _home_can_start(),
        "task_stats": _task_stats(state.task_index),
        "session_stats": _session_stats(),
        "resuming": state.resuming,
        "resume_count": state.resume_count,
        "episode_history": _episode_history(),
    }


# ------------------------------------------------------------------
# Command helpers
# ------------------------------------------------------------------


async def _notify_state_changed() -> None:
    async with _state_changed:
        _state_changed.notify_all()


def _send_task_prompt() -> None:
    """Push current task prompt to dora for policy language conditioning."""
    if node and tasks:
        node.send_output(
            "task_prompt",
            pa.array([state.task_title]),
            {"task_index": state.task_index},
        )


def _home_can_start() -> bool:
    return not state.require_home or (
        state.home_status_right == "home_ready"
        and time.monotonic() - state.home_status_at < 2.0
    )


def _command_start() -> None:
    """Start arms and begin episode. Arms emit 'started' when hardware is ready."""
    if not _home_can_start():
        return
    metadata = {"episode_number": state.episode_number, "task_index": state.task_index}
    node.send_output("arm_command", pa.array(["start"]), metadata)
    _send_task_prompt()
    state.initializing = True
    state.reset_needed = False


def _finish_episode(*, success: bool, note: str) -> None:
    """Shared logic for success/fail: record metadata, update state."""
    if eval_metadata is not None:
        eval_metadata.add_episode(
            id=str(state.episode_number),
            task_index=state.task_index,
            success=success,
            note=note,
        )
        eval_metadata.save(eval_metadata_path)

    state.collecting = False
    state.episode_number += 1
    state.reset_needed = True


def _command_success(note: str = "") -> None:
    node.send_output("arm_command", pa.array(["stop"]))
    _finish_episode(success=True, note=note)


def _command_fail(note: str = "") -> None:
    node.send_output("arm_command", pa.array(["stop"]))
    _finish_episode(success=False, note=note)


def _command_cancel() -> None:
    """Cancel without recording an episode result."""
    node.send_output("arm_command", pa.array(["stop"]))
    state.collecting = False
    state.reset_needed = True


def _command_quit() -> None:
    node.send_output("arm_command", pa.array(["quit"]))
    state.running = False


# ------------------------------------------------------------------
# HTTP endpoints
# ------------------------------------------------------------------


@app.get("/")
async def _root(request: Request):
    """Render the main page with initial server-side state."""
    return templates.TemplateResponse(
        request=request,
        name="root.html",
        context={
            "state": state,
            "tasks": tasks,
            "task_stats": _task_stats(state.task_index),
            "session_stats": _session_stats(),
            "eval_metadata": eval_metadata,
        },
    )


@app.post("/start")
async def _start():
    if not _home_can_start():
        return JSONResponse({"ok": False, "error": "Fresh HOME_READY required"}, status_code=409)
    if not state.collecting and not state.reset_needed and not state.initializing:
        _command_start()
        await _notify_state_changed()
    return JSONResponse({"ok": True})


@app.post("/success")
async def _success(note: str = Form(default="")):
    if state.collecting:
        _command_success(note=note)
        await _notify_state_changed()
    return JSONResponse({"ok": True})


@app.post("/fail")
async def _fail(note: str = Form(default="")):
    if state.collecting:
        _command_fail(note=note)
        await _notify_state_changed()
    return JSONResponse({"ok": True})


@app.post("/cancel")
async def _cancel():
    if state.collecting:
        _command_cancel()
        await _notify_state_changed()
    elif state.initializing:
        node.send_output("arm_command", pa.array(["stop"]))
        _arms_started.clear()
        state.initializing = False
        state.reset_needed = True
        await _notify_state_changed()
    return JSONResponse({"ok": True})


@app.post("/ready")
async def _ready():
    """Operator confirms robot has been reset; unblock START."""
    if state.reset_needed:
        state.reset_needed = False
        await _notify_state_changed()
    return JSONResponse({"ok": True})


@app.post("/quit")
async def _quit():
    _command_quit()
    await _notify_state_changed()
    return JSONResponse({"ok": True})


@app.post("/set-task")
async def _set_task(task_index: int = Form(...)):
    if 0 <= task_index < len(tasks):
        state.task_index = task_index
        state.task_title = tasks[task_index]["prompt"]
        _send_task_prompt()
        await _notify_state_changed()
    return JSONResponse({"ok": True})


@app.post("/set-description")
async def _set_description(description: str = Form(default="")):
    if eval_metadata is not None:
        eval_metadata.checkpoint_description = description
        eval_metadata.save(eval_metadata_path)
    return JSONResponse({"ok": True})


@app.post("/emergency-stop")
async def _emergency_stop():
    """Disable motors; physical E-stop remains independent of this UI."""
    if node:
        node.send_output("arm_command", pa.array(["emergency_stop" if state.require_home else "stop"]))
    _arms_started.clear()
    state.collecting = False
    state.initializing = False
    state.reset_needed = True
    state.home_status_right = "disabled"
    await _notify_state_changed()
    return JSONResponse({"ok": True})


@app.get("/state")
async def _get_state():
    """Return current state as JSON (for initial JS hydration)."""
    return JSONResponse(_state_payload())


# ------------------------------------------------------------------
# SSE streams
# ------------------------------------------------------------------


@app.get("/events", response_class=EventSourceResponse)
async def _events() -> AsyncIterable[ServerSentEvent]:
    """Push full state to browser on every change."""
    while state.running:
        async with _state_changed:
            await _state_changed.wait()
        yield ServerSentEvent(data=_state_payload())


@app.get("/stats", response_class=EventSourceResponse)
async def _stats() -> AsyncIterable[ServerSentEvent]:
    """Push camera FPS/jitter every 500 ms."""
    while state.running:
        now = time.time()
        snapshot = {}
        for name, s in camera_stats.items():
            series = camera_timestamps[name]
            if not series or now - series[-1] > CAMERA_STALE_AFTER_S:
                snapshot[name] = {"fps": 0.0, "jitter_ms": 0.0}
            else:
                snapshot[name] = {"fps": s.fps, "jitter_ms": s.jitter_ms}
        yield ServerSentEvent(data=snapshot)
        await asyncio.sleep(0.5)


# ------------------------------------------------------------------
# dora event loop
# ------------------------------------------------------------------


async def _main_dora(server: uvicorn.Server) -> None:
    """Process dora events and drive state machine."""
    while state.running:
        if node.is_empty():
            await asyncio.sleep(0.001)
            continue

        event = node.next()
        if event["type"] == "STOP":
            state.running = False
            break

        if event["type"] != "INPUT":
            continue

        event_id = event["id"]

        if event_id in CAMERA_INPUTS:
            _update_camera_stats(
                event_id,
                _event_ts_to_seconds(event["metadata"].get("timestamp")),
            )
        elif event_id == "home_status_right":
            state.home_status_right = event["value"][0].as_py()
            state.home_status_at = time.monotonic()
            state.home_detail = event.get("metadata", {}).get("detail", "")
            if state.require_home and state.home_status_right in {"home_error", "disabled"}:
                if state.collecting or state.initializing:
                    node.send_output("arm_command", pa.array(["stop"]))
                state.collecting = False
                state.initializing = False
                state.reset_needed = True
                _arms_started.clear()
            await _notify_state_changed()
        elif event_id == "tick" and state.require_home:
            await _notify_state_changed()
        elif event_id in ARM_STATUS_INPUTS:
            value = event["value"][0].as_py()
            setattr(state, event_id, value)
            if value == "started" and state.initializing:
                side = "right" if event_id == "arm_status_right" else "left"
                _arms_started.add(side)
                if _arms_started >= {"right", "left"}:
                    _arms_started.clear()
                    _send_task_prompt()
                    state.collecting = True
                    state.initializing = False
                    await _notify_state_changed()

    server.should_exit = True


# ------------------------------------------------------------------
# Entrypoint
# ------------------------------------------------------------------


async def _main_async() -> None:
    config = uvicorn.Config(app, port=port, log_level="warning")
    server = uvicorn.Server(config)
    await asyncio.gather(server.serve(), _main_dora(server))


def main() -> None:
    """Run evaluation UI web application."""
    global node, tasks, eval_metadata, eval_metadata_path
    global auto_open, port

    # DATASET_DIR is the directory that contains (or will contain) both
    # metadata.yaml (written by the recorder) and eval_metadata.yaml.
    # It must match the recorder's output directory.
    _dataset_dir = os.getenv("DATASET_DIR", ".")

    parser = argparse.ArgumentParser(description="OpenArm inference evaluation UI")
    parser.add_argument(
        "--metadata-file",
        default=os.getenv("METADATA_FILE"),
        help="Source metadata.yaml with task list. Required for a new dataset; "
        "not needed when resuming (tasks are cached in eval_metadata.yaml).",
        type=pathlib.Path,
    )
    parser.add_argument(
        "--checkpoint-path",
        default=os.getenv("CHECKPOINT_PATH", ""),
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--auto-open",
        action=argparse.BooleanOptionalAction,
        default=os.getenv("AUTO_OPEN", "") == "yes",
        help="Open a browser automatically",
    )
    parser.add_argument(
        "--port",
        default=int(os.getenv("PORT", "8000")),
        type=int,
    )
    args = parser.parse_args()

    auto_open = args.auto_open
    port = args.port

    # Load or create eval metadata.
    eval_metadata_path = (pathlib.Path(_dataset_dir) / "eval_metadata.yaml").resolve()
    if eval_metadata_path.exists():
        eval_metadata = EvalMetadata.load(eval_metadata_path)
    else:
        eval_metadata = EvalMetadata()

    # Determine the starting episode number.  The recorder's metadata.yaml is
    # the authoritative source for what physically exists on disk; eval_metadata
    # may lag behind if the UI was restarted before any episode was recorded.
    _dataset_meta_path = pathlib.Path(_dataset_dir) / "metadata.yaml"
    _recorder_episodes: list[dict] = []
    if _dataset_meta_path.exists():
        with open(_dataset_meta_path) as _f:
            _recorder_meta = yaml.safe_load(_f) or {}
        _recorder_episodes = _recorder_meta.get("episodes", [])

    _all_known_ids = {int(e["id"]) for e in eval_metadata.episodes + _recorder_episodes}
    if _all_known_ids:
        state.resuming = True
        state.resume_count = len(_recorder_episodes) or len(eval_metadata.episodes)
        state.episode_number = max(_all_known_ids) + 1

    # Resolve task list.
    #
    # Priority (first non-empty wins):
    #   1. Tasks cached in eval_metadata.yaml       — always used on resume
    #   2. DATASET_DIR/metadata.yaml                — dataset already populated
    #   3. METADATA_FILE                            — new run, user-supplied source
    if eval_metadata.tasks:
        tasks = eval_metadata.tasks
    else:
        _task_source: pathlib.Path | None = None
        if _dataset_meta_path.exists():
            _task_source = _dataset_meta_path
        elif args.metadata_file and args.metadata_file.exists():
            _task_source = args.metadata_file

        if _task_source is None:
            _checked = [str(_dataset_meta_path)]
            if args.metadata_file:
                _checked.append(str(args.metadata_file))
            raise SystemExit(
                "No task list found. For a new dataset set METADATA_FILE.\n"
                f"Checked: {', '.join(_checked)}"
            )

        with open(_task_source) as f:
            tasks = (yaml.safe_load(f) or {}).get("tasks", [])
        if not tasks:
            raise SystemExit(f"No tasks: found in {_task_source}")
        eval_metadata.tasks = tasks

    # Populate checkpoint info.
    eval_metadata.auto_populate(args.checkpoint_path or None)
    eval_metadata.save(eval_metadata_path)

    state.task_title = tasks[state.task_index]["prompt"]

    node = dora.Node()
    asyncio.run(_main_async())


if __name__ == "__main__":
    main()
