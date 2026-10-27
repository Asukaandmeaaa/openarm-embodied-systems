# dora-openarm-evaluation-ui

A [dora-rs](https://dora-rs.ai) node providing a web UI to run policy
(inference) evaluation sessions on the
[OpenArm](https://github.com/enactic/openarm) bimanual robot.

An operator drives each evaluation episode from the browser; the node
orchestrates the rest of the dataflow by emitting:

- **`arm_command`** — `start` / `stop` / `quit` for the arms and other quittable nodes
- **`task_prompt`** — the current task's language instruction, used to condition the policy

Per-episode results (success/fail and notes) are logged to `eval_metadata.yaml`.

## Features

- Single-operator episode loop: **start → success / fail / cancel → reset → repeat**
- Task switcher with per-task and session success-rate stats
- Live camera FPS / jitter monitor for all five streams
- Resume: a restart continues from the next episode number found in existing metadata
- Checkpoint panel: path, description, capture timestamp
- Keyboard shortcuts and an emergency stop

## Install

```bash
pip install -e .        # or: uv sync
```

dora can also build the node from the dataflow (`build: pip install -e .`).

## Run

The node runs inside a dora
dataflow. [`example/dataflow.yaml`](example/dataflow.yaml) is a
reference wiring of the UI with the arms, cameras, observer, policy
server, and action executor:

```bash
dora run dataflow.yaml --uv
```

The UI is served at <http://127.0.0.1:8000> (opens automatically when `AUTO_OPEN=yes`).

### Configuration

Set through the dataflow `env:` block or via CLI flags:

| Env | Flag | Default | Description |
|-----|------|---------|-------------|
| `DATASET_DIR` | — | `.` | Directory holding `metadata.yaml` and `eval_metadata.yaml`. |
| `METADATA_FILE` | `--metadata-file` | — | Source `metadata.yaml` with the task list. Required for a new dataset; not needed on resume (tasks are cached in `eval_metadata.yaml`). |
| `CHECKPOINT_PATH` | `--checkpoint-path` | — | Model checkpoint path, recorded once into `eval_metadata.yaml`. |
| `AUTO_OPEN` | `--auto-open` | `no` | Open a browser automatically (`yes` / `no`). |
| `PORT` | `--port` | `8000` | Web server port. |

### Task list

Tasks are read from a `metadata.yaml` (see the included example):

```yaml
tasks:
  - prompt: Snug the pillow into the case.
    description: SMALL PILLOW
```

Resolution order: tasks cached in `eval_metadata.yaml` (resume) →
`DATASET_DIR/metadata.yaml` → `--metadata-file`.

## Operating the UI

Each episode follows **Idle → Start** (the arms initialize and the UI waits for both to
report `started`) **→ Collecting → Success / Fail / Cancel → Reset Robot** (the operator
resets the scene, then presses **Ready**) **→ Idle**. *Emergency Stop* cancels the current
episode and disables the arm motors immediately.

| Key | Action |
|-----|--------|
| `Space` / `s` | Start (idle) or mark Success (collecting) |
| `f` | Fail |
| `x` | Cancel (no result recorded) |
| `r` | Ready — reset complete |
| `←` / `→` | Stage previous / next task |
| `Enter` | Commit the staged task change |
| `q` | Quit the session |

## Dataflow interface

**Inputs:** `tick`, `arm_status_right`, `arm_status_left`, and five camera streams
(`camera_wrist_right`, `camera_wrist_left`, `camera_head_left`, `camera_head_right`,
`camera_ceiling`).

**Outputs:** `arm_command`, `task_prompt`.

`eval_metadata.yaml` records the checkpoint info, the task list, and one entry per
episode (`id`, `task_index`, `success`, `note`, `timestamp`).

## License

Licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

Copyright 2026 Enactic, Inc.

## Code of Conduct

All participation in the OpenArm project is governed by our [Code of Conduct](CODE_OF_CONDUCT.md).
