# dora-openarm-classifier

dora-rs node that classifies whether the current state completes the
task successfully or not.

## Install

```bash
pip install dora-openarm-classifier
```

## Usage

This package provides two entry points:

- `dora-openarm-classifier`: the dora-rs node.
- `dora-openarm-classifier-server`: an optional standalone HTTP server
  that loads the model once and serves classification requests, so the
  node itself does not have to hold the model in memory.

For each incoming camera frame the node runs single-frame
[TOPReward](https://topreward.github.io/webpage/), keeps a sliding
window of the last `--window` scores, and reports the window median as
the verdict. When arm positions are provided, `SUCCESS` is only latched
once the grippers are open (release) **and** the vision window agrees,
combining vision and proprioception.

Use it in a dora dataflow:

```yaml
nodes:
  - id: classifier
    build: pip install dora-openarm-classifier
    path: dora-openarm-classifier
    env:
      QUESTION: "Is the object placed in the box?"
    inputs:
      image: camera/image
      position_right: arm-right/state
      position_left: arm-left/state
    outputs:
      - result
```

By default the node loads Qwen3-VL locally. To offload inference to the
server instead, start the server and point the node at it:

```bash
# Terminal 1: start the model server
dora-openarm-classifier-server --port 8000

# Terminal 2 (or via the dataflow): run the node against the server
dora-openarm-classifier --question "..." --server-url http://127.0.0.1:8000
```

## Inputs

| Input            | Description                                                                                       |
| ---------------- | ------------------------------------------------------------------------------------------------- |
| `image`          | JPEG bytes (uint8 array) with `width`/`height`/`encoding` metadata. Triggers a classification.    |
| `position`       | Optional 16-dim arm state `[right(8), left(8)]`; gripper joints at indices 7 and 15.               |
| `position_right` | Optional 8-dim right arm state; gripper joint at index 7.                                          |
| `position_left`  | Optional 8-dim left arm state; gripper joint at index 7.                                           |

Positions may be plain arrays or `StructArray`s with a `qpos` field.
They are stored and applied to the next incoming `image`. Without any
position input, the node falls back to a vision-only verdict.

## Outputs

| Output   | Description                                                                                                   |
| -------- | ----------------------------------------------------------------------------------------------------------- |
| `result` | Float32 window-median score. Metadata carries `verdict` (`SUCCESS`/`FAIL`) and `frame` (frame counter).      |

## Command line options

Each option can also be set through the matching environment variable.

| Option                | Environment variable | Default                      | Description                                                             |
| --------------------- | -------------------- | ---------------------------- | --------------------------------------------------------------------- |
| `--question`          | `QUESTION`           | (required)                   | Yes/no success question asked about each image.                        |
| `--server-url`        | `SERVER_URL`         | (unset; load model locally)  | URL of `dora-openarm-classifier-server`.                              |
| `--window`            | `WINDOW`             | `5`                          | Sliding-window size in frames.                                        |
| `--threshold`         | `THRESHOLD`          | `0.5`                        | `P_yes` threshold for `SUCCESS`.                                       |
| `--gripper-threshold` | `GRIPPER_THRESHOLD`  | `0.2`                        | `\|gripper joint\|` above which a gripper counts as open.               |
| `--classify-hz`       | `CLASSIFY_HZ`        | `0.0`                        | Max inference rate; frames arriving faster are dropped (`0` = no limit). |

The server (`dora-openarm-classifier-server`) accepts `--host`
(default `127.0.0.1`), `--port` (default `8000`), and `--model-id`
(default `Qwen/Qwen3-VL-4B-Instruct`).

## License

Licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

Copyright 2026 Enactic, Inc.

## Code of Conduct

All participation in the OpenArm project is governed by our [Code of Conduct](CODE_OF_CONDUCT.md).
