# dora-openarm

A [Dora](https://dora-rs.ai/) node that controls OpenArm.

## Local DM-USB2FDCAN feedback repair

This workspace requires `openarm-can==1.4.0+dmrefresh1` and
`openarm-driver==0.5.1`. The vendored CAN source spaces the eight status-query
frames by 1 ms. The Dora driver also limits explicit refresh batches to 20 Hz;
every read still processes received data, and motion command dispatch is unchanged.
Startup rejects an unpatched CAN package before opening the hardware.

From `openarm_vla_ws`, install into the interpreter used by Dora:

```bash
bash scripts/install_dora_can_fix.sh
# Or specify another environment (also supported on Nano/aarch64):
bash scripts/install_dora_can_fix.sh /path/to/dora/environment/bin/python
```

The installer builds the vendored source for that Python/architecture and installs
this local node. It does not configure CAN or enable/move motors. The local version
is not available from PyPI; the workspace's real-arm dataflow build commands
include the vendored source, and the installer supports manual setup. Ordinary
`pip install dora-openarm` would select an upstream node instead of this fix.

Feedback-only validation (CAN must already be UP / ERROR-ACTIVE):

```bash
official/dora-openarm-evaluation/.venv/bin/python scripts/validate_openarm_feedback.py can0
```

The extra query latency is about 8 ms per eight-axis batch; enabled motion and
high-frequency control must be validated separately. Do not run this CAN owner
alongside a ROS hardware controller.

## Usage

Use this node from a dora-rs dataflow configuration. For a full configuration
example, see
[enactic/dora-openarm-data-collection](https://github.com/enactic/dora-openarm-data-collection).

```yaml
nodes:
  # ...
  - id: follower-right
    build: pip install dora-openarm
    path: dora-openarm
    args: "--side right --align-trigger gripper"
    inputs:
      # Only the event ID is used. The event value is ignored.
      request_position: leader/right_follower_position
      move_position: leader/right_follower_position
    outputs:
      - position
      - status

  - id: follower-left
    build: pip install dora-openarm
    path: dora-openarm
    args: "--side left --align-trigger gripper"
    inputs:
      # Only the event ID is used. The event value is ignored.
      request_position: leader/left_follower_position
      move_position: leader/left_follower_position
    outputs:
      - position
      - status
  # ...
```

### Node arguments

| Argument | Description |
| --- | --- |
| `--side` | OpenArm side to control. Default: `right`. |
| `--config` | Path to the OpenArm configuration file. Default: `openarm_cell.yaml`. |
| `--align-trigger` | Optional trigger for the initial alignment step. Supported value: `gripper`. |
| `--align-threshold` | Alignment threshold in radians. Default: `0.1`. |
| `--align-delta-limit` | Maximum joint delta per initial-alignment command in radians. Default: `0.001`. |
| `--[no-]align` | Whether to align to incoming position commands after the arm starts. Default: enabled. |
| `--[no-]stop` | Whether to stop the arm when the node exits. Default: controlled by the `STOP` environment variable, or `true` when it is unset. |
| `--[no-]refresh-every-request` | Whether to refresh OpenArm state before each request. Default: controlled by the `REFRESH` environment variable, or `true` when it is unset. |

### Inputs

| Input | Description |
| --- | --- |
| `request_position` | Requests the current arm position. The event ID is used and the event value is ignored. |
| `request_state` | Requests the current arm state. The event ID is used and the event value is ignored. |
| `move_position` | Sends a new target position to the arm. The value may be a struct containing `qpos` (`[{"qpos": [...]}]`), a position array directly, or a legacy struct containing `new_position`. When initial alignment is enabled, this input drives the alignment until it completes. |

### Outputs

| Output | Description |
| --- | --- |
| `position` | Current arm position as a length-1 struct containing a float32 array: `[{"qpos": [...]}]`. |
| `state` | Current arm state as a length-1 struct with list fields: `[{"qpos": [...], "qvel": [...], "qtorque": [...], "tmos": [...], "trotor": [...]}]`. `qpos`, `qvel`, and `qtorque` are float32 lists; `tmos` (MOS temperature) and `trotor` (rotor temperature) are int32 lists per motor, in °C. |
| `status` | Current control state as a string array: `stopped`, `started`, or `aligned`. With alignment enabled, `aligned` is emitted once initial alignment completes. |

## License

Licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

Copyright 2026 Enactic, Inc.

## Code of Conduct

All participation in the OpenArm project is governed by our [Code of Conduct](CODE_OF_CONDUCT.md).
