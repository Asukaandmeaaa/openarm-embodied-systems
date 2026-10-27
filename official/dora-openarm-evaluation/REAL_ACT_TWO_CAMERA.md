# Real OpenArm ACT baseline: two cameras

**Status: HOME trajectory audit failed; physical HOME/ACT remain unverified. Updated 2026-10-14.**

A receive-only ROS H1/H2 capture and recorded-trajectory simulation workflow is now
available in the [workstation runbook](../../docs/ROS_H1_H2_CAPTURE_REPLAY_RUNBOOK.txt).
Its new `dataflow-recorded-home-sim-host.yaml` contains no motor node. It does not
replace the physical startup generator or authorize physical replay.

The newly recorded ROS H2 measured path has now passed right-only validation and
completed that simulator graph. The [scope/limit audit](../../results/h2_validation_scope_audit_20261014/REPORT.txt)
corrects the stale simulator J4 bound using the captured ROS URDF and keeps the
measured left arm as collision context without left replay outputs. All 36 real
training episodes have fixed left action channels, although the model remains
16D and this physical graph actively holds the left arm. This simulation result
does not approve the physical startup generator described below.

The 2026-10-14 16:20 run stopped at side-prehome (phase 1), with a 0.064232 rad
residual error; it never reached lifted HOME or opened J8. The offline audit
reproduces table-model penetration along Dora's zero-to-side interpolation.
The final HOME coordinates match ROS and all 36 local episode metadata files,
but the MoveIt collision-aware path was not ported. The installed position
driver also sends zero feedforward torque, unlike the gravity-enabled ROS
launch. The earlier recommendation to run the "corrected HOME" flow is withdrawn,
including startup HOME in the full ACT graph. See the
[audit report](../../results/dora_ros_home_audit_20261014/REPORT.txt) and
[trajectory plot](../../results/dora_ros_home_audit_20261014/paths.png).
The earlier checkpoint below is historical evidence, not current motion clearance.

This document describes the workstation two-camera ACT graph and its software
contract. The current readiness record, physical evidence, J7 diagnosis and
operator gates are in the
[current session handoff](../../docs/ACT_DORA_NEXT_SESSION_HANDOFF.md).

## Current physical checkpoint

The host has passed the following read-only or staged checks:

- Both V4L2 devices captured 30 frames. The ceiling stream negotiated 640x360
  at a measured 26.62 FPS. The wrist stream negotiated 640x480 at 22.07 FPS
  and is resized to the checkpoint's 640x360 input. The closed gripper explains
  the obstructed wrist preview; the view must be checked again after HOME opens
  the gripper.
- `can0` and `can1` each returned 100/100 replies for all eight axes at 20 Hz,
  with no missed cycles, motor errors or bus faults, and all motors disabled.
- The right-arm hold startup reached eight `ENABLED` motors with clean CAN
  health.
- The HOME smoke run reached and verified phase 0. It stopped in phase 1 with:

  ```text
  J7 measured=-0.003624 command=0.126438 error=0.130062 rad limit=0.120000
  ```

There is no final `home_ready`, no verified gripper opening and no successful
physical ACT episode. The saved reports are in
`../../results/physical_feedback_can0_20261014.json`,
`../../results/physical_feedback_can1_20261014.json` and
`../../results/real_camera_readonly/`.

## HOME and coordinate contract

The final data-collection HOME is the target in `guarded-home-right.json`:

```text
[-0.1450262670, -0.0151076284, -0.0060887956, 1.3751977241,
 -0.0031428430, -0.1984151663, -0.1222898194, -0.35]
```

It matches `home_goal_positions_rad` in the recorded collection metadata at
`../../openarm_kinesthetic_data/collections/episodes_pending_review/20261008_091627_012/metadata.json`.
The all-zero vector is a commanded recovery waypoint; it is not encoder zero
calibration and it is not the dataset HOME. The current endpoint sequence
(not a reproduction of the ROS planned path) is:

```text
measured pose -> J1-J7 zero -> side_prehome -> lifted HOME -> J8 open
```

The original phase-1 log is consistent with a command step of approximately
`6.3 rad/s * 20 ms = 0.126 rad`, which is larger than the unchanged 0.120 rad
tracking threshold before the next physical feedback sample. This is the
leading software explanation. It is not a complete physical root-cause proof:
the current log has no per-tick command/feedback timestamps, velocity or
torque trace. The trajectory and guard software tests now pass; the corrected
HOME subsequently failed in the supplied 16:20 run as described above. Do not increase the threshold or
bypass tracking protection.

The graph uses the ROS-trained coordinate configuration
`openarm_real_ros_coordinates.yaml`: zero offsets, ascending send IDs
`0x01..0x08`, receive IDs `0x11..0x18`, CAN-FD on `can0` for the right arm,
and no driver start/stop motion. The known nonascending-ID hazard in
[openarm_can issue #162](https://github.com/enactic/openarm_can/issues/162)
does not match this static configuration. Physical mapping and calibration are
inherited from the ROS validation record; Dora does not schedule duplicate
tests for them.

## ACT graph contract

The workstation graph is
[`dataflow-real-act-two-camera-host.yaml`](dataflow-real-act-two-camera-host.yaml).
The hardware-free graph is
[`dataflow-real-act-two-camera-shadow.yaml`](dataflow-real-act-two-camera-shadow.yaml).
The older Nano-oriented graph is
[`dataflow-real-act-two-camera.yaml`](dataflow-real-act-two-camera.yaml).

The graph is:

```text
evaluation UI and ticks
  -> right/left Dora OpenArm nodes
  -> ceiling + right-wrist camera nodes
  -> observer (16D state + two RGB images)
  -> local LeRobot ACT policy server
  -> actions executor (50 x 16 chunk, right 8D output)
  -> right-arm guarded command boundary and CAN
```

The checkpoint is:

```text
./outputs/act_real_combined_unreviewed_30000_20261008/checkpoints/030000/pretrained_model
```

It expects a 16-dimensional state, ceiling RGB, right-wrist RGB at
`[3, 360, 640]`, and emits a `50 x 16` action chunk. Only the right eight
outputs are eligible for right-arm motion; the left arm is not a policy motion
target in this experiment. The checkpoint remains **unreviewed**.

The policy environment is `./.venv-real-act` and
the Dora environment is
`./official/dora-openarm-evaluation/.venv`.
The vendored CAN package is `1.4.0+dmrefresh1`. Explicit CAN refreshes are
paced by the local feedback driver; cached reads drain received frames but do
not prove a fresh response by themselves.

The observer, policy bridge, local server and actions executor carry source
timing metadata. The current defaults are 200 ms maximum source age, 100 ms
maximum source skew and 500 ms maximum observation age. Stale or unsynchronized
inputs produce a latched timing fault. Physical camera timing and real ACT
timing remain unvalidated.

## Current software status

The checkpoint loaded offline and produced finite `50 x 16` output. The current
guarded HOME implementation uses the ROS endpoints and actual CAN feedback age;
the tracking threshold and hardware limits are unchanged. The evaluation suite
passes 70 tests and both shadow and host graphs build successfully.

The remaining work is defined in the
[current session handoff](../../docs/ACT_DORA_NEXT_SESSION_HANDOFF.md): fixed
observation repeatability, MuJoCo raw chunk recording, expert replay through
the Dora executor, and a bounded first physical ACT action.

## Retired physical runbook

The former sequence—run HOME smoke, let the process exit and disable motors,
then start ACT with a verify-only handoff—is retired.
The available operator commands are limited to the staged right-arm hold and
corrected HOME flows in the current session handoff. The full ACT graph must
wait for the first-action admission and execution budget.

Do not run ROS controllers, an older ACT executor and Dora hardware nodes on the
same CAN bus. Do not treat policy `Connected`, a camera preview, a nominal
timer or a process that remains alive as proof of physical control.
