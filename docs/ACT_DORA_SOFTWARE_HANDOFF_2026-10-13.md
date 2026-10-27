> Historical workstation record; paths, results and hardware instructions are not a fresh-clone quickstart. See docs/REPRODUCIBILITY.md.

# ACT + Dora software handoff (2026-10-13)

Last updated: **2026-10-14**. This records the 2026-10-14 workstation evidence
and Dora repair. For the current task and next steps, use the
[current session handoff](ACT_DORA_NEXT_SESSION_HANDOFF.md). The original software validation did not enable motors
or move the robot. The operator subsequently performed the physical checks
recorded below; those results must be distinguished from software-only tests.

**Physical ACT is unverified. Do not repeat the failed HOME smoke or physical
ACT launch sequence.** The old separate
HOME -> process exit -> verify-only ACT procedure is superseded. The Dora
trajectory, tracking and persistent-session changes described below are now
implemented; remaining physical validation is still pending.
The operator performs every motor enable, hold, HOME and physical ACT test.

## Operator checkpoint (2026-10-14)

| Check | Observed result | Scope / remaining gap |
| --- | --- | --- |
| Ceiling camera | 30 frames, 640x360, measured 26.62 FPS | Device capture passed; simultaneous Dora timing still unvalidated |
| Right wrist camera | 30 frames, native 640x480, measured 22.07 FPS; resized to 640x360 | Closed gripper explains the obstructed view; verify training-view match after opening |
| can0 feedback | 100/100 replies per axis at 20 Hz, no misses/errors, all motors disabled; max read 9.447 ms | Feedback-only pass |
| can1 feedback | 100/100 replies per axis at 20 Hz, no misses/errors, all motors disabled; max read 9.116 ms | Feedback-only pass |
| Right hold, 11:04 | Eight `ENABLED` motors, clean CAN health, measured-pose hold initialized | Operator log confirms startup; no sustained tracking trace supplied |
| Right HOME, 11:06 | Phase 0 return-to-zero verified; phase 1 stopped on J7 tracking error | No final `home_ready`, no verified gripper opening |
| Physical ACT | No successful run established | Blocked on the work below; checkpoint remains unreviewed |

Saved CAN reports:
[can0](../results/physical_feedback_can0_20261014.json),
[can1](../results/physical_feedback_can1_20261014.json).
Camera previews are in [real_camera_readonly](../results/real_camera_readonly/).
The hold/HOME event details above come from the operator's pasted terminal
logs, not a complete sampled command/feedback recording. Basic CAN repair
should not be repeated without new fault evidence.

The first can0 attempt was rejected because the interface was not
UP / ERROR-ACTIVE; the later saved report passed. Treat that earlier interface
state separately from the phase-1 J7 failure.

## HOME audit and confidence of diagnosis

Three different concepts apply: motor encoder zero, the fixed data-collection
HOME target, and the trajectory used to reach that target. `return_to_zero`
commands joint motion; it does not call `set_zero` or recalibrate encoders.

The current [HOME configuration](../official/dora-openarm-evaluation/guarded-home-right.json)
uses this final J1-J8 target, in radians:

```text
[-0.1450262670, -0.0151076284, -0.0060887956, 1.3751977241,
 -0.0031428430, -0.1984151663, -0.1222898194, -0.35]
```

It exactly matches the seven-axis `home_goal_positions_rad` and gripper
target in the [recorded collection metadata](../openarm_kinesthetic_data/collections/episodes_pending_review/20261008_091627_012/metadata.json),
which labels the episode HOME `fixed_lifted_home` and excludes initialization
from training data. This is stronger evidence than a generic simulation HOME
candidate in another configuration file. Matching endpoints does not establish
that Dora preserves the collection controller's intermediate path or timing.

The existing route in [guarded_home.py](../official/dora-openarm-evaluation/nodes/dora-openarm/src/dora_openarm/guarded_home.py)
is:

```text
measured pose -> J1-J7 zero -> side_prehome -> fixed lifted HOME -> J8 open
```

Phase 0 interpolates the measured pose to zero over 200 nominal 10 ms steps,
then requires measured settling. J8 stays at its measured starting position
until the final phase. At 11:06:19 phase 0 passed; the following phase failed:

```text
HOME_TARGET: phase=1 state=homing ... J7 target=0.1343242132
HOME_ERROR: phase=1 state=homing: command tracking error:
J7 measured=-0.003624 command=0.126438 error=0.130062 rad limit=0.120000
```

Confirmed in code: tracking is checked against the last dispatched command;
HOME phase motion uses per-axis velocity limits as its slew rate. J7's
`6.3 rad/s` permits a `0.126 rad` command increment at a 20 ms interval, already
larger than the `0.120 rad` tracking threshold. Scheduling delay and feedback
latency therefore matter even with healthy CAN. A velocity ceiling alone is
not a time-parameterized HOME trajectory.

This design mismatch is consistent with the logged command and is the leading
software explanation. The log has no per-tick command, motor-response timestamp,
velocity or torque trace, so it does not prove that the hardware was merely
one sample late, or exclude stiffness, loading, obstruction or calibration
problems. Do not label the physical root cause fully confirmed or simply
increase the threshold. The nominal 100 Hz timer, Guard's 20 ms refresh
request interval and the driver's 20 Hz explicit-refresh cap are distinct;
RX draining may provide newer feedback between explicit requests.

The runtime coordinate config uses zero offsets and ascending send/receive
IDs (J7 `0x07` / `0x17`). The nonascending-ID failure in
[openarm_can #162](https://github.com/enactic/openarm_can/issues/162)
does not explain this configuration; static configuration inspection does not
prove the physical motor labels or encoder calibration.

The ROS control pipeline already supplied the hardware validation for the
mechanical zero jig, URDF/physical frames, J7 command/feedback mapping,
gravity compensation, and J7 stiffness. Dora reuses that ROS coordinate
configuration and gains. This work therefore does not schedule duplicate
hardware tests for those items; it is limited to the Dora trajectory,
dispatch timing, and tracking boundary.

## Remaining preparation plan

The target milestone is one bounded, supervised right-arm ACT smoke test.
Preserve the dataset HOME, `tracking_tolerance=0.120`, joint bounds, CAN patch,
watchdogs and existing environments. Do not recalibrate encoders, increase PD
gains or relax protections as part of this first batch.

| Priority | Change and affected files | Required evidence before advancing |
| --- | --- | --- |
| P0-A | **Implemented:** repair HOME timing in `guarded_home.py` with a smooth, bounded trajectory and the existing safety limits | No-hardware tests cover the J7 transition margin; physical route confirmation remains part of the supervised test |
| P0-B | **Implemented:** add delayed-feedback and scheduling-jitter coverage in `tests/test_guarded_home.py` and driver-boundary tests | Healthy delayed feedback, jam, stale feedback, out-of-bounds input, watchdog and driver rejection cases pass; no phase advance before measured settling |
| P1-A | **Implemented in the host graph:** keep one right-arm CAN owner for HOME through ACT and retain the fresh HOME gate | Graph/build and no-hardware checks pass; camera/policy readiness and left-arm physical behavior remain staged checks |
| P1-B | Add first-action admission and an enforced short-test execution budget at the guarded command boundary and executor | Finite 16D policy output, correct right-8D extraction, measured-HOME continuity, bounded interpolation including J8; queued tails cancelled on expiry/STOP; all right and left command outputs accounted for |
| P1-C | Add a real-camera observation-only graph/logger and fresh camera/policy readiness in the UI | Simultaneous source sequences/ages/skew, camera FPS/jitter, inference latency and action-age reports; missing/stalled camera or delayed policy blocks execution |
| P2 | Publish exact operator commands for the tested revision and run the staged physical checks | Supported hold/HOME, gripper and camera verification, bounded ACT smoke, saved logs reviewed before longer trials |

The P0-A and P0-B software changes are implemented in this revision and are
covered by the no-hardware regression suite. P1-C can be prepared in parallel
without motor access; all remaining P0/P1 gates must pass before physical ACT.

Implementation requirements:

- Seed commands from fresh measured feedback. Choose a HOME-specific,
  time-parameterized trajectory with smooth phase boundaries; do not use
  maximum joint velocity as the requested HOME speed. Correlate tracking
  diagnostics with actual per-axis feedback and dispatch timestamps, including
  repeated cached reads. Bound command lead before it exceeds the existing
  protection, rather than masking an actual `>0.120 rad` error with a blanket
  grace period. Persistent lack of progress must fault even if a slower command
  stays below that bound. No watchdog bypass or automatic restart.
- Review whether the all-zero intermediate is needed for the known startup
  pose. A proposed measured -> validated waypoint(s) -> dataset HOME route is
  a design candidate, not a collision-validated replacement yet. Retain the
  explicit startup-envelope checks needed for the previously observed negative
  J4 pose; reject unsupported startup poses. Do not remove recovery protection
  or substitute a joint-space straight line without checking the route.
- Keep `test_fault_after_phase_zero_is_visible_without_ui`: it models a motor
  that never follows, so its latched fault is correct. Add a separate realistic
  delayed-response regression. The earlier suggestion to remove that jam test
  conflated missing transient coverage with an existing valid failure test.
- Record requested target, actual dispatched target, measured J1-J8, available
  velocity/torque, per-axis feedback timestamp/sequence, phase, `dt`, CAN health
  and first fault. Retain the timestamp clock/unit and configuration/model
  hashes in each report. Asynchronous/buffered logging must not stall control.
- The ACT host graph now keeps one right-arm Dora session from corrected HOME
  through observation and ACT. It performs HOME at graph startup, keeps the
  CAN owner alive, and accepts START only after fresh `HOME_READY`; there is no
  HOME-process-exit/verify-only handoff.
- START must require fresh HOME verification, both camera sources, both arm
  observations and a valid policy response. To avoid a readiness deadlock,
  provide a pre-START observation-only state where inference can run but no
  policy target reaches CAN. The arm command boundary must enforce this gate;
  UI status alone is insufficient. Keep the inactive left arm out of policy
  motion while checking its real feedback and hold behavior.
- The proposed 1-2 s first ACT trial is an execution budget to implement and
  test, not the current default (`episode_timeout` is currently 30 s). Do not
  rely on a human counting seconds or change the 30 Hz model action timebase.
  Freeze/cancel further policy targets at the budget boundary using the defined
  stop behavior. A logical hold is not guaranteed physical immobilization;
  emergency disable and supported process shutdown remain separate operations.
- Camera soak uses real cameras with replay/synthetic state and an action log
  sink only; label those state inputs accurately and do not include enabling
  arm nodes. A complete real-camera plus real-joint check follows in the
  operator's supported session. Host OpenCV timestamps are not sensor exposure
  timestamps; backend buffering and both resized training views still need
  verification. Retain the existing 200/100/500 ms timing limits and exercise
  camera stalls, excessive skew, delayed inference and expired chunk tails.

Mechanical calibration, URDF/physical frames, J7 command/feedback mapping,
gravity compensation and J7 stiffness are inherited from the ROS validation
record and are not Dora work items. Reopen one only if new evidence directly
contradicts that validated contract; do not schedule duplicate hardware tests
as a prerequisite for this software pipeline.

## Community references and their applicability

No exact public J7/Dora HOME failure matching this log was found in the earlier
search. These are related evidence, not validated fixes for this robot:

- [ROS2 #66](https://github.com/enactic/openarm_ros2/issues/66): zero-initialized
  commands caused startup motion; measured-state seeding and removing automatic
  return-to-zero were discussed.
- [ROS2 #68](https://github.com/enactic/openarm_ros2/issues/68): tracking under
  load improved with gain changes in that setup; it does not justify copying
  V10 gains into this Dora deployment.
- [ROS2 #77](https://github.com/enactic/openarm_ros2/issues/77): transient torque
  relaxation on trajectory changes involves the ROS trajectory controller;
  use it to guide diagnostics, not as proof of the same Dora cause.
- [OpenArm #323](https://github.com/enactic/openarm/issues/323) and the
  [calibration workflow](https://docs.openarm.dev/hardware/openarm-cell/calibration-workflow/):
  motor zero calibration has its own physical procedure, separate from HOME.
- [Driver README](https://github.com/enactic/openarm_driver): command dispatch
  success is not measured arrival. Current local implementation must still be
  checked against the pinned installed version.

## Readiness correction and fix (2026-10-14)

The successful transport and MuJoCo tests do not establish readiness for
physical ACT execution. The audit found that the Dora observer reused its
latest cached inputs and stamped each emitted observation with the current
time; it did not reject stale camera frames or excessive source timestamp
skew. The policy bridge also dropped source timing from action metadata. The
stale-snapshot probe tested the separate ROS snapshot publisher, not this Dora
path.

The Dora path now carries and checks source timing end to end. The observer
rejects missing, future, stale, or unsynchronized arm/camera inputs before
emitting an observation. The workstation graph uses a local camera node that
continuously drains each V4L2 device into one latest-frame slot, stamps the
capture with host monotonic time, and never republishes a cached frame as new.
The physical OpenArm nodes stamp positions from the oldest actual CAN reply.
The policy client, local policy server, and actions executor require the timing
metadata and reject an old observation or action chunk. Defaults are 200 ms
maximum source age, 100 ms maximum source skew, and 500 ms maximum observation
age. Rejection is logged, the current action chunk expires at the original
observation deadline, and a latched `timing_fault` stops the guarded arm.
Physical camera timing and physical ACT remain unvalidated until the staged
host procedure below is completed.

Motor-feedback and control-tick watchdog tests are separate evidence. Do not
use the hold diagnostic or the real ACT graph as a motors-disabled connection
test: the guarded hardware node enables motors and sends a measured-position
hold during startup, before the UI START gate.

## Current architecture

The real graph is:

```text
evaluation UI and ticks
  -> right/left OpenArm Dora nodes (feedback and guarded command boundary)
  -> ceiling + right-wrist camera nodes
  -> observer (16D state + two RGB images as Arrow)
  -> local LeRobot ACT policy server (Unix socket)
  -> actions executor (50 x 16 chunk, right 8D output)
  -> right OpenArm controller and CAN layer
```

The Nano graph is
[dataflow-real-act-two-camera.yaml](../official/dora-openarm-evaluation/dataflow-real-act-two-camera.yaml).
The workstation copy for the connected host is
[dataflow-real-act-two-camera-host.yaml](../official/dora-openarm-evaluation/dataflow-real-act-two-camera-host.yaml);
it points at the local checkpoint and the host's camera aliases and wires
timing faults into the guarded arm boundary. The legacy
ROS ACT bridge/executor is a separate path and must not run with Dora.

For safe workstation validation,
[dataflow-real-act-two-camera-shadow.yaml](../official/dora-openarm-evaluation/dataflow-real-act-two-camera-shadow.yaml)
replaces every hardware and V4L2 node with headless MuJoCo. It contains no
CAN, ROS, motor, HOME, or physical camera node.

## Checkpoint and environment

Checkpoint:

```text
/home/lushilin/openarm_vla_ws/outputs/act_real_combined_unreviewed_30000_20261008/checkpoints/030000/pretrained_model
```

The checkpoint declares `observation.state` `[16]`, ceiling and right-wrist
RGB images `[3, 360, 640]`, action `[16]`, `chunk_size=50`, and
`n_action_steps=10`. It remains **unreviewed**; these checks do not establish
task success.

The ACT environment is `.venv-real-act` (LeRobot 0.6.1, Torch 2.11.0+cu128,
CUDA available). The Dora environment is
`official/dora-openarm-evaluation/.venv`. Model loading and server inference
used local checkpoint files with `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`.
Dora builds used their existing dependency installation commands.

## Software validation commands and recorded results

The inference/IPC results below are prior software evidence. For the current
HOME/session revision, the evaluation suite passed **70 tests**, Python compile
checks passed, and both shadow and host graphs built successfully. The attempted
shadow run stopped because its local policy socket was absent, so it does not
count as a current end-to-end pass. No CAN node was run.

Offline model and processor validation:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .venv-real-act/bin/python scripts/validate_real_act_checkpoint.py \
  outputs/act_real_combined_unreviewed_30000_20261008/checkpoints/030000/pretrained_model
```

Result: checkpoint load `0.557 s`, preprocessing plus prediction `0.262 s`,
both expected camera features accepted, and finite output shape `50 x 16`.
The helper labels its timing `predict_post_s`, but ends that timer before
postprocessing; it is not a complete latency benchmark. Offline environment
flags were set; these flags alone do not constitute network isolation.

Start the workstation policy server:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
OPENARM_PRETRAINED_PATH=/home/lushilin/openarm_vla_ws/outputs/act_real_combined_unreviewed_30000_20261008/checkpoints/030000/pretrained_model \
SOCKET=/dev/shm/openarm_official_act_policy.sock \
REQUIRE_SOURCE_TIMING=yes MAX_SOURCE_AGE_MS=200 \
MAX_SOURCE_SKEW_MS=100 MAX_OBSERVATION_AGE_MS=500 \
.venv-real-act/bin/python official/dora-openarm-evaluation/src/local_policy_server.py \
  /dev/shm/openarm_official_act_policy.sock
```

With that server running, the Arrow IPC smoke test is:

```bash
.venv-real-act/bin/python scripts/test_local_policy_server_ipc.py \
  /dev/shm/openarm_official_act_policy.sock
```

Result: synthetic 16D state and two 360×640 RGB frames produced finite
`(50, 16)` positions, interval `33333333` ns, and cutoff `15` Hz.

Build and run the safe Dora graph:

```bash
cd /home/lushilin/openarm_vla_ws/official/dora-openarm-evaluation
.venv/bin/dora build dataflow-real-act-two-camera-shadow.yaml --uv --local
.venv/bin/dora run dataflow-real-act-two-camera-shadow.yaml --uv --stop-after 8s
```

The build completed. The run showed MuJoCo camera renderers, observer output,
repeated policy chunks with `n=50 dim=16`, action replacement at about 10 Hz,
100 Hz executor upsampling, and clean shutdown. It also logged deliberate
`OBSERVATION_REJECTED` messages for source skew over the 100 ms limit while
continuing with later fresh observations. No hardware node or CAN interface
was present. The real graph was build-checked without starting it:

```bash
.venv/bin/dora build dataflow-real-act-two-camera-host.yaml --uv --local
```

This resolved the vendored `openarm-can==1.4.0+dmrefresh1` package and all
nodes successfully; build-only validation did not initialize CAN.

Safety and timing tests:

```bash
.venv/bin/python -m pytest -q tests nodes/dora-openarm-observer/tests
```

Recorded result: `65 passed` in the evaluation suite, plus 5 fake-only root
read-only-probe tests. These no-CAN tests cover stale motor feedback rejection,
watchdog timeout, action timeout, tracking and position bounds, startup health,
Arrow observation construction, and stale/source-skew rejection in the
observer, policy bridge, local server, and actions executor. Separately,
`scripts/real_act_inference.py` rejected a synthetic 12.9 s old ROS snapshot
before image decode and raised its idle timeout without writing an action file.
That script is not in the Dora graph.

## Changes made for this handoff

- Added the workstation-only shadow graph and the host physical graph.
- Fixed the right-arm-only and bimanual output boundary in
  `nodes/dora-openarm-actions-executor/src/dora_openarm_actions_executor/main.py`
  so Dora receives Arrow float arrays rather than NumPy arrays.
- Added source-age/skew metadata and rejection gates across the Dora observer,
  policy bridge, local server, and actions executor. Running chunks expire at
  the original observation deadline and publish a latched `timing_fault` to
  the guarded arm. Tests are in `tests/test_observation_timing.py` and
  `tests/test_action_freshness.py`.
- Added the source-stamped `dora-openarm-camera` node and fake-camera tests in
  `nodes/dora-openarm-observer/tests/test_camera.py`.
- Added repeatable offline validation helpers:
  `scripts/validate_real_act_checkpoint.py` and
  `scripts/test_local_policy_server_ipc.py`.
- Added read-only host checks in `scripts/validate_openarm_feedback.py` and
  `scripts/validate_real_cameras_readonly.py`. The feedback check fails if any
  motor is already enabled; it never enables, disables, or commands a motor.
- Added host-only guarded hardware flows:
  `dataflow-right-can-hold-diagnostic-host.yaml` and
  `dataflow-right-home-smoke-host.yaml`.

## Validation gates after implementation

The no-CAN suite and both graph builds now pass for this revision. The nominal
shadow run, camera/policy soak and supervised physical checks remain pending.

1. **Passed:** run the no-CAN suite, including new realistic feedback-delay,
   trajectory and driver-boundary tests. Remaining first-action and stop work
   must be covered when implemented.
2. Run the hardware-free shadow graph for at least 30 s; inspect nominal
   throughput and deliberate fault cases separately. A graph merely staying
   alive while discarding every observation is not a successful nominal run.
3. **Build passed:** the revised host graph resolves the pinned CAN version
   without running hardware nodes. Save source/configuration/model hashes for
   the supervised physical revision.
4. Run the proposed real-camera observation-only soak, targeting at least 100
   valid policy chunks after warm-up, plus separately recorded fault cases.
   Confirm that no policy output has a route to a physical command sink.
5. The operator repeats read-only CAN/camera checks when needed after setup or
   software changes, then conducts supported hold and corrected HOME tests.
   Require measured settling, no health/tracking faults, correct J8 opening,
   and an unobstructed wrist view matching data collection. Save time-series
   logs; startup messages alone do not pass the sustained hold test.
6. Once a software-enforced 1-2 s ACT budget and first-action gate have been
   implemented and tested, use one CAN-owning session to verify live
   arm/camera/policy readiness; the operator then explicitly starts the
   bounded ACT test. Use the trained
   scene arrangement, reviewed first-action predictions and a clear initial
   motion segment. Do not assume an arbitrary empty scene is model-equivalent.
7. Review command-versus-feedback, stop behavior, intervention and timing logs
   before extending duration or attempting the complete manipulation task.

Exact motion commands will be published for the verified revision after the
software gates pass. The host HOME/ACT YAML now keeps the right-arm CAN owner
alive from corrected HOME through ACT; the old launch chain must not be used as
the next step.

## Operator read-only checks

These existing checks remain useful; their 2026-10-14 results are recorded
above. They do not authorize or automatically advance to an enabled-motor
test. The feedback probe sends CAN read requests but no motion/enable/disable
commands. The operator handles hardware ownership and physical setup.

1. From a terminal on the connected host, confirm CAN is up and no competing
   owner is running. These commands are read-only:

   ```bash
   cd /home/lushilin/openarm_vla_ws
   ip -details link show can0
   ip -details link show can1
   ps -ef | grep -E 'ros2|dora|openarm|act|camera|can0|can1' | grep -v grep || true
   ```

   Stop and close any ROS controller, ROS ACT bridge/executor, or C1/C2 camera
   owner before continuing. Do not run ROS and Dora controllers on the same CAN
   buses.

2. With the arm mechanically supported and the emergency stop ready, check the
   two camera devices without starting CAN, ROS, or Dora:

   ```bash
   cd /home/lushilin/openarm_vla_ws
   .venv-real-act/bin/python scripts/validate_real_cameras_readonly.py \
     --ceiling /dev/v4l/by-id/usb-Intel_R__RealSense_TM__Depth_Camera_435i_Intel_R__RealSense_TM__Depth_Camera_435i_254343063881-video-index2 \
     --wrist-right /dev/v4l/by-id/usb-DCX-250107-ZW_DECXIN-video-index0 \
     --frames 30 --width 640 --height 360 \
     --output-dir results/real_camera_readonly
   ```

   Inspect the saved `*_captured.png` and `*_policy_640x360.png` files. Confirm
   the intended camera devices, consistent captured shapes, and a policy
   preview of 640×360×3. A failed open/read or unexplained black image needs
   investigation. Nearly identical frames of a stationary scene do not prove
   a frozen stream; image mean variation alone is not a freshness test. The
   operator confirmed that the closed gripper obstructs the wrist view before
   HOME. That expected obstruction does not block motors-disabled CAN feedback
   checks; confirm the unobstructed task view after HOME and before ACT. Do not
   re-aim the camera merely to compensate for the closed gripper. DECXIN
   `video-index0` is the image stream; `video-index1` is metadata only. Its
   reported formats do not include native 640×360, so the observed 640×480
   capture and resize are expected; training-view compatibility still needs
   checking before ACT. This check opens each V4L2
   device sequentially and releases it before the next one; a stuck V4L2 open
   or read is killed after 20 seconds.

3. Run the motors-disabled CAN feedback check, one bus at a time:

   ```bash
   cd /home/lushilin/openarm_vla_ws
   mkdir -p results
   official/dora-openarm-evaluation/.venv/bin/python scripts/validate_openarm_feedback.py \
     can0 --cycles 100 --hz 20 \
     --output "results/physical_feedback_can0_$(date +%Y%m%d_%H%M%S).json"

   official/dora-openarm-evaluation/.venv/bin/python scripts/validate_openarm_feedback.py \
     can1 --cycles 100 --hz 20 \
     --output "results/physical_feedback_can1_$(date +%Y%m%d_%H%M%S).json"
   ```

   The expected result is `ok: true`, 100 replies per axis, zero missed cycles,
   zero bus faults, zero motor error codes, and every `motor_enabled` value
   `false`. The validator only reads feedback; it does not start, stop, enable,
   disable, calibrate, or move a motor. If either command reports an enabled motor,
   a CAN fault, a missed reply, or an error code, stop and leave the arm
   unchanged for operator investigation.

Enabled-motor launch commands from the previous version have been withdrawn
from this current runbook. Follow the pending validation gates above before
issuing a revised operator procedure. A logical `home_error`/STOP can leave
motors enabled at the last target; it is not proof of motor disable. Process
exit and emergency disable have different behavior and require physical
support. No software test or documentation update constitutes physical ACT
success.
