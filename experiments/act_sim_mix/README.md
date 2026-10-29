# ACT simulation collection, recovery and evaluation

独立的仿真采集、五类失败纠正、数据审核转换与 ACT 测试工具。
代码可在本仓库运行，不依赖开发者原来的 Dora/VLA 工作区；数据和权重由使用者提供。

This package collects contact-dynamics expert demonstrations in MuJoCo and
provides an experimental real/simulation ACT workflow. It does **not** establish
successful policy learning, sim-to-real transfer, or physical deployment.
All commands here operate on simulated or saved inputs. None opens CAN,
starts ROS/Dora hardware nodes, enables motors or opens physical cameras.

## What is included

| Tool | Purpose |
| --- | --- |
| `sim_collect` | Normal expert episodes and five scripted recovery types; real `mj_step` contact, independent replay, atomic publication, writer locks and resumable attempts |
| `prepare_dataset` | Provenance audit, family/seed/layout split isolation and original-JPEG-in-Parquet conversion |
| `adapt_chunk` | Audited ACT 50→100 adaptation; per-parameter change report and byte-identical processor copies |
| `train`, `review_training` | Experimental 30K update recipe, fixed domain sampling, optimizer/RNG recovery, periodic checkpoints and stage diagnostics |
| `prepare_eval_scenes`, `closed_loop` | Source-bound validation/test scenes, task success criteria, saturation/latency/tracking diagnostics |
| `reference_diagnostic` | Recorded expert ZOH, expert through the chunk executor, and policy from the same recorded initial pose |
| `diagnose_policy` | Raw/converted alignment checks, saved-input inference and bounded short rollouts with input-sensitivity probes |
| `metrics`, `evaluate_offline`, `evaluate_session` | Phase/horizon error, missing gripper events, recovery-type summaries and read-only existing-session log analysis |

Recovery types are `closed_approach`, `lateral_offset`, `bad_height`,
`recoverable_drop`, and `pre_release_hold`. These are **scripted perturbations**,
not demonstrations collected from arbitrary failures of the current policy.
The erroneous prefix is retained for diagnosis; conversion includes only the
expert-valid recovery suffix.

## Locked CPU tests and short physics smoke

From the repository root, on Linux with Python 3.12 and `uv`:

```bash
uv sync --frozen --group sim-mix
make test-sim-mix
make smoke-sim-mix
```

The CPU tests use miniature fixtures generated in temporary directories, so no
private demonstrations or checkpoints are needed. ACT-dependent tests skip
when PyTorch is absent. The smoke compiles the included scene and advances
0.1 seconds of physics without rendering. It is not a full grasp success test.

The MJCF references the existing `official/openarm_mujoco/` assets. Model
geometry is unchanged from the source experiment; the include path is relative
to this checkout. Collection and evaluation fingerprints include robot assets.

## Collection: validate, pilot, then batch

`MUJOCO_GL=egl` requires a working EGL implementation; use `osmesa` if that is
the rendering backend available on your machine. Normal/recovery collection
renders 480×640 RGB images for both cameras **after** the contact rollout, from
each exact saved simulation pose. Rendering is not the physical control path.

Run the commands below yourself when ready; full collection can take substantial
time and disk space. Always use a new output for a new source/configuration.

```bash
OUT="$PWD/outputs/act_sim_mix"
mkdir -p "$OUT"

# One normal and each recovery type: dynamics/replay validation, no images.
MUJOCO_GL=egl uv run --frozen --group sim-mix python -m experiments.act_sim_mix.sim_collect \
  validate --seed 20261015 --output "$OUT/validate"

# Small image pilot. Failed attempts are expected and retained.
MUJOCO_GL=egl uv run --frozen --group sim-mix python -m experiments.act_sim_mix.sim_collect \
  normal --episodes 3 --max-attempts 12 --seed 20261028 --output "$OUT/pilot_normal"
MUJOCO_GL=egl uv run --frozen --group sim-mix python -m experiments.act_sim_mix.sim_collect \
  recovery --families 1 --retries 2 --max-attempts 30 --seed 21261028 --output "$OUT/pilot_recovery"
uv run --frozen --group sim-mix python -m experiments.act_sim_mix.prepare_dataset audit \
  --collection "$OUT/pilot_normal" --collection "$OUT/pilot_recovery" --pilot
```

Review `episodes/index.jsonl`, `attempts.jsonl`, and each episode's
`metadata.json`, `quality.json`, `replay.json` and sample camera images. A pilot
report has `complete=false` because it is not the full 100+50 batch; `--pilot`
audits the smaller set without authorizing its conversion as a full batch.

```bash
MUJOCO_GL=egl uv run --frozen --group sim-mix python -m experiments.act_sim_mix.sim_collect \
  normal --episodes 100 --max-attempts 1800 --seed 20261028 --output "$OUT/normal"
MUJOCO_GL=egl uv run --frozen --group sim-mix python -m experiments.act_sim_mix.sim_collect \
  recovery --families 10 --retries 8 --max-attempts 500 --seed 21261028 --output "$OUT/recovery"

uv run --frozen --group sim-mix python -m experiments.act_sim_mix.prepare_dataset audit \
  --collection "$OUT/normal" --collection "$OUT/recovery"
uv run --frozen --group sim-mix python -m experiments.act_sim_mix.prepare_dataset convert \
  --collection "$OUT/normal" --collection "$OUT/recovery" --output "$OUT/converted"
```

This converter intentionally implements the **100 normal + 10 recovery
families × 5 types** protocol: normal split 80/10/10, recovery split 40/5/5.
It refuses incomplete batches and splits that would leak a shared
family/seed/layout. This is not a generic arbitrary-count dataset converter.
Rerun an interrupted collector with identical arguments to resume; retry and
attempt limits are cumulative. Do not delete failed attempts or edit frozen
source manifests to bypass checks. Conversion requires a new output directory.

Raw episodes contain `numeric.npz`, two JPEG directories and quality/provenance
records. Numeric data includes absolute-radian `action`, feedback `state`,
`qpos/qvel`, 30 Hz timestamps, phase, validity and contact/object diagnostics.
The 16-axis order is right J1…J7/gripper, then left J1…J7/gripper. Open right
gripper is −0.45 rad, carry preload −0.085, return/park −0.02. The left finger's
MJCF sign conversion is explicit. These values are demonstration conventions,
not universal robot hard limits.

Converted splits use the project's compact schema: embedded original JPEG
bytes, float32 16-D state/action, per-episode Parquet, and a
`meta/visual_expert_manifest.json`. Images are not expanded to PNG. The custom
reader consumes this schema directly; arbitrary LeRobot datasets may need an
additional adapter.

## Optional real-data audit and ACT training

No real dataset, processors or model weights are distributed. A compatible
real dataset must already have `train/` and `validation/` subdirectories in the
same compact schema. Its manifest needs original episode/event provenance for
stage metrics; this is not inferred from joint motion alone.

Use a separate Python environment with a suitable CUDA PyTorch/torchvision
build and the optional `requirements/act-sim-mix.txt` profile (LeRobot 0.6.1).
For example, after installing that profile, set `ACT_PYTHON` to its interpreter.
The CPU lock does not install or certify the optional CUDA stack.

```bash
export OPENARM_ACT_REAL_DATA=/path/to/real_dataset
ACT_PYTHON=/path/to/act_environment/bin/python

# Optional comparison with real train statistics, without changing statistics.
uv run --frozen --group sim-mix python -m experiments.act_sim_mix.prepare_dataset audit \
  --collection "$OUT/normal" --collection "$OUT/recovery" --real-root "$OPENARM_ACT_REAL_DATA"

"$ACT_PYTHON" -m experiments.act_sim_mix.adapt_chunk \
  --source /path/to/act50/pretrained_model --output models/act100_init/pretrained_model
"$ACT_PYTHON" -m experiments.act_sim_mix.train --check

# Explicit operator-run training. This starts optimizer updates.
"$ACT_PYTHON" -u -m experiments.act_sim_mix.train --output "$OUT/runs/mix30k_01"
"$ACT_PYTHON" -m experiments.act_sim_mix.review_training --run "$OUT/runs/mix30k_01"
```

The default recipe is 30,000 optimizer updates, batch 8 (4 real, 2 normal sim,
2 recovery sim), AdamW at 1e−5, native ACT L1+10KL, gradient clip 10, checkpoint
and validation every 5,000 updates. It uses frozen training-only real-data
processors; validation/test never fit normalization. Real episode counts are
not hardcoded to the original developer's 30/7 split. Use `train --protocol`
with a modified JSON recipe for explicit count assertions or different paths.
All recipe paths are checkout-relative unless absolute.

Adaptation changes only `decoder_pos_embed` and the length-dependent sinusoidal
`vae_encoder_pos_enc`; it reports every changed tensor. New query rows are an
initialization, not already-trained extra actions. This is a new fine-tuning
experiment, not seamless continuation of a 50-step architecture. Training
resume requires a complete checkpoint and unchanged recorded contract; use a
new run output with `--resume /path/to/checkpoint_005000`.

## Freeze and compare simulation evaluations

With optional ACT dependencies, converted data and a trained model:

```bash
uv run --frozen --group sim-mix python -m experiments.act_sim_mix.prepare_eval_scenes \
  --converted "$OUT/converted" --split validation --output "$OUT/scenes_validation.json"

MODEL="$OUT/runs/mix30k_01/checkpoint_030000"
MUJOCO_GL=egl "$ACT_PYTHON" -u -m experiments.act_sim_mix.closed_loop \
  --model "$MODEL" --scenes "$OUT/scenes_validation.json" \
  --exec-steps 50 --command-envelope deployment --output "$OUT/eval/exec50"

MUJOCO_GL=egl "$ACT_PYTHON" -u -m experiments.act_sim_mix.reference_diagnostic \
  --model "$MODEL" --scenes "$OUT/scenes_validation.json" \
  --scene-id validation_normal_000 --scene-id validation_recovery_010 \
  --modes zoh chunk policy --exec-steps 50 --output "$OUT/eval/reference"
```

`closed_loop` predicts 100 targets and executes indices 0…49 through the
attributed Hermite/250 Hz lowpass reference. It then observes again. This is a
simulation execution mode; it does not install a new physical Dora executor.
`--exec-steps` is an evaluation scheduling parameter, not something native ACT
learns automatically. The retained reference requires at least two targets,
so this tool does not implement one-action execution.

The default `strict_expert` envelope rejects demonstration/planner violations.
The explicit `deployment` profile clips after filtering using the included
**historical endpoint snapshot**, reports each saturation, and retains
collision failures. It is not a discovery of your robot's current limits.
Normal benchmark scenes start at deployment zero; recovery scenes restore the
recorded expert takeover state. Reference diagnostics instead use the same
recorded initial pose for all modes and must not replace benchmark results.
Expert chunk timing uses an explicitly reported fixed inference approximation;
policy timing is measured during simulation, not on hardware.

Reports separate arm/gripper errors by horizon and phase, missing opening/
closing/release events, five recovery types, task success, command saturation,
block-boundary jumps, tracking and modeled timing. Save original reports and
use a new output directory for each comparison. A changed implementation or
asset invalidates old scene freezes; regenerate a new freeze rather than
editing hashes. Keep final-test scenes out of model selection.

## Limits and provenance

The scene is approximate: cameras, contact/friction, actuator gains and ideal
gravity compensation are not identified against a physical cell. Smooth
commands or small teacher-forced error do not prove successful grasping or
recovery. No successful ACT policy or sim-to-real result is claimed here.

Only source, example configuration, MJCF and tests are published. Raw images,
demonstrations, trained weights, machine reports, session identifiers and
private experiment notes stay local and are ignored by Git. To relocate your
own frozen data, set `OPENARM_ACT_PATH_MAP` to a JSON object mapping old absolute
roots to new roots; no developer-specific mapping is enabled by default.

See `config/port_manifest.json` for historical port hashes and
`config/reference_provenance.json` for upstream interpolation/filter provenance.
The two upstream reference classes are checked against the included official
source without importing its runtime. The retained Apache-2.0 license is in
`assets/REFERENCE_LICENSE`; see the repository's [NOTICE](../../NOTICE.md).
