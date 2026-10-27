> Historical workstation record; paths, results and hardware instructions are not a fresh-clone quickstart. See docs/REPRODUCIBILITY.md.

# Expanded real-robot ACT data gate

As of 2026-10-08, the copied Nano data contains 23 episodes under
`openarm_kinesthetic_data/collections/episodes_pending_review/20261008_*`.
All 23 pass the automatic stream/camera audit, but all 23 remain
`review_status=pending` and `operator_marked_success=false`. They are not yet
eligible for training. Automatic replay completion only proves that the data
streams are usable; it does not prove grasp, placement, or release success.

The exporter now enforces human approval, freezes 30 Hz and 640×360 images,
and records `source_trajectory_trial_id` (falling back to `trial_id` for the
first replay). The split helper assigns all replays of one source trajectory
to one split.

## Manual review

Review every candidate episode before export. Replays of the same source
trajectory share a split group, but each replay still needs a decision because
its camera recording and physical outcome can differ. The interactive reviewer
plays both camera streams and displays measured/reference joint values:

```bash
python scripts/review_real_episodes.py \
  --source openarm_kinesthetic_data/collections/episodes_pending_review \
  --pattern '20261008_*' \
  --write
```

Controls: `SPACE` pause/resume, `A`/`D` seek one second, `R` restart, `Y`
approve, `X` reject, `P` leave pending, `N` next without changing metadata,
and `Q` quit. A short note is requested after each decision. `Y` sets
`review_status=approved` and `operator_marked_success=true`; `X` sets
`review_status=rejected`. Without `--write`, the same command is a dry run.
Rejected episodes are excluded by the split generator and recorded in its
manifest; episodes left pending block the split until reviewed.

For a non-GUI integrity check, use `--check`.

Approve only when the cube is visibly grasped, placed into the black box, and
released, with no camera/data corruption and a sensible return to HOME. Reject
missed grasps, slips, failed placement/release, unsafe or stalled behavior,
severe tracking anomalies, or unusable images.

After reviewing the images and updating each approved metadata file, create a
leakage-safe manifest:

```bash
python scripts/make_real_episode_split.py \
  openarm_kinesthetic_data/collections/episodes_pending_review \
  results/act_real_expanded/episode_split.json \
  --pattern '20261008_*'
```

Then run the converter with that manifest. It refuses pending episodes by
default; `--allow-pending` is intended only for an audit and must not be used
to create a training dataset:

```bash
python scripts/convert_real_episodes_act.py \
  --source openarm_kinesthetic_data/collections/episodes_pending_review \
  --manifest results/act_real_expanded/episode_split.json \
  --roles train validation \
  --output data/lerobot/openarm_real_expanded_20261008 \
  --report results/act_real_expanded/conversion_report.json
```

Do not start ACT training until the manifest and conversion report show only
human-approved episodes and the train/validation/test groups are frozen.

## Combine with the original 13-episode dataset

The original dataset is already a LeRobot dataset. After converting the new
train/validation episodes, merge the two datasets without changing either
source:

```bash
.venv-real-act/bin/lerobot-edit-dataset \
  --new_repo_id local/openarm_real_combined_20261008 \
  --new_root data/lerobot/openarm_real_combined_20261008 \
  --operation.type merge \
  --operation.repo_ids "['local/openarm_real_fixed13_20260930','local/openarm_real_expanded_20261008']" \
  --operation.roots "['/home/lushilin/openarm_vla_ws/data/lerobot/openarm_real_fixed13_20260930','/home/lushilin/openarm_vla_ws/data/lerobot/openarm_real_expanded_20261008']"
```

The combined-data baseline should be trained from scratch first, using
`configs/training/act_real_combined_20261008.yaml`. The existing 13-episode
checkpoint remains valuable as a baseline and for a later warm-start ablation,
but it should not replace the scratch run: it was trained without a held-out
split and its optimizer/model state reflects only the old data distribution.

```bash
OPENARM_TRAIN_LOG=results/act_real_combined_20261008/train.log \
.venv-real-act/bin/python scripts/train_act_smoke_with_log.py \
  --config_path=configs/training/act_real_combined_20261008.yaml
```

Only after the scratch baseline is evaluated should you optionally fine-tune
`outputs/act_real_fixed13_10000_20260930/checkpoints/last/pretrained_model`
on the same combined dataset as a separate experiment. Keep the two output
directories and logs separate so the comparison is meaningful.
