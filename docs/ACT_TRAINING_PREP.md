> Historical workstation record; paths, results and hardware instructions are not a fresh-clone quickstart. See docs/REPRODUCIBILITY.md.

# OpenArm ACT training preparation

2026-10-24：当前实物数据为 nearby30＋spread7，共37条 closed_transit_v2，已完成30/7完整episode划分及50K训练。请先读 [当前数据、训练结果与离线评估](ACT_TRAINING_RUNBOOK.md)；不要使用旧32条的 `vision_grasp_baseline/train.sh` 准备当前数据。

以下保留早期仿真/四轨迹准备流程供溯源，其中数据数量、MuJoCo回放要求和训练命令不是本批37条实物数据的当前操作单。

The training gate distinguishes two claims:

1. **Memorization feasibility:** ACT can reproduce the replay-valid trajectories
   already in the workspace.
2. **Generalization feasibility:** ACT succeeds on held-out object poses and
   yaws after training on a larger demonstration set.

The current raw pilot contains five episodes. Strict actuator replay accepts
episodes `0,1,2,3`. Episode `4` reaches the lift-height threshold but maintains
the complete success condition for only `0.311 s`, so it is excluded from
training even though its recorded metadata says `success: true`.

## 1. Run the preflight

```bash
PYTHON_BIN=/home/lushilin/miniconda3/envs/vla/bin/python \
scripts/preflight_act_training.sh
```

The gate checks the OpenArmDataset structure, cameras and state/action streams,
strictly replays every expert action through MuJoCo, traverses the converted
LeRobot v3 dataset, and audits low-variance state dimensions.

## 2. Run the four-episode memorization gate

This run uses only the four replay-valid episodes. It deliberately disables the
evaluation split and gives the data approximately 24 passes, matching the
exposure of the previously successful single-episode experiment.

```bash
conda activate vla
export PYTHONPATH="$PWD:$PWD/official/openarm_mujoco/src:$PWD/official/openarm_dataset/src"
OPENARM_TRAIN_LOG=results/act_training/act_openarm_cube_pick_4ep_identity_12000.log \
python scripts/train_act_smoke_with_log.py \
  --config_path=configs/training/act_openarm_cube_pick_4ep_memorization.yaml
```

`STATE=IDENTITY` is intentional. Do not restore ACT's default `STATE=MEAN_STD`
for this dataset: unused left-arm dimensions have near-zero standard deviation
and turn sub-milliradian reset differences into extreme normalized inputs.

## 3. Collect enough data for a generalization claim

Four trajectories can prove memorization, not generalization. Collect at least
30 replay-valid demonstrations; 50 is the recommended target for the first
proper ACT experiment. Use a new output directory and a non-overlapping seed
range:

```bash
conda activate vla
export PYTHONPATH="$PWD:$PWD/official/openarm_mujoco/src:$PWD/official/openarm_dataset/src"
python -m openarm_cube_pick_expert.collect \
  --episodes 50 \
  --max-attempts 100 \
  --seed 20261000 \
  --output data/raw/openarm_cube_pick/expert_robust_50
```

First run the raw/replay part of the preflight and keep its report:

```bash
RAW_DATASET=data/raw/openarm_cube_pick/expert_robust_50 \
MIN_ROBUST_EPISODES=30 \
SKIP_LEROBOT=1 \
REPORT=results/openarm_expert/expert_robust_50_replay.json \
scripts/preflight_act_training.sh
```

Build a clean OpenArmDataset view containing only episodes that passed strict
replay. The default uses symlinks, so it does not duplicate the JPEG data:

```bash
python scripts/filter_replay_validated_openarm.py \
  data/raw/openarm_cube_pick/expert_robust_50 \
  results/openarm_expert/expert_robust_50_replay.json \
  data/cleaned/openarm_cube_pick_expert_robust
```

Convert that clean view with conversion-time smoothing disabled:

```bash
PYTHONPATH="$PWD/official/openarm_dataset/src" \
/home/lushilin/miniconda3/envs/vla/bin/python -c \
  'from openarm_dataset.convert import main; main()' \
  data/cleaned/openarm_cube_pick_expert_robust \
  data/lerobot/openarm_cube_pick_expert_robust_v3 \
  --format lerobot_v3.0 \
  --state qpos \
  --fps 30 \
  --success-only \
  --no-smoothing
```

Now run the complete gate against the cleaned raw view and its converted data:

```bash
RAW_DATASET=data/cleaned/openarm_cube_pick_expert_robust \
LEROBOT_DATASET=data/lerobot/openarm_cube_pick_expert_robust_v3 \
REPO_ID=local/openarm_cube_pick_expert_robust_v3 \
MIN_ROBUST_EPISODES=30 \
scripts/preflight_act_training.sh
```

Do not train from the recorder's `success` flag alone.

After at least 30 replay-valid episodes remain, start the generalization run:

```bash
conda activate vla
export PYTHONPATH="$PWD:$PWD/official/openarm_mujoco/src:$PWD/official/openarm_dataset/src"
OPENARM_TRAIN_LOG=results/act_training/act_openarm_cube_pick_robust_100000.log \
python scripts/train_act_smoke_with_log.py \
  --config_path=configs/training/act_openarm_cube_pick_robust.yaml
```
