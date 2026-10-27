> Historical workstation record; paths, results and hardware instructions are not a fresh-clone quickstart. See docs/REPRODUCIBILITY.md.

# OpenArm ACT training runbook

更新：2026-10-24（Asia/Shanghai）。当前实物数据版本为 **37 条 closed_transit_v2**，其 50K 训练已完成。旧32条、旧27条及下文仿真流程不能作为当前默认入口。

## 当前数据与训练结果

工作区：`/home/lushilin/openarm_vla_ws`。

| 类别 | 合格原始轨迹 | 训练轨迹 | 验证轨迹 |
|---|---:|---:|---:|
| nearby 基本位置 | 30 | 24 | 6 |
| spread 偏移位置 | 7 | 6 | 1 |
| rotation 旋转 | 0 | 0 | 0 |
| 合计 | 37 | 30 | 7 |

训练集 **40,772 帧**，验证集 **9,532 帧**，均按30Hz完整episode导出。nearby `episode_005` 待审核、spread `episode_007` 执行前失败，均未用于训练。所有原始记录和旧27条快照保留；两版验证划分不同，不直接横比损失。

- 数据：`data/lerobot/act_closed_v2_37eps_20261023b/`；[冻结清单](../data/lerobot/act_closed_v2_37eps_20261023b/split_manifest.json)、[导出完成标记](../data/lerobot/act_closed_v2_37eps_20261023b/READY.json)。
- 配置：[act_closed_v2_37eps_20261023b_50k.json](../configs/training/act_closed_v2_37eps_20261023b_50k.json)。
- 训练入口：`vision_grasp_baseline/train_closed.sh` → `scripts/train_act_closed.py`，本地 LeRobot 0.6.1 ACT。
- [训练日志](../results/act_training/act_closed_v2_37eps_20261023b_50k.log)、[实际训练配方](../outputs/act_closed_v2_37eps_20261023b_50k/closed_training_recipe.json)。
- 模型：`outputs/act_closed_v2_37eps_20261023b_50k/checkpoints/`；10个5K间隔存档完整，`last` 指向 `050000`。

日志记录2026-10-23约19:16开始、20:54:55结束，出现 `End of training`；[最终训练状态](../outputs/act_closed_v2_37eps_20261023b_50k/checkpoints/050000/training_state/training_step.json) 为 `step: 50000`。本次文档核对没有重新训练、运行推理或操作机械臂。

### 模型与记录的准确含义

- 输入：顶视 `ceiling` RGB、右腕 `wrist_right` RGB，两路均为640×480；另有16维实测 `JointState.position`，单位rad。
- 动作监督：`JTC output.positions` 形成的16维**绝对位置目标**，不是实测反馈、关节增量、末端XYZ或稀疏规划路点。原始速度记录不作为独立ACT输出。
- 顺序：右J1–J7、右夹爪、左J1–J7、左夹爪；左夹爪标签为初始实测保持值，不能解释成已下发的主动夹爪指令。实物任务只输出右侧8维，左侧保持。
- 覆盖：闭爪零位 → 闭爪接近预抓取点 → 原地开爪 → 下降抓取 → 搬运/释放 → 撤离 → 空爪闭合 → 返回零位；完整双相机时间线保留，无掩码或任务剪辑。
- 每条episode同时保留 `demonstration_action.jsonl`、`target_*.jsonl`、`controller_raw.jsonl`、`joint_observation.jsonl` 和事件/相机记录。原始控制器消息含 `output.positions`、右臂 `output.velocities`、`reference/feedback/error`。
- 对应 `runs/cycle_*/continuous/raw_can.jsonl` 保存本机观察到的CAN指令帧与反馈帧。记录到指令不等于电机实际达到目标，需对照反馈；模型本身不负责安全验收或episode停止。

本版37条共167,788条原生动作标签已核对原始控制器来源、因果对齐、反馈来源和完整阶段。训练导出为独立嵌入原始JPEG字节的Parquet，不依赖采集目录持续在线读取。

### 训练配置与全量验证

Fresh ACT + 缓存的ImageNet ResNet18，未续训旧20K/27条模型；batch8、50K步、chunk50、`n_action_steps=10`。50帧是约1.67秒预测窗，前10帧是约0.33秒的部署队列约定，不代表训练只学习10帧。

`STATE=IDENTITY`、`ACTION=MEAN_STD`、RGB使用ImageNet统计；验证使用训练集归一化统计。每5K保存并验证全部9,532帧，`max_eval_samples=0` 在当前安装版本表示不截断。旧32条入口的256帧前缀验证未覆盖完整任务，不能与本版全量验证等同。

新入口使用本地 `SelectedColumnReader`，读取未来动作窗时先选动作列，避免无关图像重复解码。目标、padding和图像不变，已做逐Tensor一致性与spawn双worker测试；未改ACT核心损失或site-packages。

| 步数 | 日志中的全量验证损失 |
|---:|---:|
| 5K | 0.1631 |
| 10K | 0.1227 |
| 15K | 0.1208 |
| 20K | 0.1115 |
| 25K | 0.1164 |
| 30K | 0.1011 |
| 35K | 0.1070 |
| 40K | 0.1041 |
| 45K | 0.1066 |
| 50K | 0.1015 |

表中为日志打印精度。**30K是本轮日志中验证损失最低的候选，50K是最后存档**，不是已验证“30K实物最好”；两者差0.0004也不足以说明实物效果有显著差异。当前尚未发现37条模型的分阶段离线评估报告或实物成功率记录；已有 `act_closed_smoke_20261023_175210_phase_eval.json` 属于旧27条的两步短测，不能冒充本轮结果。

### 下一步：只做离线评估

以下读取已保存模型和验证数据，不需要H1、网关、相机或CAN；会占用GPU，但不控制机器人。输出文件已存在时应换新文件名，不覆盖结果。

```bash
cd /home/lushilin/openarm_vla_ws
./vision_grasp_baseline/train_closed.sh evaluate \
  --checkpoint outputs/act_closed_v2_37eps_20261023b_50k/checkpoints/030000/pretrained_model \
  --output results/act_training/act_closed37_030000_phase_eval.json

./vision_grasp_baseline/train_closed.sh evaluate \
  --checkpoint outputs/act_closed_v2_37eps_20261023b_50k/checkpoints/050000/pretrained_model \
  --output results/act_training/act_closed37_050000_phase_eval.json
```

比较下降、闭爪、释放、闭爪回零各阶段的右臂/夹爪MAE(rad)，并对照“保持当前姿态”的基线。该评估是示教观测上的离线预测，不是闭环抓取成功率。

### 未完成的部署与采集问题

1. 新采集右臂控制接口为 `position+velocity`；现有旧ACT `ros_bridge.py` 仍要求position-only。新模型预测位置序列，部署前须实现/验证一致的q/qd生成、时序和过滤，不能直接套用旧 `actctl start`。
2. 旧visual32是张爪起点，新数据为闭爪出发，H1、交接和零位验收必须匹配。不要按旧部署文档安装张爪H1候选库来覆盖采集配置。
3. spread `cycle_007` 在控制器就绪查询阶段连续3次超时；[结果](../vision_grasp_baseline/collections/spread/runs/cycle_007/continuous/result.json) 与[失败标记](../vision_grasp_baseline/collections/spread/failed/episode_007_FAILED/FAILED.json) 均证明尚未下发本轮完整轨迹/开始录制。既有重建只读客户端的措施未完全消除该问题，根因未确认，不能记为彻底修复。
4. 当前实际硬件状态未知；此前退出日志或合格数据不代表现在可以恢复运动。本次不重启、不清故障、不放宽CAN/碰撞门槛。
5. 没有旋转类数据，不能据本轮训练宣称旋转/更大范围偏移泛化成功。

### 训练入口使用提醒

本轮50K已经完成，不要在同一输出目录重复启动。`train_closed.sh train` 不带环境变量仍指向旧27条配置；`vision_grasp_baseline/train.sh` 则是更旧32条入口。

本轮历史启动命令如下，仅供核对配置与日志归属，不是要求重跑：

```bash
cd /home/lushilin/openarm_vla_ws
OPENARM_CLOSED_CONFIG=configs/training/act_closed_v2_37eps_20261023b_50k.json \
OPENARM_TRAIN_LOG=results/act_training/act_closed_v2_37eps_20261023b_50k.log \
./vision_grasp_baseline/train_closed.sh train
```

继续采集时保留原始数据并追加编号；新数据不会自动进入冻结快照，另建新name才生成新版本。采集细节见 [当前说明](../vision_grasp_baseline/README.txt) 与[分类/审核规则](../vision_grasp_baseline/collections/README.txt)。

## 以下为旧仿真/早期数据训练历史

旧32条实物流程留存在 [2026-10-22训练记录](ACT_VISUAL32_TRAINING_20261022.txt)，不用于当前37条数据。下方保留原文溯源，不作为当前实物准入要求。

## Readiness criteria

Do not start the multi-episode ACT run until all of these are true:

- at least 20 successful episodes are present (30--50 is preferred);
- every accepted raw episode reproduces task success under strict actuator replay;
- the LeRobot v3 dataset passes a full loader traversal;
- conversion uses `--no-smoothing`;
- ACT state normalization is `IDENTITY` because the inactive left-arm axes have
  near-zero variance; action normalization remains `MEAN_STD`.

The original five-episode pilot is a smoke-test asset, not the final training
set. Strict replay currently passes episodes 0--3 and rejects episode 4, so do
not train a new policy from that five-episode dataset unchanged.

## 1. Collect a fresh dataset

The following asks for 30 committed successful demonstrations. Failed attempts
are retained only in `attempts.jsonl`.

```bash
PYTHONPATH="$PWD:$PWD/official/openarm_mujoco/src" \
python -m openarm_cube_pick_expert.collect \
  --episodes 30 \
  --max-attempts 60 \
  --seed 20261000 \
  --output data/raw/openarm_cube_pick/act_train_30ep
```

## 2. Validate and convert

```bash
python sim/validation/validate_pilot.py \
  data/raw/openarm_cube_pick/act_train_30ep \
  --min-success 30

PYTHONPATH="$PWD:$PWD/official/openarm_mujoco/src" \
python sim/validation/expert_action_replay.py \
  --dataset data/raw/openarm_cube_pick/act_train_30ep \
  --episode all

.venv/bin/openarm-dataset-convert \
  data/raw/openarm_cube_pick/act_train_30ep \
  data/lerobot/openarm_cube_pick_act_train_30ep \
  --format lerobot_v3.0 \
  --state qpos \
  --fps 30 \
  --success-only \
  --no-smoothing

scripts/preflight_act_training.sh \
  data/raw/openarm_cube_pick/act_train_30ep \
  data/lerobot/openarm_cube_pick_act_train_30ep \
  20
```

If strict replay rejects any episode, exclude or recollect that episode before
conversion. A metadata `success: true` flag alone is not sufficient evidence.

## 3. Train

Activate the LeRobot 0.6 training environment, then run:

```bash
conda activate vla
scripts/train_act_openarm.sh \
  data/lerobot/openarm_cube_pick_act_train_30ep \
  outputs/act_openarm_cube_pick_30ep_20k
```

Defaults are 20,000 steps, batch size 8, 10% held-out episodes, checkpoints
every 5,000 steps, and deterministic cuDNN. Override them with environment
variables, for example `STEPS=30000 BATCH_SIZE=4 scripts/train_act_openarm.sh ...`.
