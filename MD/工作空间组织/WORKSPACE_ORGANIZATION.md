> Historical workstation record; paths, results and hardware instructions are not a fresh-clone quickstart. See docs/REPRODUCIBILITY.md.

# 工作空间组织（修改于 2026-09-11）

- 建立日期：2026-09-03
- 修改日期：2026-09-11

## 文档入口

- `MD/当前阶段/蓝色方块_固定黑盒_ACT_200轨迹_50k训练与30seed闭环_建立2026-09-11_修改2026-09-11.md`：当前最新权威状态。
- `MD/README.md`：当前文档索引和结论摘要。
- `MD/当前阶段/蓝色方块_ACT_200轨迹_50k训练与闭环验证_建立2026-09-09_修改2026-09-09.md`：上一阶段附近随机放置状态与可视化报告。
- `MD/当前阶段/蓝色方块_ACT_200轨迹与50k训练前状态_建立2026-09-08_修改2026-09-08.md`：上一阶段训练准备快照。
- `MD/当前阶段/蓝色方块_ACT_100轨迹与20k训练状态_建立2026-09-08_修改2026-09-08.md`：上一阶段基线报告。
- `MD/历史任务文档/`：旧阶段报告，仅用于追溯。
- `MD/问题修复记录/`：独立问题及修复说明。

## 当前任务实现

- `openarm_cube_into_black_box_expert/`：固定黑盒任务contract、场景位置和完整候选预演入口。
- `configs/collection/openarm_cube_into_black_box_expert_params.yaml`：黑盒、随机范围、运输和落盒参数。
- `configs/collection/openarm_cube_into_black_box_metadata.yaml`：固定黑盒任务prompt和数据合同。
- `openarm_cube_pick_expert/collect.py`：蓝色方块完整任务采集、连续轨迹执行和事务式提交。
- `openarm_cube_pick_expert/expert.py`：IK、抓取候选和完整 FSM 阶段定义。
- `openarm_cube_pick_expert/guard.py`：16D position action 安全网关。
- `configs/collection/openarm_cube_pick_expert_params.yaml`：随机范围、速度、HOLD、放置和 Guard 参数。
- `sim/validation/expert_action_replay.py`：严格 actuator replay 和完整流程判定。
- `sim/validation/replay_expert_viewer.py`：专家动作可视化，支持同一 MuJoCo 窗口腕部画中画。
- `sim/validation/act_closed_loop.py`：ACT→官方 executor→MuJoCo 闭环评估。

## 当前蓝色数据

### 最新固定黑盒数据

```text
data/raw/openarm_cube_into_black_box/expert_blue_box_200_20260909/
data/cleaned/openarm_cube_into_black_box_train160_20260909/
data/cleaned/openarm_cube_into_black_box_holdout40_20260909/
data/lerobot/openarm_cube_into_black_box_train160_20260909/
data/lerobot/openarm_cube_into_black_box_holdout40_20260909/
```

- 216 attempts、200 online success、严格 replay `200/200`；
- 固定切分160/40，split seed `20260909`；
- train 185,890 frames，holdout 46,429 frames；
- 转换采用LeRobot v3、qpos、30 Hz、`--no-smoothing`；
- train与holdout逐帧验证均为 `valid=true`。

以下数据属于上一阶段附近随机放置任务。

### 原始主批次

```text
data/raw/openarm_cube_pick/expert_blue_pick_place_100_20260907/
```

- 大小约 6.2 GB；
- 143 attempts；
- 在线成功 100；
- 严格 replay 有效 96；
- 回放证据：`results/expert_blue_pick_place_100_replay_report.json`。

### 补采批次

```text
data/raw/openarm_cube_pick/expert_blue_pick_topup_20260907/
```

- 大小约 375 MB；
- 6 条严格 replay 全部有效；
- 回放证据：`results/expert_blue_pick_topup_replay_report.json`。

### 旧 20k 基线的 LeRobot 训练输入

```text
data/lerobot/openarm_cube_pick_expert_blue_100_train80/
```

- 大小约 149 MB；
- 80 episodes；
- 55,778 frames；
- 30 Hz；
- observation.state/action：float32 `[16]`；
- cameras：ceiling、wrist_right，视频分辨率 960×600。

数据集 `meta/info.json` 自带 `train 0:64 / val 64:80` 描述；20k 训练的
`train_config.json` 另外记录 `dataset.eval_split=0.1`。这两层 split 元数据必须分别
记录，不能直接用其中任意一个替代“独立 20 条原始保留集”的审计清单。

### 新增 120 条在线成功候选

```text
data/raw/openarm_cube_pick/expert_blue_pick_place_add120_20260908/
```

- 175 attempts、120 online success、成功率68.57%；
- 严格 replay `118/120`；
- 原始大小约7.4 GB；
- 证据：`results/data_validation/expert_blue_add120_replay_20260908.json`。

### 当前冻结 200 条数据

```text
data/cleaned/openarm_cube_pick_expert_blue_200_20260908/
data/cleaned/openarm_cube_pick_expert_blue_200_train160_20260908/
data/cleaned/openarm_cube_pick_expert_blue_200_holdout40_20260908/
```

- 最终200条由旧clean100和新增严格有效100条组成；
- split seed：20260908；
- 训练160条，holdout 40条；
- 构建与切分报告位于 `results/data_validation/`。

### 当前 LeRobot 训练输入

```text
data/lerobot/openarm_cube_pick_expert_blue_200_train160_20260908/
```

- 160 episodes、111,275 frames、30 Hz；
- cameras：ceiling、wrist_right；
- 完整验证：`valid=true`、`errors=[]`；
- 验证报告：`results/data_validation/lerobot_blue_200_train160_validation_20260908.json`。

## 当前模型

最新固定黑盒模型：

```text
outputs/act_openarm_cube_into_black_box_200_50k_20260910/
```

- 50,000 steps，checkpoints `005000..050000`；
- 日志：`results/act_training/act_black_box_200_50k_20260910.log`；
- 50k model SHA256：`bbf2516f2d91c34c1c801826be834cf2920b49254dfb79be4fbb62c703686cef`。

以下模型属于上一阶段附近随机放置任务。

```text
outputs/act_openarm_cube_pick_blue_100_20k/
```

- checkpoints：005000、010000、015000、020000；
- 最终策略目录：`checkpoints/020000/pretrained_model/`；
- 训练日志：`results/act_training/act_openarm_cube_pick_blue_100_20k.log`；
- 模型体积所在输出目录约 2.4 GB。

新模型目录：

```text
outputs/act_openarm_cube_pick_blue_200_50k_20260908/
```

- 训练完成：50,000 steps；
- checkpoints：005000至050000，每5k一份；
- 日志：`results/act_training/act_openarm_cube_pick_blue_200_50k_20260908.log`；
- 50k model SHA256：`661f4e33c036414878e8ce6e0ad23fe04dac7d4e17ad57a2a09dc186ebe2b0fd`。

## 当前闭环结果目录

固定黑盒50k的30个未见seed结果：

```text
results/closed_loop_black_box_200_50k_unseen30/
```

- seeds `20273000..20273029`；
- `task_success=27/30=90.0%`；
- `full_sequence_success=27/30=90.0%`；
- 失败为2条lift/hold和1条placement，没有独立回位失败。

以下结果属于上一阶段附近随机放置任务。

```text
results/closed_loop_blue_100/
```

旧20k在中央范围的抓取保持成功率为 `13/30=43.3%`；50k同口径为
`26/30=86.7%`。旧验证器未继续判定放置、松爪和回位，两者均不能称为完整任务成功率。

新完整范围结果位于：

```text
results/closed_loop_blue_200_50k_full_task/
```

阶段结果为 lift/hold 27、release 27、placement 26、return-home 0。当前口径将
`26/30=86.7%` 记为物体任务成功率，并将学习式回位单独记为复位指标。

## 旧任务资产

- `data/raw/openarm_cube_pick/expert_pilot_g3_frozen_20260903/`：旧橙色仅抓取数据。
- `data/lerobot/openarm_cube_pick_expert_pilot_v3_nosmoothing/`：旧橙色无平滑转换数据。
- `outputs/act_openarm_cube_pick_4ep_identity_12000/`：旧橙色 4 轨迹 12k 模型。
- `results/g5_5_2_*`：旧 action interface/cadence 研究证据。

这些资产保留用于回归，不应和蓝色完整任务的数据、prompt、checkpoint 或成功率混用。

## 目录管理规则

- 新状态报告放入 `MD/当前阶段/`，文件名包含建立和修改日期。
- 历史快照不覆盖，只在索引中降级为历史入口。
- `official/` 保持上游镜像语义；项目验证代码放在 `sim/validation/`。
- raw、LeRobot、checkpoint 和 results 使用不同目录，禁止原地覆盖。
- 训练/保留 episode 映射应有独立 manifest 和 hash。
