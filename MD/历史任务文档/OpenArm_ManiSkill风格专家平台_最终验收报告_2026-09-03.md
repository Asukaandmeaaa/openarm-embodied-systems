> Historical workstation record; paths, results and hardware instructions are not a fresh-clone quickstart. See docs/REPRODUCIBILITY.md.

# OpenArm ManiSkill 风格专家数据平台最终验收报告

- 日期：2026-09-03
- 状态：**G0–G5 门禁已完成**
- 唯一权威状态文档：本文件
- 任务：`Pick up the orange cube with the right arm.`

## 1. 结论

OpenArm MuJoCo 自动专家、事务式原始数据、LeRobot v3.0 转换、严格 actuator replay 和 ACT 小样本语义门禁均已形成可重复、可审计的闭环。

ACT smoke 的工程门禁通过，但固定 seed 的 ACT 任务结果为失败：模型能经官方 executor 等价链路驱动机械臂产生明显运动，却没有形成双指接触。这一失败被如实保留，不解释为抓取成功。

## 2. 阶段验收

| 阶段 | 状态 | 验收证据 |
|---|---:|---|
| G0 契约与基线 | 通过 | 专家平台测试 24/24；覆盖 TCP/FK、quaternion、16D、guard、碰撞、路径、reset、事务、loader、replay 和 keyboard 回归 |
| G1 单 seed | 通过 | 当前 scene/params/代码下 seed `20260902` 连续 5/5 成功 |
| G2 随机化鲁棒性 | 通过 | 当前代码重跑 seeds `20260902..20260921`，20/20 成功，成功率 100% |
| G3 正式 pilot | 通过 | 5 条成功、两相机、OpenArm Dataset 官方 validator/loader 通过 |
| G4 LeRobot 与回放 | 通过 | 重新执行 v3.0 转换；LeRobot loader 全量遍历 5006 帧；严格随机 replay 达到 6.79 cm / 2.214 s |
| G5 ACT smoke | 通过 | 1000-step train/eval loss、checkpoint reload、官方 executor 等价链固定 seed rollout 均已完成；任务结果明确为 fail |

## 3. G4 严格 actuator replay

修复后的回放遵循：

```text
episode metadata initial_qpos + exact cube pose
→ recorded safe position target
→ data.ctrl
→ 1000 Hz mj_step
→ post-physics observation comparison
→ contact/lift/hold evaluator
```

不再把已经过安全过滤的 recorded action 再送入第二次 ActionGuard。

Episode 0 验收结果：

- seed：`20260902`
- physics：1000 Hz
- frames：1010
- 最大抬升：`0.0679045 m`
- 连续有效保持：`2.214 s`
- observation 跟踪误差：mean `0.00574 rad`，p95 `0.00902 rad`，max `0.02393 rad`
- timestamp 最大误差：`1 ns`
- success：true

证据：`results/expert_replay/g4_strict_episode_0.json`。

额外全量压力回放为 4/5 直接通过；episode 4 在原终止时刻得到 `0.998 s`，与成功边界相差 2 ms。G4 要求的随机一条严格 replay 已通过，这一敏感边界保留为已知限制。

## 4. LeRobot v3.0 转换与 loader

实际执行：

```bash
.venv/bin/openarm-dataset-validate --no-update-metadata \
  data/raw/openarm_cube_pick/expert_pilot_g3_frozen_20260903

.venv/bin/openarm-dataset-convert \
  data/raw/openarm_cube_pick/expert_pilot_g3_frozen_20260903 \
  data/lerobot/openarm_cube_pick_expert_pilot_v3_g4_frozen \
  --format lerobot_v3.0 --state qpos --fps 30 --success-only

/home/lushilin/miniconda3/envs/vla/bin/python \
  sim/validation/validate_lerobot.py \
  data/lerobot/openarm_cube_pick_expert_pilot_v3_g4_frozen \
  --output results/openarm_expert/g4_lerobot_full_validation.json
```

全量 loader 结果：

- 5 episodes；5006 frames 全部解码；
- `observation.state` 与 `action` 均为 float32 `[16]`；
- 两路图像均为 `[3,600,960]`；
- task、episode/frame/timestamp 单调性、NaN/Inf、夹爪符号均通过；
- ceiling 最小图像标准差 `0.35394`；wrist_right `0.29441`，无黑帧。

## 5. ACT smoke

### 5.1 训练与 loss

- LeRobot：0.6.0
- train/eval：4 episodes / 1 held-out episode
- steps：1000
- checkpoints：每 100 steps 保存
- train loss：step 10 `46.366` → step 1000 `2.130`，下降 `95.4%`
- eval loss：step 100 `0.5043` → step 1000 `0.4324`，下降 `14.3%`
- checkpoint 1000 reload：通过，ACTPolicy 参数量 `51,617,680`

证据：

- `results/act_training/act_openarm_cube_pick_g5_500.log`
- `results/act_training/g5_1000_loss_summary.json`
- `outputs/act_openarm_cube_pick_smoke_g5_500/checkpoints/001000/pretrained_model/`

### 5.2 修复后的闭环验证器

执行链直接复用官方类：

```text
ACT 30 Hz chunk
→ dora_openarm_actions_executor.HermiteUpsampler
→ dora_openarm_actions_executor.BiquadLowpass @ 250 Hz
→ ActionGuard
→ data.ctrl
→ MuJoCo @ 1000 Hz
```

修复内容：

- 成功判定从 `chunks == 1` 分支移出；
- 每个 physics step 更新 height/contact/hold；
- rollout deadline 从 stabilize 完成时刻计算；
- stabilize 与 policy guard 统计分开；
- stabilize 与专家采集同为 1.5 s 连续静态目标；
- 初始化状态相对专家数据最大误差 `0.000583 rad`；
- 保存 executor 源码路径与 SHA256。

1000-step checkpoint 固定 seed 结果：

- rollout：完整运行 `45.001 s`、45,001 physics steps；
- 最大关节状态位移：`2.095 rad`；
- cube 最大抬升：`0.00970 m`；
- 双指接触 physics steps：0；
- task success：false；
- failure reason：`lift_hold_condition_not_met`；
- policy 阶段每步仍触发 velocity/acceleration 限制，需要在后续策略/控制器迭代中处理。

证据：`results/act_rollout/g5_1000_seed_20260902.json`。

## 6. 当前冻结的专家结果

### G1

- 固定 seed `20260902`；
- 当前 params hash：`b50d4fbf...58c024`；
- 5/5 success；
- 每次最大抬升 `0.065789 m`；
- 每次连续 hold `1.0 s`。

证据：`results/openarm_expert/g1_current_frozen_fixed_seed.json`。

### G2

- 固定窗口：`20260902..20260921`；
- attempts：20；successes：20；
- success rate：100%；
- failures：空；
- 所有 episode 均达到 ≥5 cm、≥1 s。

证据：

- `results/openarm_expert/g2_current_frozen/expert_summary.json`
- `results/openarm_expert/g2_current_frozen/attempts.jsonl`

## 7. 测试

执行命令：

```bash
PYTHONPATH="$PWD:$PWD/official/openarm_mujoco/src:$PWD/official/openarm_dataset/src" \
  .venv/bin/python -m pytest -q \
  tests/test_expert_contracts.py tests/test_expert_extended.py \
  --junitxml=results/tests/expert_platform_junit.xml
```

结果：`24 passed in 77.77s`。

新增覆盖包括：

- multi-start IK 与单次 timeout fallback；
- candidate failure 隔离；
- home→pregrasp collision；
- Cartesian step ≤5 mm；
- 不完整路径不可执行；
- joint limits 与 joint6/joint7 margin；
- collector/guard/FSM/recorder reset；
- KeyboardInterrupt attempt 独立持久化；
- OpenArm 官方 validator/loader；
- LeRobot v3 全量 5006 帧 loader；
- 严格 actuator replay；
- keyboard dataflow 18 项嵌套回归；
- ACT loss 与固定 seed 闭环证据。

## 8. 冻结 hash

完整 manifest：`results/openarm_expert/g4_g5_frozen_manifest.json`。

关键 tree SHA256：

| 产物 | SHA256 |
|---|---|
| OpenArm raw dataset | `6fa537d4710c8d1774195b315e3b79fcb9132134606f28c1f9af8a82df431874` |
| LeRobot v3 dataset | `2aabee66706c6adb67c8339f25ec281c008a6b37b63f6fb94b8806fb59853e28` |
| 当前源代码 bundle | `835063cc828db156105b454b183b7f1a7b3060f63883be6057fe2d96e2189a41` |
| ACT 1000 pretrained model | `cc4ca684bb0bf70222968953b321682e2b5eb7a326904e40f1d025226ad26d2f` |
| 当前 G2 结果 | `682a35990104699ee3587f2d4063bfdec30c0eec21f24c82a48dee58f9d2b4ef` |

manifest 同时保存 5 个嵌套 Git 仓库的 HEAD、dirty 状态和未提交文件清单；这些原有修改继续视为用户资产。

## 9. 本轮修改

- `sim/validation/expert_action_replay.py`：严格 metadata/timestamp/direct-ctrl replay；
- `sim/validation/act_closed_loop.py`：逐 physics-step 评估、正确 deadline、分相统计、官方 executor 复用；
- `openarm_cube_pick_expert/expert.py`：bounded IK timeout、fallback、精确 collision allow-list；
- `openarm_cube_pick_expert/collect.py`：实际 FSM、全状态 reset、逐 attempt fsync 持久化；
- `sim/validation/validate_lerobot.py`：LeRobot 全量遍历 validator；
- `sim/validation/validate_g1_fixed_seed.py`：当前冻结版本 G1；
- `sim/validation/summarize_act_training.py`：loss 审计摘要；
- `sim/validation/freeze_manifest.py`：数据/模型/代码/repo 冻结；
- `scripts/train_act_smoke_with_log.py`：持久化训练日志；
- `tests/test_expert_extended.py`：扩展门禁测试。

## 10. 已知限制与下一步

1. ACT 1000-step smoke 没有完成抓取；它只证明数据可训练、loss 下降、checkpoint 可加载、执行语义和失败判定正确。
2. ACT raw target 与 conservative ActionGuard 之间仍持续发生限速/限加速度；下一阶段应先决定部署时采用 executor 平滑还是 guard 插值的单一责任边界，避免双重动态约束。
3. 全量 replay 的 episode 4 对 1 s 边界敏感 2 ms；如果未来把“5/5 全部严格 replay”设为门槛，应在 recorder success 时强制写 terminal frame，并记录 initial qvel/ctrl/actuator state。
4. 当前自动专家主采集仍以单环境为主；多进程 batch/MimicGen 属于后续扩展。
5. 在进入 π0 前，应由项目负责人确认接受 ACT smoke 的“工程门禁通过、任务结果失败”结论；不得把该 rollout 描述为抓取成功。

