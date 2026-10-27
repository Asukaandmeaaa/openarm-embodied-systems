> Historical workstation record; paths, results and hardware instructions are not a fresh-clone quickstart. See docs/REPRODUCIBILITY.md.

> **2026-10-24 当前交接，优先于下文历史记录：**
> 请先阅读 [37条实物数据/50K训练结果与后续离线评估](ACT_TRAINING_RUNBOOK.md)。
> 数据为nearby30、spread7、rotation0；训练30条/40772帧，验证7条/9532帧。
> `act_closed_v2_37eps_20261023b_50k` 已于10月23日20:54完成，last->050000；
> 日志验证最低候选为030000(0.1011)，最后050000为0.1015，均未完成本批实物抓取验收。
> 新数据是闭爪出发和position+velocity控制；旧张爪/position-only ACT部署步骤不兼容。
> spread cycle_007在动作下发前因控制器状态查询超时停止，已有7条合格偏移数据未受影响；根因尚未确认。
> 下一步是离线比较30K/50K及验证执行接口；本次只更新文档，当前硬件状态未知，不自动启动/清故障。

> **2026-10-15 历史交接入口：**
> 请先阅读 [ACT 实物测试新对话交接](ACT_PHYSICAL_TEST_NEXT_CHAT_2026-10-15.txt)
> 和 [H1 → H2 → ACT 手动流程](ACT_TRACKING_V2_FRESH_START_2026-10-15.txt)。
> 当前 ACT 使用 Dora 推理、gateway 独占转发、ROS 重力补偿硬件执行；
> 不要使用下文历史的 Dora 直连 CAN / 插值 HOME 实物建议。
> 最近新增的原生 B 键专家回放是独立流程，不是 ACT，不能与 ACT 控制器混跑。
> 所有实物操作由用户手动执行；旧姿态与急停确认不视为当前确认。

# ACT + Dora 当前会话交接

更新时间：2026-10-14

当前新增工作流：[ROS H1/H2 实测录制与 Dora 仿真回放操作单](ROS_H1_H2_CAPTURE_REPLAY_RUNBOOK.txt)。
已在操作员现场确认后执行实体 ROS H1/H2；H2 两阶段均 ACCEPTED，最终 HOME_READY，
操作员确认 HOME 与之前 ROS H2 一致。ROS overlay 为 `install-h1h2-capture`。
首次 H1 录制因 ROS 元数据 infinity 序列化失败，保留不完整原始记录；修复后重新开启被动
录制，获得完整 H2 的 16 关节实测、命令、时间戳、参数及完整 MoveIt 计划，没有重复 H1。
完整 H2 数据：`results/h2_capture_20261014_172126/`；实测轨迹：
`results/h2_capture_20261014_172126_measured.json`。
[实体执行报告](../results/h2_capture_20261014_172126_review/REPORT.txt) 的初次失败解读已由
[限位与作用范围审计](../results/h2_validation_scope_audit_20261014/REPORT.txt) 更新：
ROS URDF 已允许 J4 到 -0.02 rad，原仿真限位 0 是过期配置；左夹爪仍低于其 ROS 下限，
但在明确的右臂仿真模式中作为实测碰撞环境，不输出左臂命令，不单独阻断右臂仿真。
右臂仿真使用录制的 ROS URDF 并保留 MoveIt J6/J7 更严格范围；49418 个姿态检查通过。
实测 H2 已完成 Dora 纯仿真回放：16.21822258 s、1623 个 right8 输出，物理授权仍 false。
命令源现可完整归档并保留起点偏差诊断，但仍未通过命令仿真准入；未执行物理回放。
36 集真实训练数据的左动作在每集内完全恒定，模型 state/action 仍为 16D；ROS/Dora
ACT 任务动作只发右8。注意当前物理 Dora graph 仍启用并保持左臂，不是左侧无命令。
操作员确认双臂支撑后已关闭 ROS，两侧硬件正常 deactivate/shutdown。
下一步为物理控制语义与新鲜起点/场景检查；Dora 的实时重力前馈与跟踪停止控制仍需实现。
不能修改原始轨迹或放宽门限来绕过失败，也不能把 synthetic 软件测试当实体轨迹。

> 2026-10-14 轨迹审计更正（优先于下文历史交接）：当前 Dora HOME **不等价于已验证的 ROS H2**。
> HOME/side-prehome 终点数值一致，但 Dora 用逐段关节 smoothstep，未迁移 MoveIt 桌面碰撞规划。
> 用户 16:20 日志在 phase 1 无进展超时，未到 lifted HOME，也未打开 J8。
> 离线重建在 ROS 当前桌面几何下检出 Stage A 穿入桌面模型；当前 smoke HOME 和包含同一
> startup-home 的 ACT graph 不应继续实体运行。下文允许执行“corrected HOME”的旧建议撤回。
> ROS 启用重力前馈；当前 Dora position driver 的 MIT 命令前馈力矩为零，不能称为已继承重力补偿。
> 审计：[结果与修复要求](../results/dora_ros_home_audit_20261014/REPORT.txt)、
> [量化数据](../results/dora_ros_home_audit_20261014/audit.json)、
> [轨迹图](../results/dora_ros_home_audit_20261014/paths.png)。未运行硬件、未改变运行时控制参数。
>
> 下一步优先恢复 ROS H2 的规划/碰撞与执行语义，并核对重力前馈、J8 和终点容差。
> 36 个本地 episode 均明确排除初始化轨迹；不能拿其第一帧或旧 plan-only 路径当 H2 回放。
> Test A/B/C 仍是后续 ACT 工作，不解决此次 HOME 路径穿桌问题。

这份文档是新会话的入口。任务是让 OpenArm 的 Dora ACT 物理控制链路在不重复 ROS 已完成硬件验证的前提下可靠工作，并定位此前 ROS/Dora 控制差异导致的运动异常。

## 我们在做什么

目标链路是：双相机和 16 维关节观测 → ACT 推理 → Dora action chunk executor → 右臂 8 维受保护命令边界。左臂在物理 ACT 测试中保持非运动状态。

需要区分四个问题来源：

1. ACT 在固定输入上是否确定地产生相同输出。
2. Dora MuJoCo 中，ACT chunk 和 executor 是否产生周期性反向运动。
3. 已知专家轨迹经过同一个 Dora executor 时是否平滑。
4. 软件回归通过且 CAN 稳定后，操作员可先做右臂 hold 和 corrected HOME；
   只有前三项和 ACT 安全门通过后，才做短时物理 ACT。

## 已经完成的工作

ROS 控制链路已经验证过以下硬件事实，Dora 直接复用这些结果，不再重复安排硬件测试：

- 官方校准夹具完成的机械零位。
- URDF 与实体坐标系的一致性。
- J7 命令到反馈的关节映射。
- 重力补偿。
- J7 关节刚度。

Dora 软件侧已经完成：

- HOME 从实时测量姿态开始，并使用平滑、限速的三段轨迹：零位、side-prehome、lifted HOME，最后打开 J8。
- 保留 ROS H1 的测量姿态到零位 200 步、10 ms 的回零语义。
- HOME 阶段转换使用新的实时反馈作为起点，并重新计算阶段时长。
- 命令速度根据实际 CAN 反馈年龄限制，保留原有 `tracking_tolerance=0.120`，没有放宽跟踪保护。
- 跟踪、超时、无进展、CAN 健康检查和动作时序故障继续锁存。
- 右臂 Dora 节点在一个 CAN 会话内完成 HOME、等待观测和 ACT，不再使用 HOME 进程退出后再启动 verify-only ACT 的交接。
- 右臂继续使用 ROS 坐标配置和增益：`official/dora-openarm-evaluation/openarm_real_ros_coordinates.yaml`。
- 静态代码审查发现，已安装的 LeRobot `predict_action_chunk` 会进入 eval 模式；
  正常推理的 ACT VAE latent 使用零值，随机 latent 属于训练路径。当前 checkpoint
  为 STATE=IDENTITY、ACTION/VISUAL=MEAN_STD、`chunk_size=50`、
  `n_action_steps=10`。这些静态事实不能代替 Test A 的 20 次输出实测。

当前验证结果：评估测试 70 个通过；Python 编译通过；MuJoCo shadow graph 和 host graph 构建通过。没有在这次修复后执行实体机器人运动。shadow graph 曾因本地 policy socket 不存在而无法完成端到端运行，这不是硬件结论。

## 当前卡点

以下工作仍未完成，不能把物理 ACT 称为已验证：

- Test A 的“同一帧图像和同一关节状态重复 20 次”专用工具还需要补齐。现有 `benchmark_real_act.py` 使用合成输入，只能测时延；现有 `real_act_shadow.py` 读取连续快照，不是固定输入复测。
- Test B 的 shadow graph 没有把原始 policy action chunks 和 executor 输出保存成可分析文件。
- Test C 的现有专家回放直接操作 MuJoCo，绕过 Dora executor，不能作为 executor 验证。
- 物理 ACT host graph 还没有强制的 first-action admission、1–2 秒执行预算和完整相机/策略 readiness gate。
- policy server 一次只接受一个 Unix socket 客户端；IPC smoke test 结束后必须重启 server。

## 下一步计划

### 1. 先完成 Test A

实现一个固定输入复测工具，启动时只读取一次 manifest、两张 JPEG 和 16 维 state，然后用生产路径 `predict_action_chunk → postprocessor` 重复 20 次。工具必须显式记录 `eval()`、`torch.inference_mode()`、checkpoint hash、输入 hash、每次完整 chunk、逐元素最大差异和是否逐元素相等。

通过标准：20 次完整 chunk 完全一致，或明确量化差异并解释来源。若不一致，先查 eval 模式、随机采样、预处理/反归一化、policy reset 和 VAE latent；不要先改机器人保护参数。

### 2. 再完成 Test B

在 `dataflow-real-act-two-camera-shadow.yaml` 中加入 raw action chunk recorder，同时保存 executor 输出和 MuJoCo 右臂观测。使用固定 seed、`--scene demo --keyframe home --ctrl --render --cameras camera_wrist_right,camera_ceiling`，检查周期性反向是否存在于：原始 ACT chunk、Hermite/low-pass 后命令、MuJoCo 实际关节轨迹。

### 3. 完成 Test C

选用已审核成功的专家 episode：

`openarm_kinesthetic_data/collections/episodes_pending_review/20261008_090909_002`

把其中已对齐的 16 维 controller-reference action 按 30 Hz 送入与 ACT 相同的 `dora-openarm-actions-executor`，在 MuJoCo 中保存 raw expert commands、executor commands 和 observed qpos。现有 `sim/validation/expert_action_replay.py` 不能作为这个测试，因为它绕过 Dora executor。

### 4. 运行软件回归并准备物理分阶段测试

```bash
cd /home/lushilin/openarm_vla_ws/official/dora-openarm-evaluation
.venv/bin/python -m pytest -q tests nodes/dora-openarm-observer/tests
.venv/bin/dora build dataflow-real-act-two-camera-shadow.yaml --uv --local
.venv/bin/dora build dataflow-real-act-two-camera-host.yaml --uv --local
```

软件回归通过、CAN 接口状态与反馈稳定后，操作员可先做右臂 CAN hold 和
corrected HOME，不必等待 Test A/B/C。短时物理 ACT 必须等 Test A/B/C、
first-action admission、执行预算和 readiness gate 全部通过。当前两个独立
分阶段诊断命令是：

```bash
cd /home/lushilin/openarm_vla_ws/official/dora-openarm-evaluation
.venv/bin/dora run dataflow-right-can-hold-diagnostic-host.yaml --uv
```

确认 hold 日志稳定后，操作员支撑机械臂再退出 hold，然后执行：

```bash
.venv/bin/dora run dataflow-right-home-smoke-host.yaml --uv
```

这两个流程都是独立诊断，不得把 HOME 进程退出后的姿态交给旧的 verify-only ACT 流程。当前完整 ACT graph 仍不能作为短 ACT 命令使用，直到 first-action admission 和执行预算落地并通过测试。

## 绝对不要再踩的坑

- 不要重新做机械零位、URDF/实体坐标系、J7 映射、重力补偿或 J7 刚度硬件测试，除非出现直接反驳 ROS 验证结果的新证据。
- 不要同时运行 ROS controller、旧 ACT executor 和 Dora CAN 节点。
- 不要使用“HOME 进程退出 → 电机禁用 → verify-only ACT”旧交接。
- 不要把固定输入实验替换成合成输入 benchmark，也不要把连续快照 shadow 当成固定输入复测。
- 不要把直接 MuJoCo expert replay 当成 Dora executor 验证。
- 不要把 policy `Connected`、camera preview、计时器存活或启动日志当成物理控制成功。
- 不要通过增大 `tracking_tolerance`、删除 jam test、提高 PD gain 或改 motor zero 来掩盖 Dora 调度问题。
- policy server 做过 IPC smoke test 后要重新启动，不能把已占用或已关闭的 socket 交给 shadow graph。
- 物理流程退出前必须支撑机械臂；右臂 guarded STOP 会锁存保持，进程退出或 emergency stop 才会禁用电机。
- 不要在没有日志记录 command、feedback、timestamp、phase、CAN health 和 first fault 的情况下延长实体运行时间。

## 给新会话的下一步提示词

```text
你正在继续 OpenArm ACT + Dora 物理控制软件任务。目标是让 Dora ACT 右臂链路在不重复 ROS 已验证硬件测试的前提下可靠工作。ROS 已验证机械零位、URDF/实体坐标系、J7 命令反馈映射、重力补偿和 J7 刚度；不要重新安排这些硬件测试。

已完成：guarded_home.py 已实现从测量姿态开始的平滑 HOME、ROS H1 回零语义、实时反馈年龄限速、跟踪/无进展锁存和单 CAN 会话；70 个软件测试通过，shadow/host graph 构建通过。修复后还没有执行实体运动。

当前卡点：Test A 固定同一 observation 重复 20 次的专用工具、Test B MuJoCo raw action chunk/executor 记录、Test C 专家轨迹经过同一 Dora executor 的 MuJoCo 回放，以及物理 ACT 的 first-action admission 和 1–2 秒执行预算尚未完成。

下一步按顺序完成：
1. 实现并运行固定输入 ACT chunk 20 次复测，保存完整输出和逐元素差异。
2. 给 dataflow-real-act-two-camera-shadow.yaml 加 raw chunk、executor command、MuJoCo qpos 记录，判断反向运动出现在 ACT、executor 还是仿真层。
3. 用 openarm_kinesthetic_data/collections/episodes_pending_review/20261008_090909_002 的专家 action，通过同一个 dora-openarm-actions-executor 在 MuJoCo 回放；不要使用绕过 executor 的 expert_action_replay.py 作为结论。
4. 运行 70+ 软件回归和两个 graph build；确认 policy socket 是新启动的单客户端 server。
5. 软件回归和 CAN 稳定后，操作员可先执行 right CAN hold，再执行 corrected HOME；支撑机械臂后退出。只有 software Test A/B/C、日志和 ACT 安全门通过后，才设计受保护的短 ACT。

每一步报告：改动文件、实际命令、退出码、输出文件、结论和未解决风险。绝对不要同时运行 ROS 和 Dora CAN 控制器，不要恢复 HOME 进程退出后的 verify-only 交接，不要放宽 tracking guard 或重复 ROS 硬件校准。
```
