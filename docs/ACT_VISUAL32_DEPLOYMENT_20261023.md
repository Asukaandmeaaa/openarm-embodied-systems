> Historical workstation record; paths, results and hardware instructions are not a fresh-clone quickstart. See docs/REPRODUCIBILITY.md.

# ACT visual32 20K 实物部署

> **2026-10-24 范围说明：本页仅为旧32条/张爪起点模型的历史部署流程。**
> 当前37条 `closed_transit_v2` 模型已完成50K训练，数据/结果见 [当前训练记录](ACT_TRAINING_RUNBOOK.md)。
> 新示教闭爪出发、右臂为position+velocity接口；旧ACT bridge仍要求position-only，新执行链尚未验收。
> **不要把下文checkpoint路径替换为新37条模型后直接start，也不要为新采集安装下文旧张爪H1候选库。**
> 先完成30K/50K候选的分阶段离线评估，再核对闭爪起点、q/qd生成、执行时序与零位停止条件。
> 训练完成、验证损失或旧硬件保持日志都不代表当前实物状态安全或已获得新模型抓取成功率。

本流程只控制右臂和右夹爪：

1. H1 启动并保持机械零位。
2. `zero-handoff` 验收双臂机械零位及右夹爪打开状态。
3. ACT 完成抓取、放置并自主返回机械零位。
4. 只有在 ACT 已经离开过零位，并再次稳定回到零位 0.5 秒后，系统才结束本 episode。

返回零位是 episode 完成条件，不是抓取成功判据。结果文件中的 `task_success` 仍为 `null`，需要操作员观察物体是否正确放置。

## 开始前

- 保持急停可触达，首次试验全程有人值守。
- 机械臂周围清空，物体、桌面和相机位置与 visual32 数据采集时一致。
- 右夹爪起始位置必须在 `[-0.46, -0.30] rad`。
- 双臂 J1-J7 必须在机械零位的 `±0.06 rad` 内并已静止。
- 不要对 visual32 执行旧的 `actctl h2`；旧 H2 会前往旧模型 HOME，和本模型的训练起点不一致。

每次电脑重启后，先以普通用户准备主机资源：

```bash
cd /home/lushilin/openarm_vla_ws
./scripts/actctl prepare-host
```

这一步创建 `/run/lock/openarm-can` 并加载 `can_raw` 内核模块（如需要会请求 sudo 密码），使 `/proc/net/can` 接收者登记表可被检查；不会打开 CAN socket、启用电机或启动 ROS。不要使用 `sudo ./scripts/actctl h1`。

CAN 接口还须按本机已有配置启用为 CAN-FD，仲裁速率 1Mbps、数据速率 5Mbps。`prepare-host` 不修改接口或清空发送队列。

## 安装 H1 夹爪保持修复（2026-10-23）

该修复将配置阶段的真实夹爪反馈写入 ROS 缓存，并在 H1 回零结束后保留右夹爪 `-0.35 rad` 的打开目标，防止周期控制重新发送旧的 `0 rad`。没有控制器的左夹爪保持其初始实测位置。没有改动夹爪增益或提高速度故障阈值。

候选硬件库已经在 `.h1_gripper_candidate_20261023` 编译并完成离线 C++ 缓存/控制接口测试；还没有进行实物复验。当前库不会自动被替换。

保持机械臂断电，在原终端分别用 Ctrl+C 停止 policy/Dora（如果启动过）、gateway 和 H1，然后执行：

```bash
cd /home/lushilin/openarm_vla_ws
./scripts/actctl install-h1-fix
```

安装器会检查 H1/gateway 已退出，校验候选库、源码和测试结果，备份原库到 `results/h1_gripper_before_*` 后原子替换。该命令不启动 ROS，也不发送电机命令。

看到 `installed` 后，按下面的 H1 流程重新上电启动。H1 应打印 `H1 gripper handoff on 'right_': ... hold target=-0.350000`，并在控制器启动后继续保持打开。后续执行 `zero-handoff` 验收，无需额外执行 `zero-open-gripper`。若没有保持打开，停在验收阶段检查日志。

## 准备 20K 模型

```bash
cd /home/lushilin/openarm_vla_ws
./scripts/actctl prepare \
  --profile visual32 \
  --motion-generator synchronized_v1 \
  --checkpoint /home/lushilin/openarm_vla_ws/outputs/act_visual32_20k_20261022/checkpoints/020000/pretrained_model
```

`visual32` 使用双相机原生 `640×480` 输入、50 步预测、每次执行前 10 步。按 2026-10-23 操作者要求，不再设固定 60 秒运行上限（`duration_s: null`）。手动 `actctl stop`、现有故障监控，以及离开后返回机械零位并稳定的 episode 结束条件仍然有效。必须有人值守；不保证策略一定完成抓取或自行回零。

更新前准备的试验仍记录 `duration_s: 60.0`，不能直接复用。先停止旧 policy/Dora，使用原检查点重新执行 `prepare`，再启动新 policy/Dora 并通过 `ready` 后显式 `start`。不需要为此重启仍在稳定保持的 H1/gateway；切换软件也不会自动回零。启动提示应为 `maximum_duration_s=None` 和 `no automatic duration stop`。

## 五个终端

终端 1：启动 ROS 硬件并执行 H1。

```bash
cd /home/lushilin/openarm_vla_ws
./scripts/actctl h1
```

终端 2：H1 控制器全部 active 后启动命令网关。

```bash
cd /home/lushilin/openarm_vla_ws
./scripts/actctl gateway
```

终端 3：做机械零位验收和保持交接。此步骤不执行旧 H2 HOME。

安装上述修复后，H1 会打开右夹爪并保持。这里直接验收双臂零位及夹爪打开状态；不要为绕过验收再运行独立开夹爪程序。若此前仅因夹爪未打开导致纯验收失败，再次验收会归档原来的失败日志。

```bash
cd /home/lushilin/openarm_vla_ws
./scripts/actctl zero-handoff
```

必须看到：

```text
ZERO_AND_HANDOFF_VERIFIED
```

终端 4：启动 20K 策略服务。

```bash
cd /home/lushilin/openarm_vla_ws
./scripts/actctl policy
```

终端 5：启动 visual32 的 640×480 Dora 图。

```bash
cd /home/lushilin/openarm_vla_ws
./scripts/actctl dora
```

回到终端 3，先等待所有实时检查通过，再显式开始：

```bash
./scripts/actctl ready
./scripts/actctl start
```

运行中可只读查看状态：

```bash
./scripts/actctl status
```

## 停止与下一轮

正常情况下，ACT 离开起点后再稳定返回机械零位，结果原因为：

```text
returned_to_mechanical_zero
```

系统随后停止动作生产并让 ROS 保持最后位置，不会自动开始下一轮。需要提前停止时运行：

```bash
./scripts/actctl stop
```

出现危险趋势时优先使用物理急停，不要等待软件停止条件。

下一轮之前先停止 policy 和 Dora 两个终端，确认机械臂仍在零位，然后：

```bash
./scripts/actctl next-trial --profile visual32
./scripts/actctl policy
./scripts/actctl dora
./scripts/actctl ready
./scripts/actctl start
```

每轮证据都保存在 `results/act_manual_*` 下。检查 `act/result.json`：返回零位完成时 `reason` 为 `returned_to_mechanical_zero`，但抓取/放置是否成功仍需人工确认。

## 当前保护边界

- 保留 ROS/MoveIt 关节限位、速度/加速度限制、反馈新鲜度、动作时延、控制周期和网关所有权检查。
- visual32 的 MuJoCo 碰撞准入关闭，因为现有仿真场景对应旧任务，不代表当前桌面和物体环境。
- 没有力/接触保护；夹爪跟踪误差只记录诊断。
- 自动回零判断使用实测位置和速度：双臂 J1-J7 在 `±0.06 rad`、右夹爪在 `[-0.46,-0.30] rad`、双臂及右夹爪最大速度不高于 `0.05 rad/s`，连续稳定 0.5 秒。
- 为防止启动即结束，系统必须先检测到右臂离开零位至少 `0.12 rad`，或夹爪闭合到 `-0.20 rad` 以上，之后才允许回零完成。
