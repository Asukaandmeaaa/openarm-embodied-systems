# OpenArm v2 右臂真机示教与 ACT 数据采集

当前状态更新于 2026-09-28。固定 H1/H2 真机流程已经验证；当前默认工作流是右臂
`continuous_marked` 连续示教、dense-path 安全回放和双相机 ACT episode 采集。

日期＋序号文件命名、统一 collection 目录、B 键触发一次双相机采集，以及固定 HOME
终点修正已完成首条 Nano 真机 pilot：`20260928_151345_001` 已由 `finish_success` 保存到
`episodes_pending_review`，**仍待落盘图像的人工审核和 ACT 格式导出**。操作说明见
[ACT_BUTTON_COLLECTION_WORKFLOW.md](docs/ACT_BUTTON_COLLECTION_WORKFLOW.md)。
日常示教、ACT 采集与检查的终端命令统一见
[实物数据采集现场指令.md](docs/实物数据采集现场指令.md)。

## 当前工作流

```text
H1：硬件使能并 return_to_zero
  → H2：zero → side-prehome → fixed lifted HOME → J8 OPEN
  → LT 笛卡尔微调 / LT+LB J1–J7 passive 手拖 / L3 电动夹爪
  → X 开始 continuous_marked 连续录制
  → Y/A 只标记回放时 close/open；录制时夹爪不动作；无需独立 LB
  → 第二次 X 自动 PLAN ONLY、返回 HOME、实测验收并保存完整动作 JSON
  → 轨迹 MoveIt PLAN ONLY → 有人值守原速 dry replay
  → 双相机 + measured observation + controller-reference action episode
```

固定方块/黑盒场景的首条正式两-marker 轨迹已在 Nano 保存并通过原速 dense-path PLAN ONLY；
操作者反馈无相机完整实物回放正常。首条新编号双相机 raw episode 已保存待审核，
当前还没有可直接用于 ACT 训练的正式数据集。轨迹与验收数值见
[CURRENT_HARDWARE_CHECKPOINT.md](docs/CURRENT_HARDWARE_CHECKPOINT.md)。

固定 lifted HOME（右 J1–J7）：

```text
[-0.1450262670, -0.0151076284, -0.0060887956,
  1.3751977241, -0.0031428430, -0.1984151663, -0.1222898194]
```

H2 的 J1–J7 使用固定桌面碰撞体和两段 fresh MoveIt 规划；到达 HOME 后再单独命令
J8 张开到 `-0.35 rad`，只有夹爪实测位置验收通过才输出 `HOME_READY`。人工示教微调不再在
10 Hz 控制回路内同步查询桌面碰撞，仍保留关节限位、方向保真、速度限制和 watchdog。
回放仍执行 PlanningScene、碰撞和状态有效性检查。

## 当前控制约定

- `LT`：笛卡尔微调。左摇杆为 world X/Y，D-pad 上下为 world Z；D-pad 左右和右摇杆为
  工具局部 roll/pitch/yaw。
- `LT+LB`：J1–J7 hardware passive。机械臂可手拖；`LT+LB+L3` 实际电动切换夹爪。
- 退出手拖后必须完整松开一次 `LT`，再重新按 `LT` 才能微调。
- continuous_marked 录制期间 A/Y 只记录 close/open，不驱动真夹爪；独立 LB 默认无录制语义。
- 标准示教顺序：HOME 夹爪已张开 → X 开始 → 抓取位按 Y 记 CLOSE → 放置位按 A 记 OPEN → 松开手拖键并恢复刚性 → 第二次 X 自动返回 HOME 后保存。
- 回放沿全部连续样本并保留录制时间戳；`--playback-rate 1.0` 为示教原速，只在 A/Y
  marker 对应样本处暂停执行实际夹爪动作。
- `RT/RB`：软件停止锁存，不替代实体急停。

## 入口与文档

- H2：`home_guarded_executor`
- 手柄：`real_gamepad`
- 真机控制：`right_kinesthetic_bridge`
- 动作录制：`right_motion_recorder`
- 动作回放/返回 HOME：`recorded_motion_player`
- 数据采集：`wrist_camera`、`real_episode_recorder`

现场唯一命令来源是 [HOME_TEACHING_WORKFLOW.md](docs/HOME_TEACHING_WORKFLOW.md)。当前实现和
待验证项见 [CURRENT_HARDWARE_CHECKPOINT.md](docs/CURRENT_HARDWARE_CHECKPOINT.md)，按键表见
[GAMEPAD_CONTROL.md](docs/GAMEPAD_CONTROL.md)，数据格式见
[DATA_CONTRACT.md](docs/DATA_CONTRACT.md)，下一阶段双相机与方块位置泛化采集见
[ACT_REAL_DATA_COLLECTION_PLAN.md](docs/ACT_REAL_DATA_COLLECTION_PLAN.md)。

## 源码边界

2026-10-13 的 Dora CAN 反馈修复共用 VLA 工作区中的源码与运行环境：

```bash
bash scripts/install_dora_can_fix.sh
bash scripts/validate_dora_can_feedback.sh can0
bash scripts/validate_dora_can_feedback.sh can1
```

安装脚本可接收目标 Dora 环境的 Python 路径；复验可用 `DORA_PYTHON` 指定该环境。
VLA 不在相邻目录时设置 `OPENARM_VLA_WS`。只读复验只请求反馈，不使能或移动电机。
修复包含每轴查询间隔 1 ms、主动查询最多 20 Hz，以及未修复版本的启动拦截。
本工作区的 `replay/official_filters.py` 只复用 Dora 插值算法，不直接访问 CAN；
ROS 示教控制器和 `references/` 下的历史源码快照未应用这份 Python Dora 补丁。
详见 [ACT 执行记录](docs/ACT_NANO_REAL_EXECUTION_2026-10-07.md#dora-can-反馈修复2026-10-13)。

- `/home/lushilin/openarm_kinesthetic_ws`：本项目源码、配置、流程与 ROS adapter。
- `/home/lushilin/openarm_ws`：上游硬件/MoveIt 工作区；当前项目另外修改了
  `openarm_hardware/src/openarm_simple_hardware.cpp`，使 J8 参与 hardware passive。
- `/home/lushilin/openarm_vla_ws`：VLA/仿真参考，不作为现场命令来源。

Nano 部署必须同时同步 recorder 工作区和上述 hardware driver，然后分别编译
`openarm_hardware`、`openarm_kinesthetic_recorder`。

## 安全边界

- H2 只能从 H1 建立的机械零位开始；任意中途姿态不得重跑 H2。
- 示教第二次 X 自动规划并返回 HOME 后保存；dry replay 后使用独立 runtime return HOME。
- 自动回放前停止 bridge，保留 gamepad 作为 RT/RB 软件停止输入。
- 首次执行任何新动作必须清空运动包络、全程有人值守并手持实体急停。

## 最近软件验证

- `openarm_hardware` 编译通过，夹爪接触检测 18 项测试通过。
- `openarm_kinesthetic_recorder` 编译通过，本轮包内测试 100 项通过。
- 顶层测试在项目既有 VLA/MuJoCo 环境中 196 项通过。
