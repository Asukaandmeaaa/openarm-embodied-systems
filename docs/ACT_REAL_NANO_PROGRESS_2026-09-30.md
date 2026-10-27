> Historical workstation record; paths, results and hardware instructions are not a fresh-clone quickstart. See docs/REPRODUCIBILITY.md.

# 实物 ACT：Nano 部署与只读联调进度（2026-09-30）

> 本文是旧 `act_real_fixed13_010000` 模型的历史记录。文中的“当前结论”和
> “下一步”仅描述 2026-09-30 的状态，不适用于现在的工作站 Dora 链路。
> 当前任务和下一步以 [ACT Dora current session handoff](ACT_DORA_NEXT_SESSION_HANDOFF.md)
> 为准。

## 当时的结论

13 条实物 episode 的 ACT 训练已结束，模型已复制到 Jetson Orin Nano Super，并在 Nano
完成合成输入测速和 100 帧真实相机、真实关节状态的**只读**推理。当前没有 ACT 动作执行节点，
没有向机械臂发送模型预测，也没有实物抓放成功率结果。下一项是 10 步动作 chunk 的只读审计。

## 模型与数据

| 项目 | 已核实内容 |
| --- | --- |
| 数据 | 13 条 episode、15,400 帧、30 Hz；双路 RGB `640×360`、16 维实测关节与 16 维 controller-reference 动作 |
| 顺序 | 右臂 J1–J7、右夹爪、左臂 J1–J7、左夹爪 |
| 策略 | ACT + ResNet18；`chunk_size=50`、`n_action_steps=10`；STATE=IDENTITY，ACTION/VISUAL=MEAN_STD |
| 训练 | 10,000 steps；`train_exit_code.txt=0`，`train.log` 于 13:05:45 记录 `End of training` |
| 离线检查 | 训练 episode 抽样 260 帧；右臂 MAE `0.01469 rad`、右夹爪 MAE `0.01436 rad`；退出码 0 |
| 模型 | 工作站 `outputs/act_real_fixed13_10000_20260930/checkpoints/010000/pretrained_model/`；Nano `/home/ubuntu/act_models/act_real_fixed13_010000/`，含权重、配置、预处理和后处理文件 |

离线 MAE 使用训练集图像和实测状态，不是留出集准确率或实物闭环成功率。转换报告中
13 条 episode 只有 1 条标记成功，其余 12 条仍为 `pending` 且未标记成功；原始审核标签
没有被训练转换过程修改。详见 `docs/ACT_REAL_FIXED13.md`、
`results/act_real_fixed13/conversion_report.json` 和
`results/act_real_fixed13/offline_training_error.json`。

## Nano 环境

- 硬件：Jetson Orin Nano Engineering Reference Developer Kit Super，7.4 GiB 内存。
- 系统：Jetson Linux R39.2、CUDA 13.2、Python 3.12.3、ROS 2 Jazzy。
- 推理镜像：`act-real:0.6.1`，基于 NVIDIA `nvcr.io/nvidia/pytorch:26.03-py3` ARM64 镜像；
  内含 PyTorch `2.11.0a0+a6c236b`、LeRobot `0.6.1`，补装 `av==15.1.0`。
- ROS 运行在 Nano 宿主机系统 Python；ACT 运行在 Docker NVIDIA runtime。
  `/dev/shm/act_live` 是两进程共享的只读观测交接目录。
- ROS 使用 `ROS_DOMAIN_ID=97` 和 `ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST`。

镜像从 DaoCloud 的 `m.daocloud.io/nvcr.io/nvidia/pytorch:26.03-py3` 拉取后标记为
原 NGC 名称，本地 Docker 构建成功。宿主机 `act_real` Conda 环境没有用于 ROS 桥接；
桥接必须以 `/usr/bin/python3` 启动，否则该 Conda 环境缺少 ROS 所需的 `yaml`。

## 已完成的推理验证

### 合成输入测速

使用 `scripts/benchmark_real_act.py`，每次重置策略以强制一次 ACT 推理；20 次计时：

| 指标 | Nano 结果 |
| --- | ---: |
| 平均推理时间 | 50.53 ms |
| P95 | 52.81 ms |
| 最大 | 55.28 ms |
| PyTorch 峰值 CUDA 分配 | 254.7 MiB |

该数据不包含相机解码、ROS 接收、控制器或机械臂运动，也不代表系统总内存占用。

### 真实观测，只读首步预测

现场 C1 全局相机、C2 右腕相机和 `/joint_states` 出帧后，
`scripts/real_act_observation_bridge.py` 以 5 Hz 保存最新双相机 JPEG 与有序 16 维状态；
`scripts/real_act_shadow.py` 加载 checkpoint，自带处理器做预处理与反归一化，只记录预测。
100 帧日志已从 Nano 拷到 `results/act_real_fixed13/first_home.jsonl`，序号 1616–1718。

| 指标 | 100 帧结果 |
| --- | ---: |
| 推理平均 / P95 / 最大 | 76.179 / 75.4 / 870.0 ms |
| 去掉首次推理后的 99 帧平均 / P95 / 最大 | 68.161 / 75.4 / 77.4 ms |
| 快照年龄平均 / P95 / 最大 | 12.559 / 20.8 / 120.3 ms |
| `/joint_states` 接收年龄平均 / 最大 | 8.556 / 10.0 ms |
| 全局相机接收年龄平均 / 最大 | 20.511 / 55.5 ms |
| 右腕相机接收年龄平均 / 最大 | 7.548 / 40.3 ms |
| 首步右臂及夹爪预测与实测最大差（各维中的最大值） | 0.0182 rad |
| 相邻快照首步预测跳变（各维中的最大值） | 0.01182 rad |

870 ms 尖峰只出现在第一帧；第二帧快照年龄为 120.3 ms，其余推理均不超过
77.4 ms。这与首次 CUDA 执行的启动开销一致，但尚未做重复冷启动实验。
100 帧中的实测关节状态保持固定；右臂 J1–J7 对固定 lifted HOME 的最大偏差为
`0.02039 rad`，小于 H2 的 `0.03 rad` 验收容差。该测试只覆盖 HOME 附近，
不能说明抓取路径或夹爪闭合阶段的预测质量。

汇总命令：

```bash
python3 scripts/summarize_real_act_shadow.py results/act_real_fixed13/first_home.jsonl
```

## 当时的软件边界

- `scripts/real_act_observation_bridge.py` 仅订阅 ROS 话题并写共享内存快照。
- `scripts/real_act_shadow.py` 仅读取快照、推理和记录 JSONL；没有 ROS 控制器客户端。
- 现有 `recorded_motion_player` 是示教轨迹回放器，不能直接作为 ACT 执行节点。
- 左臂虽在模型的 16 维接口中，实物测试应保持左臂固定。
- 现场 H1/H2/H5、C1/C2 的启动与 HOME 限制见
  `/home/lushilin/openarm_kinesthetic_ws/docs/实物数据采集现场指令.md`；
  本工作区的只读联调命令见 `docs/REAL_ACT_NANO_SHADOW_TEST.md`。

## 当时计划的下一步（已由当前交接文档取代）

1. **10 步 chunk 只读审计：**已在 `real_act_shadow.py` 加入 `--chunk-actions 10`，
   本地用训练画面验证可记录 10 步；Nano 上尚未运行。对实时 HOME 观测采集
   `home_chunk.jsonl`，汇总每步跳变、相对实测目标差和按 30 Hz 换算的目标变化率。
2. **视觉核对：**保存现场两路实时图像，确认镜头、方向、裁切、方块和黑盒位置与采集数据一致。
3. **实物动作接口：**在 chunk 审计通过后实现独立执行节点。首先只允许右臂与右夹爪，
   保留固定 HOME 起点、唯一命令源、关节与夹爪限位、最大单步/速度限制、
   观测和推理超时、控制器状态反馈、RT/RB 软件停止及实体急停；先做不发命令的
   执行计划检查，再做有人值守的短时低速试动。
4. **数据审核与结果：**人工复核仍为 `pending` 的 12 条 episode；实物完整抓放测试
   另行记录起点、成功/失败、停止原因和视频，不能以训练集 MAE 代替。

截至本文档建立时，`home_chunk.jsonl`、ACT 控制节点和 ACT 实物闭环结果均不存在。
