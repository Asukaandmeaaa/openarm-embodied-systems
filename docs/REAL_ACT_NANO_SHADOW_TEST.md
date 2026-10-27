> Historical workstation record; paths, results and hardware instructions are not a fresh-clone quickstart. See docs/REPRODUCIBILITY.md.

# Nano ACT 实时只读测试（历史记录）

> **状态：历史 Nano/ROS 只读方案，不是当前物理 ACT 运行手册。**
> 最后复核：2026-10-14。本文使用旧的 `act_real_fixed13_010000` 模型、Nano
> 路径和 ROS 观测桥；它不验证当前工作站的
> `act_real_combined_unreviewed_30000_20261008` 检查点，也不包含当前 Dora
> 当前 Dora HOME/J7 软件修复已经完成；本文仍只记录旧 Nano/ROS 只读实验，
> 不要据此启动物理 HOME、hold 或 ACT。
>
> 当前任务、已完成的软件修复、Test A/B/C 和物理测试顺序见
> [ACT Dora current session handoff](ACT_DORA_NEXT_SESSION_HANDOFF.md)。
> 文末命令只能复现历史 ROS 只读数据，不能代替当前 Dora 实验。

此阶段只读取 ROS 观测并记录 ACT 预测，**不会向控制器发送动作**。模型目录为
`/home/ubuntu/act_models/act_real_fixed13_010000`，容器镜像为 `act-real:0.6.1`。

## 1. 从工作站复制两个脚本

```bash
rsync -av /home/lushilin/openarm_vla_ws/scripts/real_act_observation_bridge.py \
  /home/lushilin/openarm_vla_ws/scripts/real_act_shadow.py \
  ubuntu@192.168.3.112:/home/ubuntu/
```

## 2. 在 Nano 启动现场 ROS 节点

按 `/home/ubuntu/openarm_kinesthetic_ws/docs/实物数据采集现场指令.md` 启动 H1、H2、
H5、C1、C2。H2 必须从 H1 建立的零位开始；已有同一次会话的 `HOME_READY` 时不重跑。
把方块、黑盒和相机恢复到训练时的固定位置。不要启动 H3、H4、轨迹播放器或 B 键回放。

各新终端使用现场文档的“共用终端环境”。检查三条观测流：

```bash
ros2 topic echo /joint_states --once
ros2 topic hz /camera/global_camera/color/image_raw/compressed
ros2 topic hz /camera/wrist_right/image_raw/compressed
```

`topic hz` 按 Ctrl-C 退出检查，不要关闭相机进程。

## 3. 宿主机终端：保存最新观测到共享内存

```bash
source /opt/ros/jazzy/setup.bash
source /home/ubuntu/openarm_ws/install/setup.bash
source /home/ubuntu/openarm_kinesthetic_ws/ros2_ws/install/setup.bash
unset ROS_LOCALHOST_ONLY
export ROS_DOMAIN_ID=97
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
python3 /home/ubuntu/real_act_observation_bridge.py --rate 5 --max-age 0.25
```

另一个终端可检查 `ls -lh /dev/shm/act_live/latest.json`。没有该文件时，检查
`/joint_states` 是否包含双臂和双夹爪全部 16 个关节，以及两路压缩图像是否在更新。

## 4. Nano 容器终端：运行 100 次实时预测

```bash
mkdir -p /home/ubuntu/act_shadow_runs /home/ubuntu/.cache/torch
docker run --rm --runtime nvidia --ipc=host \
  -e TORCH_HOME=/cache \
  -v /home/ubuntu/.cache/torch:/cache:rw \
  -v /home/ubuntu/act_models:/models:ro \
  -v /dev/shm/act_live:/live:ro \
  -v /home/ubuntu/act_shadow_runs:/runs:rw \
  -v /home/ubuntu/real_act_shadow.py:/real_act_shadow.py:ro \
  act-real:0.6.1 \
  python /real_act_shadow.py /models/act_real_fixed13_010000 \
    --input /live --output /runs/first_home.jsonl --samples 100
```

完成后会显示 `ACT_SHADOW_COMPLETE`；每条 JSONL 含观测年龄、推理耗时、
右臂和夹爪的实测位置、预测位置及差值。结果在
`/home/ubuntu/act_shadow_runs/first_home.jsonl`。脚本每次使用最新观测重新预测第一步；
该测试不衡量 10 步 chunk 的执行节拍或实物任务成功率。

### 10 步 chunk 只读审计

更新 `real_act_shadow.py` 到 Nano 后，在桥接仍运行时，把上面命令最后两行换成：

```bash
  python /real_act_shadow.py /models/act_real_fixed13_010000 \
    --input /live --output /runs/home_chunk.jsonl --samples 30 --chunk-actions 10
```

这仍然只记录预测，不发送动作。拷回 `home_chunk.jsonl` 后用
`scripts/summarize_real_act_shadow.py` 汇总 10 步 chunk 内的最大跳变和按 30 Hz
计算的目标变化率。首次推理含 CUDA 冷启动开销，不用于闭环性能结论。

## 5. 下一阶段判断

先查看实时测试中的源观测年龄、完整端到端耗时、预测跳变和右臂/夹爪范围。
只有这些检查通过，才设计独立的动作执行节点；它需要 30 Hz 定时执行、限位、
速度和单步变化限制、丢帧及推理超时停止、RT/RB 停止和实体急停。
