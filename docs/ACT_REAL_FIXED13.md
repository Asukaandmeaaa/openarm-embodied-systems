> Historical workstation record; paths, results and hardware instructions are not a fresh-clone quickstart. See docs/REPRODUCIBILITY.md.

# 实物固定场景 13 条 ACT 试训（2026-09-30）

Nano 部署与实时只读测试的最新进度见
[ACT_REAL_NANO_PROGRESS_2026-09-30.md](ACT_REAL_NANO_PROGRESS_2026-09-30.md)。

本轮按用户决定纳入全部 13 条待审核 episode，不修改原始成功/审核标签。
全部用于训练，未设置独立验证集；训练 loss 和离线动作误差不能作为实物成功率。

## 数据

- 来源：`openarm_kinesthetic_data/collections/episodes_pending_review`。
- 输入：双路 RGB（ceiling、wrist_right），等比例缩小到 640×360；16 维实测关节。
- 标签：16 维 controller reference，右臂 7 关节+夹爪，再左臂 7 关节+夹爪，单位沿用源数据 rad。
- 共 15,400 帧，约 8.56 分钟有效动作。
- 30 Hz；仅保留相机、实测状态、动作参考均覆盖的公共时间区间。
- 每个时间点取不晚于该时刻的最新记录，不使用未来图片、不平滑关节或夹爪指令。
- 终止阶段动作记录有 0.3–0.5 秒间隔，前后参考完全相同；保留最后记录的目标。
- 图片小间隔采用上一帧，具体最大 age、P99、超过 100ms 的帧数见转换报告。
- 原始关节顺序、输入 JSON/JSONL 的 SHA256、原始审核状态均保存到转换报告。
- 输出：`data/lerobot/openarm_real_fixed13_20260930`，图像型 LeRobot 数据，不依赖视频解码。

## 环境和训练

独立环境 `.venv-real-act` 继承本机 conda vla 的依赖，单独安装 LeRobot 0.6.1、draccus 0.11.6。
完整版本：`results/act_real_fixed13/environment_full.txt`。

配置：`configs/training/act_real_fixed13_10000.yaml`。
ACT ResNet18 ImageNet 预训练视觉骨干；ACT 主体从头训练。
batch=8、10,000 steps、chunk_size=50（约1.67秒）、n_action_steps=10（约0.33秒）。
每 500 步保存；STATE=IDENTITY，ACTION/VISUAL=MEAN_STD。禁用上传和在线日志。
左臂保留为数据接口的一部分，实物部署需明确左臂固定策略。

首次启动（如果已有训练进程，不要重复执行）：

```bash
cd /home/lushilin/openarm_vla_ws
bash scripts/run_act_real_fixed13.sh
```

日志：`results/act_real_fixed13/train.log`、后台 `runner.log`。
训练退出码：`results/act_real_fixed13/train_exit_code.txt`，0 代表正常结束。
模型：`outputs/act_real_fixed13_10000_20260930/checkpoints/last/pretrained_model`。
训练结束自动在训练集取每条 20 帧，测首步预测与记录动作的 MAE：
`results/act_real_fixed13/offline_training_error.json`。
这项检查使用真实状态/图片逐帧输入，不是闭环评估。

回放原始示范：

```bash
/home/lushilin/miniconda3/envs/vla/bin/python scripts/replay_real_episode.py 20260929_115522_015
```

原始回放不是 ACT 预测回放。实物闭环还需要相同的图像预处理、关节顺序、
保存的归一化处理器以及控制频率。这里只训练/离线检查，不连接实物控制。
