> Historical workstation record; paths, results and hardware instructions are not a fresh-clone quickstart. See docs/REPRODUCIBILITY.md.

# Nano ACT 推理输出

> 历史 Nano/ROS 推理路径记录，使用旧 `act_real_fixed13_010000` 检查点。
> 当前工作站 Dora ACT 任务、测试 A/B/C 和物理操作顺序见
> [当前会话交接](ACT_DORA_NEXT_SESSION_HANDOFF.md)。本文的 `latest_action.json`
> 文件链路不是当前 Dora executor 验证结果。

在这条历史路径中，本工作区负责 ACT 模型和 Nano 上的推理部分；机械臂控制节点属于
`openarm_kinesthetic_ws`。推理程序不会连接 ROS，也不会直接命令机械臂。

## 推理程序

[`scripts/real_act_inference.py`](../scripts/real_act_inference.py) 读取观测桥写入的
`latest.json`，保持 ACT policy 的 action queue，生成连续动作 chunk，并将完整 JSON
通过原子 rename 写入 `latest_action.json`。默认每个 chunk 为 10 步，按 30 Hz 解释，
即约 0.333 秒的右臂和右夹爪目标。

动作文件包含完整 16 维 `actions`，以及供执行器直接使用的右侧 8 维 `right_actions`。
顺序固定为右臂 J1--J7、右夹爪、左臂 J1--J7、左夹爪。左侧输出暂不执行。
每条文件还有单调递增的 `action_sequence`、对应的观测序号、单调时钟发布时间与
`valid_until_monotonic_s`。执行端须拒绝过期文件、重复序号及不匹配的关节顺序。
推理端按 chunk 时长限制发布频率；5 Hz 观测桥下的 10 步 chunk 通常约每 0.4 秒更新。

## Nano 上运行

将脚本同步到 Nano 后，在已有的 ACT 运行环境中执行。该环境需要能读取
`/home/ubuntu/act_models/act_real_fixed13_010000`，并读写
`/dev/shm/act_live`。若沿用先前 shadow 使用的 `act-real:0.6.1` 环境，
容器内对应路径为 `/models/act_real_fixed13_010000` 和 `/live`，命令为：

```bash
python /real_act_inference.py /models/act_real_fixed13_010000 \
  --input /live --output /live/latest_action.json --chunk-actions 10
```

这里不创建新镜像，也不要求额外的长期容器。现有容器启动配置须给 `/live` 写权限；
先前 shadow 的只读挂载不能用于动作文件输出。

这条历史路径的程序只写动作文件。`act_executor` 尚未加入这条路径；不要把
`latest_action.json` 当作已发送给机械臂的命令。

## 工作站同步

```bash
rsync -av scripts/real_act_inference.py \
  ubuntu@192.168.3.112:/home/ubuntu/real_act_inference.py
```

Nano 上的观测桥仍需先运行并将 `/dev/shm/act_live/latest.json` 更新成功。
