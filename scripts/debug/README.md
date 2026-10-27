# Debug Scripts

这些是调试用的临时脚本，用于诊断早期 ACT 和 expert 问题。

## 文件说明

### read_expert_episode0.py
**用途**: 读取 expert dataset episode 0, frame 0 的数据
- 加载 LeRobot parquet 格式数据
- 显示 observation.state 和 action
- 用于验证数据集格式和初始状态

**运行**:
```bash
python scripts/debug/read_expert_episode0.py
```

---

### debug_act_output.py
**用途**: 调试 ACT policy 输出
- 加载 ACT checkpoint
- 检查 policy 输出的 action 值
- 对比 observation.state mean
- 用于诊断 ACT rollout 失败

**运行**:
```bash
python scripts/debug/debug_act_output.py
```

---

### test_act_5x_velocity.py
**用途**: 测试放宽 5x velocity limits
- 临时实验：放宽 ActionGuard 速度限制
- 验证 ACT policy 输出方向是否正确
- 快速测试（5秒）

**运行**:
```bash
python scripts/debug/test_act_5x_velocity.py <checkpoint> --seed 20260902
```

---

### test_stabilize_debug.py
**用途**: 调试 stabilize 阶段
- 测试机器人稳定化逻辑
- 周期性重新提交 target
- 诊断 ActionGuard 行为

**运行**:
```bash
python scripts/debug/test_stabilize_debug.py --seed 20260902
```

---

## 状态

这些脚本是早期调试工具，现在已被正式验证脚本取代：

**正式验证脚本**:
- `sim/validation/expert_action_replay.py` - Expert strict replay
- `sim/validation/act_closed_loop.py` - ACT closed-loop rollout
- `sim/validation/g5_5_executor_equivalence.py` - G5.5 executor equivalence gate

**建议**: 使用正式验证脚本，这些 debug 脚本仅用于参考或快速测试。
