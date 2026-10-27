#!/bin/bash
# ACT Smoke Test 验证脚本
# 用于完成 G5: ACT smoke 测试的最后验证

set -e

echo "========================================"
echo "G5: ACT Smoke Test - 执行验证"
echo "========================================"
echo ""

# 环境检查
echo "[1/5] 检查环境..."
conda run -n vla python -c "
import torch
print(f'✓ PyTorch: {torch.__version__}')
print(f'✓ CUDA available: {torch.cuda.is_available()}')
import lerobot
print(f'✓ LeRobot: {lerobot.__version__}')
"

# Checkpoint 加载测试
echo ""
echo "[2/5] 测试 Checkpoint 加载..."
conda run -n vla python -c "
from pathlib import Path
from lerobot.policies import get_policy_class

checkpoint = Path('outputs/act_openarm_cube_pick_smoke/checkpoints/000200')
policy = get_policy_class('act').from_pretrained(str(checkpoint))
print(f'✓ Checkpoint 加载成功')
print(f'✓ 总参数: {sum(p.numel() for p in policy.parameters()):,}')
"

# Action Replay 测试（如果脚本存在）
echo ""
echo "[3/5] 测试 Action Replay..."
if [ -f "sim/validation/replay_episode.py" ]; then
    conda run -n vla python sim/validation/replay_episode.py \
        data/raw/openarm_cube_pick/expert_pilot_g3_frozen_20260903/episodes/0 \
        2>&1 | head -20
    echo "✓ Replay 测试完成（查看上面的输出）"
else
    echo "⚠ replay_episode.py 不存在，跳过"
fi

# ACT 闭环 Rollout（单次测试）
echo ""
echo "[4/5] 测试 ACT 闭环 Rollout..."
echo "运行固定 seed 20260902 的 rollout..."

mkdir -p results/act_rollout

conda run -n vla python sim/validation/act_closed_loop.py \
    outputs/act_openarm_cube_pick_smoke/checkpoints/000200 \
    --seed 20260902 \
    --max-seconds 45 \
    --output results/act_rollout/rollout_seed_20260902.json

echo "✓ Rollout 完成！"
echo ""
echo "结果保存在: results/act_rollout/rollout_seed_20260902.json"

# 显示结果摘要
echo ""
echo "[5/5] 显示结果摘要..."
if [ -f "results/act_rollout/rollout_seed_20260902.json" ]; then
    python -c "
import json
with open('results/act_rollout/rollout_seed_20260902.json') as f:
    data = json.load(f)
print('─' * 50)
print('ACT Rollout 结果:')
print('─' * 50)
for key, value in data.items():
    print(f'{key}: {value}')
print('─' * 50)
"
fi

echo ""
echo "========================================"
echo "G5 验证完成！"
echo "========================================"
echo ""
echo "总结:"
echo "  ✓ Checkpoint 可以加载"
echo "  ✓ 训练完成 200 steps"
echo "  ✓ 闭环 rollout 已运行"
echo ""
echo "查看详细报告: results/g5_act_smoke_test_report.md"
echo ""
