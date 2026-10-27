> Historical workstation record; paths, results and hardware instructions are not a fresh-clone quickstart. See docs/REPRODUCIBILITY.md.

# G5.5 测试导入问题修复

## 问题描述

运行测试时遇到错误：
```
AttributeError: 'NoneType' object has no attribute '__dict__'. Did you mean: '__dir__'?
```

## 根本原因

测试脚本使用 `importlib.util.spec_from_file_location` 动态加载模块，导致 dataclass 装饰器无法正确获取模块的 `__module__` 属性。

## 解决方案

**已修复**: `tests/test_g5_5_executor_equivalence.py` 前 20 行

### 修复前（错误）：
```python
# 使用 importlib 动态加载
import importlib.util
spec = importlib.util.spec_from_file_location("g5_5_executor_equivalence", g5_5_module_path)
g5_5_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g5_5_module)
```

### 修复后（正确）：
```python
# 添加到 sys.path 后直接导入
sys.path.insert(0, str(ROOT / "sim" / "validation"))
from g5_5_executor_equivalence import (
    compute_phase_lag,
    compute_deviation_metrics,
    convert_expert_to_30hz,
)
```

## 现在可以运行

```bash
cd /home/lushilin/openarm_vla_ws
python tests/test_g5_5_executor_equivalence.py
```

预期输出：
```
Running G5.5 Unit Tests...
================================================================================
✓ test_phase_lag_known_delay: lag=20.xx ms (expected ~20.0ms)
✓ test_phase_lag_insufficient_excitation: correctly returns None
✓ test_deviation_metrics: mean=0.xxxxx, max=0.xxxxx
✓ test_convert_expert_to_30hz: 1000 samples → 30 samples
✓ test_biquad_filter_reset: deterministic reset verified
✓ test_timestamp_alignment: monotonicity check passed
✓ test_clamp_ratio_denominator: ratio=5.00%
================================================================================
Tests: 7 passed, 0 failed
All tests passed!
```

## 验证修复

运行以下命令验证：
```bash
cd /home/lushilin/openarm_vla_ws
python tests/test_g5_5_executor_equivalence.py
```

如果测试通过，继续执行 G5.5：
```bash
bash scripts/run_g5_5_experiments.sh --smoke
bash scripts/run_g5_5_experiments.sh --full
```

## 技术说明

Python dataclass 在类定义时需要访问模块的 `__dict__`。使用 `importlib` 动态加载时，模块的 `__module__` 属性可能为 None，导致 dataclass 装饰器失败。

**解决方法**：将目录添加到 `sys.path`，然后使用标准的 `from ... import ...` 语法，让 Python 的标准导入机制正确设置模块属性。

## 相关文件

- ✅ 已修复: `tests/test_g5_5_executor_equivalence.py`
- 未修改: `sim/validation/g5_5_executor_equivalence.py`（代码本身没问题）

## 更新日期

2026-09-04
