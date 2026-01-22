# 🧪 DeepM3 Reproducibility Scripts

This directory contains the scripts used to generate the experimental results and figures reported in the paper.

| Script Name | Target Figure/Table | Description |
| :--- | :--- | :--- |
| **`exp_routing.py`** | **Fig. 4** (Ablation) | Compares routing accuracy between Fixed Threshold, MLP, and Neural ODE policies. |
| **`exp_efficiency.py`** | **Fig. 5** (System) | Benchmarks Latency (ms) and Token Cost ($) reduction compared to pure LLM approaches. |
| **`exp_alignment.py`** | **Fig. 6** (Safety) | Validates the JSON format compliance and safety alignment of the Agent. |

## Usage
Run directly from the project root:
```bash
python scripts/experiments/exp_routing.py
```
### 🛠️ 执行动作 3：防炸 Smoke Test (`tests/test_imports.py`)

**操作**：新建文件 `tests/test_imports.py`。
**内容**：确保环境路径没问题。

```python
import sys
import os
import pytest

# Ensure project root is in path
sys.path.append(os.getcwd())

def test_core_imports():
    """Smoke Test: Verify critical modules can be imported without error."""
    try:
        import src.api
        import src.agent.core
        import src.dynamics.modeling
        print("✅ Core modules imported successfully.")
    except ImportError as e:
        pytest.fail(f"❌ Import failed: {e}. Check PYTHONPATH or directory structure.")

def test_config_structure():
    """Verify config file exists."""
    assert os.path.exists("configs/config.yaml"), "❌ configs/config.yaml not found!"
```