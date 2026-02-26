# DeepM3 Reproducibility Scripts

This directory contains the scripts used to generate the experimental results and figures reported in the paper.

| Script Name | Target Figure/Table | Description |
| :--- | :--- | :--- |
| **`exp_routing.py`** | **Fig. 4** (Ablation) | Compares routing accuracy between Fixed Threshold, MLP, and Neural ODE policies. |
| **`exp_efficiency.py`** | **Fig. 5** (System) | Benchmarks Latency (ms) and Token Cost ($) reduction compared to pure LLM approaches. |
| **`exp_alignment.py`** | **Fig. 6** (Safety) | Validates the JSON format compliance and safety alignment of the Agent. |
| **`tune_baselines.py`** | Baseline Fairness | Grid-searches SASRec/TiSASRec (lr/epochs) and exports best settings. |

## Usage
Run directly from the project root:
```bash
python scripts/experiments/exp_routing.py
```

## Paper Artifacts (Recommended)
For reproducible paper tables/figures (Table 1, Table 2, Fig. 3, Fig. 4), run:
```bash
bash scripts/experiments/run_paper_artifacts.sh
```

Generated outputs:
- `results/ml1m/overall_metrics.csv`
- `results/ml1m/table1_main.csv`
- `results/ml1m/table2_irregular.csv`
- `results/ml1m/fig3_seq_len.csv`
- `results/ml1m/fig4_sensitivity.csv`
- `results/ml1m/significance.csv`
- `assets/Fig3_SeqLen.pdf`
- `assets/Fig4_Sensitivity.pdf`

If you already have CSVs and only want to re-plot figures:
```bash
python scripts/experiments/plot_paper_fig.py \
  --fig3_csv results/ml1m/fig3_seq_len.csv \
  --fig4_csv results/ml1m/fig4_sensitivity.csv
```
### Smoke Test (`tests/test_imports.py`)

Create `tests/test_imports.py` to verify import paths and config presence.

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
        print("Core modules imported successfully.")
    except ImportError as e:
        pytest.fail(f"Import failed: {e}. Check PYTHONPATH or directory structure.")

def test_config_structure():
    """Verify config file exists."""
    assert os.path.exists("configs/config.yaml"), "configs/config.yaml not found!"
```
