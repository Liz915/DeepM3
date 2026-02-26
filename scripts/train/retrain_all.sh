#!/bin/bash
set -e
cd "$(dirname "$0")/../.."

echo "=== Phase 2: Retraining all 4 model variants ==="
echo "Start time: $(date)"

echo ""
echo "--- [1/4] Baseline GRU ---"
python scripts/train/train.py --solver baseline --epochs 15 --save_name model_gru_baseline.pth --seed 42

echo ""
echo "--- [2/4] DeepM3 (none = GRU encoder, no ODE step) ---"
python scripts/train/train.py --solver none --epochs 15 --save_name model_ode_none.pth --seed 42

echo ""
echo "--- [3/4] DeepM3 (Euler) ---"
python scripts/train/train.py --solver euler --epochs 15 --save_name model_ode_euler.pth --seed 42

echo ""
echo "--- [4/4] DeepM3 (RK4) ---"
python scripts/train/train.py --solver rk4 --epochs 15 --save_name model_ode_rk4.pth --seed 42

echo ""
echo "=== All training complete! ==="
echo "End time: $(date)"
echo ""

echo "--- Running evaluation pipeline ---"
python scripts/experiments/build_paper_artifacts.py --device auto
echo ""
echo "=== Paper artifacts generated in results/paper/ ==="
