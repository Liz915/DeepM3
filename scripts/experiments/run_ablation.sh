#!/bin/bash
set -e

# 控制变量：固定随机种子，确保只比较模型架构差异
SEED=42

echo "🧪 Starting Ablation Study (Seed=$SEED)..."
echo "Model,Solver,HR@10,NDCG@10,Latency(ms)" > results/ablation_report.csv

# --- 实验 1: Baseline GRU (纯离散模型) ---
echo "--------------------------------"
echo "🔄 [1/3] Running Baseline: GRU (No ODE)..."
# solver=none 意味着只运行 GRU，跳过 ODE 演化
python scripts/train.py \
    --seed $SEED \
    --solver none \
    --save_name "model_baseline_gru.pth" \
    --config configs/config.yaml

# 评估
RESULT=$(python scripts/evaluate.py --model_path "checkpoints/model_baseline_gru.pth")
VALUES=$(echo $RESULT | awk -F',' '{print $2 "," $3 "," $4}')
echo "Baseline(GRU),None,$VALUES" >> results/ablation_report.csv
echo "✅ GRU Done: $VALUES"


# --- 实验 2: Neural ODE (Euler Solver) ---
echo "--------------------------------"
echo "🔄 [2/3] Running ODE with Euler Solver (Fast but less accurate)..."
python scripts/train.py \
    --seed $SEED \
    --solver euler \
    --save_name "model_ode_euler.pth" \
    --config configs/config.yaml

# 评估
RESULT=$(python scripts/evaluate.py --model_path "checkpoints/model_ode_euler.pth")
VALUES=$(echo $RESULT | awk -F',' '{print $2 "," $3 "," $4}')
echo "DeepM3,Euler,$VALUES" >> results/ablation_report.csv
echo "✅ Euler Done: $VALUES"


# --- 实验 3: Neural ODE (RK4 Solver) ---
echo "--------------------------------"
echo "🔄 [3/3] Running ODE with RK4 Solver (Ours)..."
# 注意：如果你刚才已经跑过 seed=42 的 RK4，其实可以直接用。
# 但为了严谨，这里再跑一次，确保环境一致。
python scripts/train.py \
    --seed $SEED \
    --solver rk4 \
    --save_name "model_ode_rk4.pth" \
    --config configs/config.yaml

# 评估
RESULT=$(python scripts/evaluate.py --model_path "checkpoints/model_ode_rk4.pth")
VALUES=$(echo $RESULT | awk -F',' '{print $2 "," $3 "," $4}')
echo "DeepM3,RK4,$VALUES" >> results/ablation_report.csv
echo "✅ RK4 Done: $VALUES"

echo "--------------------------------"
echo "🏆 Ablation Study Finished!"
echo "👇 Final Comparison:"
cat results/ablation_report.csv