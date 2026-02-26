#!/bin/bash
# Master experiment runner for KBS submission
# Trains and evaluates ALL models on a single dataset
set -e
cd "$(dirname "$0")/../.."

DATASET="${1:-ml1m}"
DIM="${2:-256}"
EPOCHS="${3:-15}"
SEED=42
DATA_DIR=""

if [ "$DATASET" = "amazon" ]; then
    DATA_DIR="data/amazon"
fi

echo "============================================"
echo " KBS Full Experiment Suite"
echo " Dataset: $DATASET | dim=$DIM | epochs=$EPOCHS"
echo "============================================"

# ---- 1. Train all models ----

echo ""
echo "=== [1/6] Training GRU Baseline ==="
python scripts/train/train_unified.py \
    --model gru --epochs $EPOCHS --seed $SEED \
    --hidden_dim $DIM --dataset $DATASET \
    --save_name "${DATASET}_gru_d${DIM}.pth" \
    ${DATA_DIR:+--data_dir $DATA_DIR}

echo ""
echo "=== [2/6] Training SASRec ==="
python scripts/train/train_unified.py \
    --model sasrec --epochs $EPOCHS --seed $SEED \
    --hidden_dim $DIM --dataset $DATASET \
    --save_name "${DATASET}_sasrec_d${DIM}.pth" \
    ${DATA_DIR:+--data_dir $DATA_DIR}

echo ""
echo "=== [3/6] Training TiSASRec ==="
python scripts/train/train_unified.py \
    --model tisasrec --epochs $EPOCHS --seed $SEED \
    --hidden_dim $DIM --dataset $DATASET \
    --save_name "${DATASET}_tisasrec_d${DIM}.pth" \
    ${DATA_DIR:+--data_dir $DATA_DIR}

echo ""
echo "=== [4/6] Training DeepM3 (none - GRU encoder, no ODE) ==="
python scripts/train/train_unified.py \
    --model deepm3 --solver none --epochs $EPOCHS --seed $SEED \
    --hidden_dim $DIM --dataset $DATASET \
    --save_name "${DATASET}_deepm3_none_d${DIM}.pth" \
    ${DATA_DIR:+--data_dir $DATA_DIR}

echo ""
echo "=== [5/6] Training DeepM3 (Euler) ==="
python scripts/train/train_unified.py \
    --model deepm3 --solver euler --epochs $EPOCHS --seed $SEED \
    --hidden_dim $DIM --dataset $DATASET \
    --save_name "${DATASET}_deepm3_euler_d${DIM}.pth" \
    ${DATA_DIR:+--data_dir $DATA_DIR}

echo ""
echo "=== [6/6] Training DeepM3 (RK4) ==="
python scripts/train/train_unified.py \
    --model deepm3 --solver rk4 --epochs $EPOCHS --seed $SEED \
    --hidden_dim $DIM --dataset $DATASET \
    --save_name "${DATASET}_deepm3_rk4_d${DIM}.pth" \
    ${DATA_DIR:+--data_dir $DATA_DIR}

echo ""
echo "============================================"
echo " All training complete!"
echo "============================================"

# ---- 2. Evaluate all models ----
echo ""
echo "=== Running full evaluation pipeline ==="
python scripts/experiments/eval_all_models.py \
    --dataset $DATASET --hidden_dim $DIM --seed $SEED \
    ${DATA_DIR:+--data_dir $DATA_DIR}

echo ""
echo "============================================"
echo " Full experiment suite complete!"
echo " Results saved to results/${DATASET}/"
echo "============================================"
