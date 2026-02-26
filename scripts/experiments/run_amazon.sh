#!/bin/bash
# ============================================================
# Amazon Books Dataset Full Experiment Pipeline
# ============================================================
# This script automatically downloads the Amazon dataset (if needed),
# processes it, runs a small baseline grid search, trains all 4 
# model variants, and then evaluates all Phase 2 metrics.
#
# Usage:
#   bash scripts/experiments/run_amazon.sh
#   DEVICE=cuda bash scripts/experiments/run_amazon.sh
#   SKIP_DATA=1 SKIP_TUNING=1 SKIP_TRAINING=1 bash scripts/experiments/run_amazon.sh
# ============================================================
set -euo pipefail
cd "$(dirname "$0")/../.."

# ---- Configuration ----
DATASET="amazon"
DEVICE="${DEVICE:-auto}"
HIDDEN_DIM="${HIDDEN_DIM:-128}"
EPOCHS="${EPOCHS:-15}"
SEED="${SEED:-42}"
CONFIG="${CONFIG:-configs/config.yaml}"
NUM_NEG="${NUM_NEG:-100}"
TOPK="${TOPK:-10}"
OUTPUT_DIR="${OUTPUT_DIR:-results/${DATASET}}"

# Subsample Amazon users by default since it is massive (50k is reasonable for local testing)
AMAZON_MAX_USERS="${AMAZON_MAX_USERS:-50000}"

# Grid options (defaults to just 1 run per baseline to get results fast, as Amazon is huge)
TUNE_LRS="${TUNE_LRS:-3e-4}"           
TUNE_EPOCHS="${TUNE_EPOCHS:-30}"
TUNE_MODELS="${TUNE_MODELS:-sasrec,tisasrec}"

# Skip flags
SKIP_DATA="${SKIP_DATA:-0}"
SKIP_TUNING="${SKIP_TUNING:-0}"
SKIP_TRAINING="${SKIP_TRAINING:-0}"
SKIP_PHASE2="${SKIP_PHASE2:-0}"

STAMP="$(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
echo " Amazon Dataset Full Pipeline"
echo " Dataset: ${DATASET} | dim=${HIDDEN_DIM} | device=${DEVICE}"
echo " Started: ${STAMP}"
echo "============================================================"

# ---- 0. Data Preparation ----
if [[ "${SKIP_DATA}" == "0" ]]; then
    if [[ ! -f "data/amazon/ml1m_train.pkl" ]]; then
        echo ""
        echo "=== [0/3] Processing Amazon Data ==="
        python src/data/process_amazon.py --max_users ${AMAZON_MAX_USERS}
    else
        echo "=== [0/3] Data already exists. Skipping processing. ==="
    fi
else
    echo "=== [0/3] Skipped (SKIP_DATA=1) ==="
fi

# ---- 1. Baseline Grid Search ----
if [[ "${SKIP_TUNING}" == "0" ]]; then
    echo ""
    echo "=== [1/3] Baseline Grid Search ==="
    DATASET="${DATASET}" HIDDEN_DIM="${HIDDEN_DIM}" \
    LRS="${TUNE_LRS}" EPOCHS_LIST="${TUNE_EPOCHS}" \
    MODELS="${TUNE_MODELS}" DEVICE="${DEVICE}" \
    SEED="${SEED}" CONFIG="${CONFIG}" \
    NUM_NEG="${NUM_NEG}" TOPK="${TOPK}" \
    OUTPUT_DIR="${OUTPUT_DIR}/baseline_tuning" \
    bash scripts/experiments/tune_baselines.sh
else
    echo ""
    echo "=== [1/3] Skipped (SKIP_TUNING=1) ==="
fi

# ---- 2. Train Target Models ----
if [[ "${SKIP_TRAINING}" == "0" ]]; then
    echo ""
    echo "=== [2/3] Training Core DeepM3 Models ==="
    
    echo " --- Training Baseline (GRU) --- "
    python scripts/train/train_unified.py \
        --model gru --epochs $EPOCHS --seed $SEED \
        --hidden_dim $HIDDEN_DIM --dataset $DATASET \
        --save_name "${DATASET}_gru_d${HIDDEN_DIM}.pth" \
        --device $DEVICE
        
    echo " --- Training DeepM3 (None) --- "
    python scripts/train/train_unified.py \
        --model deepm3 --solver none --epochs $EPOCHS --seed $SEED \
        --hidden_dim $HIDDEN_DIM --dataset $DATASET \
        --save_name "${DATASET}_deepm3_none_d${HIDDEN_DIM}.pth" \
        --device $DEVICE
        
    echo " --- Training DeepM3 (Euler) --- "
    python scripts/train/train_unified.py \
        --model deepm3 --solver euler --epochs $EPOCHS --seed $SEED \
        --hidden_dim $HIDDEN_DIM --dataset $DATASET \
        --save_name "${DATASET}_deepm3_euler_d${HIDDEN_DIM}.pth" \
        --device $DEVICE
        
    echo " --- Training DeepM3 (RK4) --- "
    python scripts/train/train_unified.py \
        --model deepm3 --solver rk4 --epochs $EPOCHS --seed $SEED \
        --hidden_dim $HIDDEN_DIM --dataset $DATASET \
        --save_name "${DATASET}_deepm3_rk4_d${HIDDEN_DIM}.pth" \
        --device $DEVICE
else
    echo ""
    echo "=== [2/3] Skipped (SKIP_TRAINING=1) ==="
fi

# ---- 3. Phase 2 Evaluation ----
# Because we already integrated --dataset and --data_dir propagation into run_phase2.sh
if [[ "${SKIP_PHASE2}" == "0" ]]; then
    echo ""
    echo "=== [3/3] Phase 2: Comprehensive Evaluation ==="
    
    # We skip tuning in run_phase2.sh since we already did it in Stage 1 above
    # We also skip multiseed to save time by default.
    DATASET="${DATASET}" DEVICE="${DEVICE}" \
    HIDDEN_DIM="${HIDDEN_DIM}" SEED="${SEED}" \
    NUM_NEG="${NUM_NEG}" TOPK="${TOPK}" \
    OUTPUT_DIR="${OUTPUT_DIR}" \
    SKIP_TUNING=1 SKIP_MULTISEED=1 SKIP_CLEANUP=0 \
    bash scripts/experiments/run_phase2.sh
else
    echo ""
    echo "=== [3/3] Skipped (SKIP_PHASE2=1) ==="
fi

echo ""
echo "============================================================"
echo " Amazon Experiment Pipeline Complete!"
echo " Everything should be stored in: ${OUTPUT_DIR}/"
echo "============================================================"
