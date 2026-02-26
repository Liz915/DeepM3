#!/bin/bash
# ============================================================
# Phase 2: Post-Baseline Experiment Suite for KBS Submission
# ============================================================
# Run this AFTER:
#   1. retrain_all.sh (trains baseline + 3 DeepM3 variants)
#   2. run_paper_artifacts.sh (main eval tables)
#   3. tune_baselines.sh (baseline grid search — at least small grid)
#
# This script orchestrates ALL remaining experiments:
#   Stage 1: Full baseline grid search (if not already done)
#   Stage 2: Time sensitivity
#   Stage 3: Time robustness
#   Stage 4: Efficiency benchmark
#   Stage 5: Per-user significance analysis
#   Stage 6: Irregularity analysis
#   Stage 7: Multi-seed stability (trains 5×2 models — slowest)
#   Stage 8: Cleanup & archive
#
# Usage:
#   bash scripts/experiments/run_phase2.sh
#   DEVICE=mps SKIP_MULTISEED=1 bash scripts/experiments/run_phase2.sh
# ============================================================
set -euo pipefail
cd "$(dirname "$0")/../.."

# ---- Configuration (all overridable via env vars) ----
DEVICE="${DEVICE:-auto}"
DATASET="${DATASET:-ml1m}"
HIDDEN_DIM="${HIDDEN_DIM:-128}"      # Match your retrain_all.sh dimension
SEED="${SEED:-42}"
CONFIG="${CONFIG:-configs/config.yaml}"
NUM_NEG="${NUM_NEG:-100}"
TOPK="${TOPK:-10}"
OUTPUT_DIR="${OUTPUT_DIR:-results/${DATASET}}"
DATA_DIR="${DATA_DIR:-}"
if [[ -z "${DATA_DIR}" && "${DATASET}" == "amazon" ]]; then
    DATA_DIR="data/amazon"
fi

# Skip flags for resuming partial runs
SKIP_TUNING="${SKIP_TUNING:-0}"
SKIP_SENSITIVITY="${SKIP_SENSITIVITY:-0}"
SKIP_ROBUSTNESS="${SKIP_ROBUSTNESS:-0}"
SKIP_EFFICIENCY="${SKIP_EFFICIENCY:-0}"
SKIP_SIGNIFICANCE="${SKIP_SIGNIFICANCE:-0}"
SKIP_IRREGULARITY="${SKIP_IRREGULARITY:-0}"
SKIP_MULTISEED="${SKIP_MULTISEED:-0}"
SKIP_CLEANUP="${SKIP_CLEANUP:-0}"

# Baseline tuning grid (for full grid search)
TUNE_LRS="${TUNE_LRS:-1e-4,3e-4,5e-4}"
TUNE_EPOCHS="${TUNE_EPOCHS:-50,100,150}"
TUNE_MODELS="${TUNE_MODELS:-sasrec,tisasrec}"

mkdir -p "${OUTPUT_DIR}"

STAMP="$(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
echo " Phase 2: KBS Experiment Suite"
echo " Dataset: ${DATASET} | dim=${HIDDEN_DIM} | device=${DEVICE}"
echo " Started: ${STAMP}"
echo "============================================================"

# ---- Stage 1: Full Baseline Grid Search ----
if [[ "${SKIP_TUNING}" == "0" ]]; then
    echo ""
    echo "=== [Stage 1/8] Baseline Grid Search ==="
    DATASET="${DATASET}" HIDDEN_DIM="${HIDDEN_DIM}" \
    LRS="${TUNE_LRS}" EPOCHS_LIST="${TUNE_EPOCHS}" \
    MODELS="${TUNE_MODELS}" DEVICE="${DEVICE}" \
    SEED="${SEED}" CONFIG="${CONFIG}" \
    NUM_NEG="${NUM_NEG}" TOPK="${TOPK}" \
    OUTPUT_DIR="${OUTPUT_DIR}/baseline_tuning" \
    DATA_DIR="${DATA_DIR}" \
    bash scripts/experiments/tune_baselines.sh
    echo " Stage 1 done."
else
    echo ""
    echo "=== [Stage 1/8] Skipped (SKIP_TUNING=1) ==="
fi

# ---- Stage 2: Time Sensitivity ----
if [[ "${SKIP_SENSITIVITY}" == "0" ]]; then
    echo ""
    echo "=== [Stage 2/8] Time Sensitivity ==="
    python scripts/experiments/eval_time_sensitivity.py \
        --config "${CONFIG}" \
        --seed "${SEED}" \
        --num_neg "${NUM_NEG}" \
        --topk "${TOPK}" \
        --device "${DEVICE}" \
        --output_dir "${OUTPUT_DIR}" \
        --dataset "${DATASET}" \
        --hidden_dim "${HIDDEN_DIM}" \
        --data_dir "${DATA_DIR}"
    echo " Stage 2 done."
else
    echo ""
    echo "=== [Stage 2/8] Skipped (SKIP_SENSITIVITY=1) ==="
fi

# ---- Stage 3: Time Robustness ----
if [[ "${SKIP_ROBUSTNESS}" == "0" ]]; then
    echo ""
    echo "=== [Stage 3/8] Time Robustness ==="
    python scripts/experiments/eval_time_robustness.py \
        --config "${CONFIG}" \
        --seed "${SEED}" \
        --num_neg "${NUM_NEG}" \
        --topk "${TOPK}" \
        --device "${DEVICE}" \
        --output_dir "${OUTPUT_DIR}" \
        --dataset "${DATASET}" \
        --hidden_dim "${HIDDEN_DIM}" \
        --data_dir "${DATA_DIR}"
    echo " Stage 3 done."
else
    echo ""
    echo "=== [Stage 3/8] Skipped (SKIP_ROBUSTNESS=1) ==="
fi

# ---- Stage 4: Efficiency Benchmark ----
if [[ "${SKIP_EFFICIENCY}" == "0" ]]; then
    echo ""
    echo "=== [Stage 4/8] Efficiency Benchmark ==="
    python scripts/experiments/exp_efficiency.py \
        --config "${CONFIG}" \
        --seed "${SEED}" \
        --num_neg "${NUM_NEG}" \
        --topk "${TOPK}" \
        --device "${DEVICE}" \
        --output_dir "${OUTPUT_DIR}" \
        --dataset "${DATASET}" \
        --hidden_dim "${HIDDEN_DIM}" \
        --data_dir "${DATA_DIR}"
    echo " Stage 4 done."
else
    echo ""
    echo "=== [Stage 4/8] Skipped (SKIP_EFFICIENCY=1) ==="
fi

# ---- Stage 5: Statistical Significance ----
if [[ "${SKIP_SIGNIFICANCE}" == "0" ]]; then
    echo ""
    echo "=== [Stage 5/8] Significance Analysis ==="
    python scripts/experiments/analyze_significance.py \
        --config "${CONFIG}" \
        --seed "${SEED}" \
        --num_neg "${NUM_NEG}" \
        --topk "${TOPK}" \
        --device "${DEVICE}" \
        --output_dir "${OUTPUT_DIR}" \
        --dataset "${DATASET}" \
        --hidden_dim "${HIDDEN_DIM}" \
        --data_dir "${DATA_DIR}"
    echo " Stage 5 done."
else
    echo ""
    echo "=== [Stage 5/8] Skipped (SKIP_SIGNIFICANCE=1) ==="
fi

# ---- Stage 6: Irregularity Analysis ----
if [[ "${SKIP_IRREGULARITY}" == "0" ]]; then
    echo ""
    echo "=== [Stage 6/8] Irregularity Analysis ==="
    python scripts/experiments/analyze_irregularity.py \
        --config "${CONFIG}" \
        --seed "${SEED}" \
        --num_neg "${NUM_NEG}" \
        --topk "${TOPK}" \
        --device "${DEVICE}" \
        --output_dir "${OUTPUT_DIR}" \
        --dataset "${DATASET}" \
        --hidden_dim "${HIDDEN_DIM}" \
        --data_dir "${DATA_DIR}"
    echo " Stage 6 done."
else
    echo ""
    echo "=== [Stage 6/8] Skipped (SKIP_IRREGULARITY=1) ==="
fi

# ---- Stage 7: Multi-Seed Stability ----
if [[ "${SKIP_MULTISEED}" == "0" ]]; then
    echo ""
    echo "=== [Stage 7/8] Multi-Seed Stability ==="
    DEVICE="${DEVICE}" HIDDEN_DIM="${HIDDEN_DIM}" \
    DATASET="${DATASET}" CONFIG="${CONFIG}" \
    NUM_NEG="${NUM_NEG}" TOPK="${TOPK}" \
    OUT_DIR="${OUTPUT_DIR}/multiseed" \
    bash scripts/experiments/run_multiseed.sh
    echo " Stage 7 done."
else
    echo ""
    echo "=== [Stage 7/8] Skipped (SKIP_MULTISEED=1) ==="
fi

# ---- Stage 8: Cleanup & Archive ----
if [[ "${SKIP_CLEANUP}" == "0" ]]; then
    echo ""
    echo "=== [Stage 8/8] Cleanup & Archive ==="
    bash scripts/experiments/cleanup_results.sh
    echo " Stage 8 done."
else
    echo ""
    echo "=== [Stage 8/8] Skipped (SKIP_CLEANUP=1) ==="
fi

echo ""
echo "============================================================"
echo " Phase 2 Complete!"
echo " Results directory: ${OUTPUT_DIR}/"
echo " Finished: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo " Expected outputs:"
echo "   ${OUTPUT_DIR}/baseline_tuning/     - SASRec/TiSASRec grid search"
echo "   ${OUTPUT_DIR}/time_sensitivity.csv  - Original vs shuffled time"
echo "   ${OUTPUT_DIR}/time_robustness.csv   - Noise/dropout corruption"
echo "   ${OUTPUT_DIR}/efficiency.csv        - Latency & param counts"
echo "   ${OUTPUT_DIR}/significance_*.csv    - Paired t-test results"
echo "   ${OUTPUT_DIR}/irregularity_*.csv    - CV-based analysis"
echo "   ${OUTPUT_DIR}/multiseed/            - 5-seed stability"
echo "============================================================"
