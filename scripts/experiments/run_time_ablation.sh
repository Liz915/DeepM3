#!/bin/bash
# ============================================================
# Time Signal Ablation Pipeline (no-time, t-only, dt-only, t+dt)
# ============================================================
set -eo pipefail
cd "$(dirname "$0")/../.."

DATASET="${DATASET:-amazon}"
DEVICE="${DEVICE:-auto}"
HIDDEN_DIM="${HIDDEN_DIM:-128}"
EPOCHS="${EPOCHS:-15}"
SEED="${SEED:-42}"
CONFIG="${CONFIG:-configs/config.yaml}"
NUM_NEG="${NUM_NEG:-100}"
TOPK="${TOPK:-10}"
SOLVER="${SOLVER:-euler}"
LR="${LR:-3e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
OUTPUT_DIR="${OUTPUT_DIR:-results/${DATASET}/ablation}"
DATA_DIR="${DATA_DIR:-}"
if [[ -z "${DATA_DIR}" && "${DATASET}" == "amazon" ]]; then
    DATA_DIR="data/amazon"
fi
DATA_ARGS=()
if [[ -n "${DATA_DIR}" ]]; then
    DATA_ARGS+=(--data_dir "$DATA_DIR")
fi

mkdir -p "$OUTPUT_DIR"
ABLATION_CSV="${OUTPUT_DIR}/time_ablation.csv"

echo "============================================================"
echo " Time Signal Ablation Pipeline"
echo " Dataset: ${DATASET} | dim=${HIDDEN_DIM} | device=${DEVICE}"
echo "============================================================"

# List of modes
MODES=("none" "t_only" "dt_only" "full")
LABELS=("no-time" "t-only" "dt-only" "t+dt(full)")

# Reset CSV
echo "mode,label,hr@10,ndcg@10,latency_ms" > "$ABLATION_CSV"

for i in "${!MODES[@]}"; do
    MODE="${MODES[$i]}"
    LABEL="${LABELS[$i]}"
    
    echo ""
    echo "--- Running Ablation: ${LABEL} (${MODE}) ---"
    
    CKPT="checkpoints/ablation/${DATASET}_deepm3_ablate_${MODE}.pth"
    
    # Train
    python scripts/train/train_unified.py \
        --model deepm3 --solver $SOLVER --epochs $EPOCHS --seed $SEED \
        --hidden_dim $HIDDEN_DIM --dataset $DATASET \
        --save_name "ablation/${DATASET}_deepm3_ablate_${MODE}.pth" \
        --time_ablation "$MODE" \
        --lr "$LR" \
        --weight_decay "$WEIGHT_DECAY" \
        --config "$CONFIG" \
        "${DATA_ARGS[@]}" \
        --device $DEVICE
        
    # Evaluate
    echo " evaluating..."
    OUT=$(python scripts/eval/evaluate.py \
        --model_path "$CKPT" \
        --solver $SOLVER \
        --seed $SEED --num_neg $NUM_NEG --topk $TOPK \
        --time_ablation "$MODE" \
        --dataset $DATASET \
        --config "$CONFIG" \
        "${DATA_ARGS[@]}" \
        --device $DEVICE)

    OUT_LINE=$(echo "$OUT" | tail -n1)
    # parse output
    IFS=',' read -r path hr ndcg lat <<< "$OUT_LINE"
    if [[ -z "${hr:-}" || -z "${ndcg:-}" || -z "${lat:-}" ]]; then
        echo "Failed to parse evaluation output for mode=${MODE}"
        echo "Raw output:"
        echo "$OUT"
        exit 1
    fi
    echo "Result for ${LABEL}: HR=${hr} NDCG=${ndcg} Latency=${lat}ms"
    echo "${MODE},${LABEL},${hr},${ndcg},${lat}" >> "$ABLATION_CSV"
done

echo ""
echo "=== Time Signal Ablation Complete ==="
cat "$ABLATION_CSV"
