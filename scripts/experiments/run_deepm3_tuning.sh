#!/bin/bash
# ============================================================
# DeepM3 Hyperparameter Tuning (solver x lr x wd x epochs)
# ============================================================
# Example:
#   DATASET=amazon DEVICE=mps HIDDEN_DIM=128 \
#   SOLVERS="none,euler,rk4" LRS="1e-4,3e-4,1e-3" WDS="1e-5,1e-4" EPOCHS_LIST="15,30" \
#   bash scripts/experiments/run_deepm3_tuning.sh
# ============================================================
set -eo pipefail
cd "$(dirname "$0")/../.."

DATASET="${DATASET:-ml1m}"
DEVICE="${DEVICE:-auto}"
HIDDEN_DIM="${HIDDEN_DIM:-128}"
SEED="${SEED:-42}"
CONFIG="${CONFIG:-configs/config.yaml}"
NUM_NEG="${NUM_NEG:-100}"
TOPK="${TOPK:-10}"

SOLVERS="${SOLVERS:-none,euler,rk4}"
LRS="${LRS:-1e-4,3e-4,1e-3}"
WDS="${WDS:-1e-5,1e-4}"
EPOCHS_LIST="${EPOCHS_LIST:-15,30}"

DATA_DIR="${DATA_DIR:-}"
if [[ -z "${DATA_DIR}" && "${DATASET}" == "amazon" ]]; then
  DATA_DIR="data/amazon"
fi
DATA_ARGS=()
if [[ -n "${DATA_DIR}" ]]; then
  DATA_ARGS+=(--data_dir "$DATA_DIR")
fi

OUTPUT_DIR="${OUTPUT_DIR:-results/${DATASET}/deepm3_tuning}"
mkdir -p "${OUTPUT_DIR}"
mkdir -p checkpoints/tuning

ALL_CSV="${OUTPUT_DIR}/deepm3_tuning_all.csv"
BEST_CSV="${OUTPUT_DIR}/deepm3_tuning_best.csv"
export ALL_CSV
export BEST_CSV

IFS=',' read -r -a SOLVER_ARR <<< "${SOLVERS}"
IFS=',' read -r -a LR_ARR <<< "${LRS}"
IFS=',' read -r -a WD_ARR <<< "${WDS}"
IFS=',' read -r -a EPOCH_ARR <<< "${EPOCHS_LIST}"

total_runs=$(( ${#SOLVER_ARR[@]} * ${#LR_ARR[@]} * ${#WD_ARR[@]} * ${#EPOCH_ARR[@]} ))
run_id=0

echo "model,solver,lr,weight_decay,epochs,hidden_dim,seed,checkpoint,hr@10,ndcg@10,latency_ms" > "${ALL_CSV}"

echo "============================================================"
echo " DeepM3 Tuning"
echo " Dataset=${DATASET} | dim=${HIDDEN_DIM} | device=${DEVICE}"
echo " Grid: solver(${SOLVERS}) x lr(${LRS}) x wd(${WDS}) x epochs(${EPOCHS_LIST})"
echo " Total runs: ${total_runs}"
echo "============================================================"

for solver in "${SOLVER_ARR[@]}"; do
  for lr in "${LR_ARR[@]}"; do
    for wd in "${WD_ARR[@]}"; do
      for epochs in "${EPOCH_ARR[@]}"; do
        run_id=$((run_id + 1))
        lr_tag="${lr//./p}"
        wd_tag="${wd//./p}"
        ckpt_name="tune_${DATASET}_deepm3_${solver}_d${HIDDEN_DIM}_lr${lr_tag}_wd${wd_tag}_e${epochs}_s${SEED}.pth"
        ckpt_rel="tuning/${ckpt_name}"
        ckpt_abs="checkpoints/${ckpt_rel}"

        echo ""
        echo "------------------------------------------------------------------------"
        echo "[${run_id}/${total_runs}] solver=${solver} lr=${lr} wd=${wd} epochs=${epochs} dim=${HIDDEN_DIM}"

        python scripts/train/train_unified.py \
          --model deepm3 \
          --solver "${solver}" \
          --epochs "${epochs}" \
          --seed "${SEED}" \
          --hidden_dim "${HIDDEN_DIM}" \
          --dataset "${DATASET}" \
          --save_name "${ckpt_rel}" \
          --lr "${lr}" \
          --weight_decay "${wd}" \
          --device "${DEVICE}" \
          --config "${CONFIG}" \
          "${DATA_ARGS[@]}"

        OUT=$(python scripts/eval/evaluate.py \
          --model_path "${ckpt_abs}" \
          --solver "${solver}" \
          --seed "${SEED}" \
          --num_neg "${NUM_NEG}" \
          --topk "${TOPK}" \
          --dataset "${DATASET}" \
          --device "${DEVICE}" \
          "${DATA_ARGS[@]}")

        OUT_LINE=$(echo "${OUT}" | tail -n1)
        IFS=',' read -r _path hr ndcg lat <<< "${OUT_LINE}"
        if [[ -z "${hr:-}" || -z "${ndcg:-}" || -z "${lat:-}" ]]; then
          echo "Failed to parse evaluation output:"
          echo "${OUT}"
          exit 1
        fi

        echo "Result: HR@10=${hr} NDCG@10=${ndcg} Latency=${lat}ms"
        echo "deepm3,${solver},${lr},${wd},${epochs},${HIDDEN_DIM},${SEED},${ckpt_abs},${hr},${ndcg},${lat}" >> "${ALL_CSV}"
      done
    done
  done
done

python - <<'PY'
import pandas as pd
from pathlib import Path
import os

all_csv = Path(os.environ["ALL_CSV"])
best_csv = Path(os.environ["BEST_CSV"])

df = pd.read_csv(all_csv)
df = df.sort_values(["solver", "ndcg@10"], ascending=[True, False])
best_solver = df.groupby("solver", as_index=False).head(1)
best_overall = df.head(1).copy()
best_overall["solver"] = best_overall["solver"].astype(str) + " (overall best)"
best = pd.concat([best_solver, best_overall], ignore_index=True)
best.to_csv(best_csv, index=False)

print("\nBest per solver:")
print(best_solver[["solver", "lr", "weight_decay", "epochs", "hr@10", "ndcg@10", "latency_ms"]].to_string(index=False))
print("\nOverall best:")
print(best_overall[["solver", "lr", "weight_decay", "epochs", "hr@10", "ndcg@10", "latency_ms"]].to_string(index=False))
print(f"\nSaved full results: {all_csv}")
print(f"Saved best results: {best_csv}")
PY

echo ""
echo "=== DeepM3 Tuning Complete ==="
