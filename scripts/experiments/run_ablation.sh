#!/bin/bash
set -euo pipefail

SEED=${SEED:-42}
NUM_NEG=${NUM_NEG:-100}
CONFIG=${CONFIG:-configs/config.yaml}
OUT_DIR=${OUT_DIR:-results/ml1m}

echo " Starting Ablation Study (seed=${SEED}, num_neg=${NUM_NEG})..."
mkdir -p "${OUT_DIR}"
REPORT_FILE="${OUT_DIR}/ablation_report.csv"
echo "Model,Solver,HR@10,NDCG@10,Latency(ms)" > "${REPORT_FILE}"

eval_and_append () {
    local model_name="$1"
    local solver="$2"
    local ckpt="$3"
    local result_line
    result_line=$(python scripts/eval/evaluate.py \
        --config "${CONFIG}" \
        --model_path "${ckpt}" \
        --solver "${solver}" \
        --seed "${SEED}" \
        --num_neg "${NUM_NEG}" | tail -n 1)
    local values
    values=$(echo "${result_line}" | awk -F',' '{print $2 "," $3 "," $4}')
    echo "${model_name},${solver},${values}" >> "${REPORT_FILE}"
    echo " ${model_name} (${solver}) done: ${values}"
}

echo "--------------------------------"
echo " [1/4] Baseline GRU..."
python scripts/train/train.py \
    --seed "${SEED}" \
    --solver baseline \
    --save_name "model_gru_baseline.pth" \
    --config "${CONFIG}"
eval_and_append "Baseline" "baseline" "checkpoints/model_gru_baseline.pth"

echo "--------------------------------"
echo " [2/4] DeepM3 (None Solver)..."
python scripts/train/train.py \
    --seed "${SEED}" \
    --solver none \
    --save_name "model_ode_none.pth" \
    --config "${CONFIG}"
eval_and_append "DeepM3" "none" "checkpoints/model_ode_none.pth"

echo "--------------------------------"
echo " [3/4] DeepM3 (Euler Solver)..."
python scripts/train/train.py \
    --seed "${SEED}" \
    --solver euler \
    --save_name "model_ode_euler.pth" \
    --config "${CONFIG}"
eval_and_append "DeepM3" "euler" "checkpoints/model_ode_euler.pth"

echo "--------------------------------"
echo " [4/4] DeepM3 (RK4 Solver)..."
python scripts/train/train.py \
    --seed "${SEED}" \
    --solver rk4 \
    --save_name "model_ode_rk4.pth" \
    --config "${CONFIG}"
eval_and_append "DeepM3" "rk4" "checkpoints/model_ode_rk4.pth"

echo "--------------------------------"
echo " Ablation Study Finished."
cat "${REPORT_FILE}"
