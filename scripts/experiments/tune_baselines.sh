#!/bin/bash
set -euo pipefail

DATASET=${DATASET:-ml1m}
SEED=${SEED:-42}
HIDDEN_DIM=${HIDDEN_DIM:-256}
LRS=${LRS:-"1e-4,3e-4,5e-4"}
EPOCHS_LIST=${EPOCHS_LIST:-"50,100,150"}
MODELS=${MODELS:-"sasrec,tisasrec"}
NUM_NEG=${NUM_NEG:-100}
TOPK=${TOPK:-10}
DEVICE=${DEVICE:-auto}
CONFIG=${CONFIG:-configs/config.yaml}
DATA_DIR=${DATA_DIR:-}
OUTPUT_DIR=${OUTPUT_DIR:-results/${DATASET}/baseline_tuning}

cmd=(
  python scripts/experiments/tune_baselines.py
  --dataset "${DATASET}"
  --seed "${SEED}"
  --hidden_dim "${HIDDEN_DIM}"
  --models "${MODELS}"
  --lrs "${LRS}"
  --epochs_list "${EPOCHS_LIST}"
  --num_neg "${NUM_NEG}"
  --topk "${TOPK}"
  --device "${DEVICE}"
  --config "${CONFIG}"
  --output_dir "${OUTPUT_DIR}"
)

if [[ -n "${DATA_DIR}" ]]; then
  cmd+=(--data_dir "${DATA_DIR}")
fi

"${cmd[@]}"
