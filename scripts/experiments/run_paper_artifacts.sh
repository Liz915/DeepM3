#!/bin/bash
set -euo pipefail

SEED=${SEED:-42}
NUM_NEG=${NUM_NEG:-100}
TOPK=${TOPK:-10}
CONFIG=${CONFIG:-configs/config.yaml}
OUTPUT_DIR=${OUTPUT_DIR:-results/ml1m}
ASSETS_DIR=${ASSETS_DIR:-assets}

python scripts/experiments/build_paper_artifacts.py \
    --config "${CONFIG}" \
    --seed "${SEED}" \
    --num_neg "${NUM_NEG}" \
    --topk "${TOPK}" \
    --output_dir "${OUTPUT_DIR}" \
    --assets_dir "${ASSETS_DIR}"

echo " Done. Tables/CSVs: ${OUTPUT_DIR} | Figures: ${ASSETS_DIR}"
