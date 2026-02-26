#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT_DIR}"

STAMP="$(date +%Y%m%d_%H%M%S)"
ARCHIVE_DIR="results/archive/${STAMP}"
CANONICAL_DIR="results/ml1m"

mkdir -p "${ARCHIVE_DIR}" "${CANONICAL_DIR}"

sync_csv_if_exists() {
  local src_dir="$1"
  if [[ -d "${src_dir}" ]]; then
    shopt -s nullglob
    for f in "${src_dir}"/*.csv; do
      cp -f "${f}" "${CANONICAL_DIR}/"
    done
    shopt -u nullglob
  fi
}

echo "Canonical results directory: ${CANONICAL_DIR}"
echo "Archive directory: ${ARCHIVE_DIR}"

# Keep canonical copies before archiving legacy folders.
sync_csv_if_exists "results/paper"
sync_csv_if_exists "results/paper_dim256"

for path in \
  "results/ablation_report.csv" \
  "results/paper" \
  "results/paper_dim256" \
  "results/paper_multiseed"
do
  if [[ -e "${path}" ]]; then
    echo "Archiving ${path}"
    mv "${path}" "${ARCHIVE_DIR}/"
  fi
done

echo "Done. Canonical outputs live in ${CANONICAL_DIR}/"
