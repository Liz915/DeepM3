#!/bin/bash
# E9: Multi-Seed Stability — Train & evaluate across multiple seeds
# Uses train_unified.py for consistent interface
set -euo pipefail
cd "$(dirname "$0")/../.."

SEEDS="${SEEDS:-42 52 62 72 82 92 102 112 122 132}"
SOLVER="${SOLVER:-euler}"
EPOCHS="${EPOCHS:-15}"
HIDDEN_DIM="${HIDDEN_DIM:-128}"
DEVICE="${DEVICE:-auto}"
DATASET="${DATASET:-ml1m}"
CONFIG="${CONFIG:-configs/config.yaml}"
NUM_NEG="${NUM_NEG:-100}"
TOPK="${TOPK:-10}"
OUT_DIR="${OUT_DIR:-results/${DATASET}/multiseed}"
DATA_DIR="${DATA_DIR:-}"
if [[ -z "${DATA_DIR}" && "${DATASET}" == "amazon" ]]; then
    DATA_DIR="data/amazon"
fi
mkdir -p "$OUT_DIR"
BASE_RAW="${OUT_DIR}/raw_baseline.txt"
ODE_RAW="${OUT_DIR}/raw_ode.txt"

# Reset raw logs to avoid appending stale runs.
: > "$BASE_RAW"
: > "$ODE_RAW"

echo "=== E9: Multi-Seed Stability Test ==="
echo "  Seeds: $SEEDS | Solver: $SOLVER | Epochs: $EPOCHS | Dim: $HIDDEN_DIM"

TRAIN_DATA_ARGS=()
EVAL_DATA_ARGS=(--dataset "$DATASET")
if [[ -n "${DATA_DIR}" ]]; then
    TRAIN_DATA_ARGS+=(--data_dir "$DATA_DIR")
    EVAL_DATA_ARGS+=(--data_dir "$DATA_DIR")
fi

for SEED in $SEEDS; do
    echo ""
    echo "--- Seed $SEED ---"

    # Train baseline (GRU)
    python scripts/train/train_unified.py \
        --model gru --epochs $EPOCHS --seed $SEED \
        --hidden_dim $HIDDEN_DIM --dataset $DATASET \
        --save_name "multiseed/seed${SEED}_baseline.pth" \
        --device $DEVICE --config "$CONFIG" \
        "${TRAIN_DATA_ARGS[@]}"

    # Train DeepM3 (Euler)
    python scripts/train/train_unified.py \
        --model deepm3 --solver $SOLVER --epochs $EPOCHS --seed $SEED \
        --hidden_dim $HIDDEN_DIM --dataset $DATASET \
        --save_name "multiseed/seed${SEED}_${SOLVER}.pth" \
        --device $DEVICE --config "$CONFIG" \
        "${TRAIN_DATA_ARGS[@]}"

    # Evaluate both
    python scripts/eval/evaluate.py \
        --model_path "checkpoints/multiseed/seed${SEED}_baseline.pth" \
        --solver baseline \
        --seed $SEED --num_neg $NUM_NEG --topk $TOPK \
        --device $DEVICE --config "$CONFIG" \
        "${EVAL_DATA_ARGS[@]}" \
        >> "$BASE_RAW"

    python scripts/eval/evaluate.py \
        --model_path "checkpoints/multiseed/seed${SEED}_${SOLVER}.pth" \
        --solver $SOLVER \
        --seed $SEED --num_neg $NUM_NEG --topk $TOPK \
        --device $DEVICE --config "$CONFIG" \
        "${EVAL_DATA_ARGS[@]}" \
        >> "$ODE_RAW"

    echo "Seed $SEED done."
done

echo ""
echo "=== Merging multi-seed results ==="

python3 -c "
import pandas as pd
import numpy as np

seeds = '${SEEDS}'.split()
out_dir = '${OUT_DIR}'

# Parse evaluate.py output: model_path,hr,ndcg,latency
rows = []
for tag, solver in [('baseline', 'baseline'), ('ode', '${SOLVER}')]:
    fpath = f'{out_dir}/raw_{tag}.txt'
    try:
        result_idx = 0
        with open(fpath) as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) == 4:
                    rows.append({
                        'seed': seeds[result_idx] if result_idx < len(seeds) else f's{result_idx}',
                        'method': tag,
                        'solver': solver,
                        'hr@10': float(parts[1]),
                        'ndcg@10': float(parts[2]),
                        'time_per_sample_ms': float(parts[3]),
                    })
                    result_idx += 1
    except FileNotFoundError:
        print(f'Warning: {fpath} not found')

if not rows:
    print('No results found!'); exit(1)

df = pd.DataFrame(rows)

# Compute mean ± std per method
agg = df.groupby(['method', 'solver']).agg(
    hr_mean=('hr@10', 'mean'),
    hr_std=('hr@10', 'std'),
    ndcg_mean=('ndcg@10', 'mean'),
    ndcg_std=('ndcg@10', 'std'),
    lat_mean=('time_per_sample_ms', 'mean'),
).reset_index()

# Also run paired t-test if scipy available
try:
    from scipy import stats
    base_ndcg = df[df['method']=='baseline']['ndcg@10'].values
    euler_ndcg = df[df['method']=='ode']['ndcg@10'].values
    if len(base_ndcg) == len(euler_ndcg) and len(base_ndcg) > 1:
        # Paired t-test
        t_stat, p_val = stats.ttest_rel(euler_ndcg, base_ndcg)
        # Wilcoxon signed-rank test
        diff = euler_ndcg - base_ndcg
        diff = diff[diff != 0]
        if len(diff) > 0:
            w_stat, w_pval = stats.wilcoxon(diff)
        else:
            w_stat, w_pval = np.nan, np.nan
            
        # Paired effect size (Cohen's dz): mean(diff) / std(diff)
        diff_std = np.std(diff, ddof=1) if len(diff) > 1 else np.nan
        cohens_dz = np.mean(diff) / diff_std if (not np.isnan(diff_std) and diff_std > 0) else 0.0
            
        agg['ttest_pval'] = ''
        agg['ttest_stat'] = ''
        agg['wilcoxon_pval'] = ''
        agg['wilcoxon_stat'] = ''
        agg['effect_size_dz'] = ''
        agg['effect_size_d'] = ''
        agg.loc[agg['method']=='ode', 'ttest_stat'] = f'{t_stat:.4f}'
        agg.loc[agg['method']=='ode', 'ttest_pval'] = f'{p_val:.4e}'
        agg.loc[agg['method']=='ode', 'wilcoxon_stat'] = f'{w_stat:.4f}' if not np.isnan(w_stat) else ''
        agg.loc[agg['method']=='ode', 'wilcoxon_pval'] = f'{w_pval:.4e}'
        agg.loc[agg['method']=='ode', 'effect_size_dz'] = f'{cohens_dz:.4f}'
        agg.loc[agg['method']=='ode', 'effect_size_d'] = f'{cohens_dz:.4f}'
except Exception as e:
    print(f"Stats error: {e}")

df.to_csv(f'{out_dir}/multiseed_raw.csv', index=False)
agg.to_csv(f'{out_dir}/multiseed_summary.csv', index=False)
print(agg.to_string(index=False))
print(f'\nSaved to {out_dir}/multiseed_summary.csv')
"

echo "=== Multi-Seed Test Complete ==="
