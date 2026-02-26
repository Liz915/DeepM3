#!/bin/bash
# E9: Multi-Seed Stability — Train & evaluate across 5 seeds
# Uses train_unified.py for consistent interface
set -euo pipefail
cd "$(dirname "$0")/../.."

SEEDS="${SEEDS:-42 52 62 72 82}"
SOLVER="${SOLVER:-euler}"
EPOCHS="${EPOCHS:-15}"
HIDDEN_DIM="${HIDDEN_DIM:-128}"
DEVICE="${DEVICE:-auto}"
DATASET="${DATASET:-ml1m}"
CONFIG="${CONFIG:-configs/config.yaml}"
NUM_NEG="${NUM_NEG:-100}"
TOPK="${TOPK:-10}"
OUT_DIR="${OUT_DIR:-results/ml1m/multiseed}"
mkdir -p "$OUT_DIR"
BASE_RAW="${OUT_DIR}/raw_baseline.txt"
ODE_RAW="${OUT_DIR}/raw_ode.txt"

# Reset raw logs to avoid appending stale runs.
: > "$BASE_RAW"
: > "$ODE_RAW"

echo "=== E9: Multi-Seed Stability Test ==="
echo "  Seeds: $SEEDS | Solver: $SOLVER | Epochs: $EPOCHS | Dim: $HIDDEN_DIM"

for SEED in $SEEDS; do
    echo ""
    echo "--- Seed $SEED ---"

    # Train baseline (GRU)
    python scripts/train/train_unified.py \
        --model gru --epochs $EPOCHS --seed $SEED \
        --hidden_dim $HIDDEN_DIM --dataset $DATASET \
        --save_name "multiseed/seed${SEED}_baseline.pth" \
        --device $DEVICE --config "$CONFIG"

    # Train DeepM3 (Euler)
    python scripts/train/train_unified.py \
        --model deepm3 --solver $SOLVER --epochs $EPOCHS --seed $SEED \
        --hidden_dim $HIDDEN_DIM --dataset $DATASET \
        --save_name "multiseed/seed${SEED}_${SOLVER}.pth" \
        --device $DEVICE --config "$CONFIG"

    # Evaluate both
    python scripts/eval/evaluate.py \
        --model_path "checkpoints/multiseed/seed${SEED}_baseline.pth" \
        --solver baseline \
        --seed $SEED --num_neg $NUM_NEG --topk $TOPK \
        --device $DEVICE --config "$CONFIG" \
        >> "$BASE_RAW"

    python scripts/eval/evaluate.py \
        --model_path "checkpoints/multiseed/seed${SEED}_${SOLVER}.pth" \
        --solver $SOLVER \
        --seed $SEED --num_neg $NUM_NEG --topk $TOPK \
        --device $DEVICE --config "$CONFIG" \
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
        with open(fpath) as f:
            for i, line in enumerate(f):
                parts = line.strip().split(',')
                if len(parts) == 4:
                    rows.append({
                        'seed': seeds[i] if i < len(seeds) else f's{i}',
                        'method': tag,
                        'solver': solver,
                        'hr@10': float(parts[1]),
                        'ndcg@10': float(parts[2]),
                        'time_per_sample_ms': float(parts[3]),
                    })
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
        t_stat, p_val = stats.ttest_rel(euler_ndcg, base_ndcg)
        agg['t_stat'] = ''
        agg['p_val'] = ''
        agg.loc[agg['method']=='ode', 't_stat'] = f'{t_stat:.4f}'
        agg.loc[agg['method']=='ode', 'p_val'] = f'{p_val:.4e}'
except Exception:
    pass

df.to_csv(f'{out_dir}/multiseed_raw.csv', index=False)
agg.to_csv(f'{out_dir}/multiseed_summary.csv', index=False)
print(agg.to_string(index=False))
print(f'\nSaved to {out_dir}/multiseed_summary.csv')
"

echo "=== Multi-Seed Test Complete ==="
