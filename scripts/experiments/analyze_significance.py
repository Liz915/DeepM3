#!/usr/bin/env python3
"""
Statistical Significance Analysis — Paired t-test + sequence-length breakdown.
Compares Baseline (GRU) vs DeepM3 (Euler) on per-user NDCG@10.
Saves CSV outputs to the specified output directory.
"""
import argparse
import copy
import math
import os
import sys

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from scipy import stats  # type: ignore
except Exception:
    stats = None

sys.path.append(os.getcwd())
from src.data.dataset import MovieLensDataset
from src.dynamics.modeling import DeepM3Model
from src.dynamics.gru_baseline import GRUBaseline


def pick_device(name="auto"):
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if name == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
    return torch.device("cpu")


def infer_hidden_dim(state_dict):
    w = state_dict.get("item_emb.weight")
    if w is not None and getattr(w, "ndim", 0) == 2:
        return int(w.shape[1])
    gru_w = state_dict.get("gru.weight_ih_l0")
    if gru_w is not None and getattr(gru_w, "ndim", 0) == 2:
        return int(gru_w.shape[0] // 3)
    return None


def compute_ndcg_per_user(scores, k=10):
    _, indices = torch.topk(scores, k, dim=-1)
    if 0 in indices[0]:
        rank = (indices[0] == 0).nonzero(as_tuple=True)[0].item()
        return 1.0 / np.log2(rank + 2)
    return 0.0


def build_neg_pool(dataset, num_neg, seed):
    rng = np.random.default_rng(seed)
    pool = []
    for sample in dataset.samples:
        blocked = {int(sample["y"])}
        blocked.update(int(v) for v in sample["x"] if int(v) > 0)
        negs = []
        while len(negs) < num_neg:
            c = int(rng.integers(1, dataset.n_items))
            if c not in blocked:
                negs.append(c)
        pool.append(np.asarray(negs, dtype=np.int64))
    return pool


def paired_t_test(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if stats is not None:
        t_stat, p_val = stats.ttest_rel(x, y)
        return float(t_stat), float(p_val)

    diff = x - y
    n = diff.size
    if n < 2:
        return float("nan"), float("nan")
    diff_mean = float(np.mean(diff))
    diff_std = float(np.std(diff, ddof=1))
    if diff_std == 0:
        return float("inf"), 0.0
    t_stat = diff_mean / (diff_std / math.sqrt(n))
    p_val = math.erfc(abs(t_stat) / math.sqrt(2.0))
    return float(t_stat), float(p_val)


def run_analysis():
    parser = argparse.ArgumentParser(description="Statistical significance analysis")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--baseline_ckpt", default="")
    parser.add_argument("--euler_ckpt", default="")
    parser.add_argument("--output_dir", default="results/ml1m")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_neg", type=int, default=100)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--data_dir", type=str, default="")
    parser.add_argument("--dataset", type=str, default="ml1m")
    parser.add_argument("--hidden_dim", type=int, default=128)
    args = parser.parse_args()

    device = pick_device(args.device)
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    data_dir = args.data_dir or ("data/amazon" if args.dataset == "amazon" else "data")
    dataset = MovieLensDataset(mode="test", data_dir=data_dir)
    neg_pool = build_neg_pool(dataset, args.num_neg, args.seed)
    print(f"Test samples: {len(dataset)} | device={device}")

    ds = args.dataset
    dim = args.hidden_dim
    
    def get_ckpt(ds, dim, tag):
        old_path = f"checkpoints/model_{tag}.pth"
        new_path = f"checkpoints/{ds}_deepm3_{tag}_d{dim}.pth" if tag != "gru_baseline" else f"checkpoints/{ds}_gru_d{dim}.pth"
        return old_path if (ds == "ml1m" and os.path.exists(old_path)) else new_path

    # Check for CLI overrides, else calculate generic checkpoint path
    base_ckpt = args.baseline_ckpt if args.baseline_ckpt else get_ckpt(ds, dim, "gru_baseline")
    euler_ckpt = args.euler_ckpt if args.euler_ckpt else get_ckpt(ds, dim, "euler")

    # Load Baseline
    print(f" Loading Baseline from {base_ckpt}...")
    base_state = torch.load(base_ckpt, map_location=device, weights_only=False)
    base_dim = infer_hidden_dim(base_state) or config["model"]["hidden_dim"]
    base_model = GRUBaseline(n_items=dataset.n_items, hidden_dim=base_dim).to(device)
    base_model.load_state_dict(base_state, strict=True)
    base_model.eval()

    # Load DeepM3 (Euler)
    print(f" Loading DeepM3 (Euler) from {euler_ckpt}...")
    euler_state = torch.load(euler_ckpt, map_location=device, weights_only=False)
    euler_dim = infer_hidden_dim(euler_state) or config["model"]["hidden_dim"]
    local_cfg = copy.deepcopy(config)
    local_cfg["model"]["hidden_dim"] = euler_dim
    ode_model = DeepM3Model(local_cfg, n_items=dataset.n_items, solver="euler").to(device)
    ode_model.load_state_dict(euler_state, strict=True)
    ode_model.eval()

    results = []

    print(" Collecting per-user metrics...")
    with torch.no_grad():
        for idx in tqdm(range(len(dataset))):
            sample = dataset[idx]
            x = sample["x"].unsqueeze(0).to(device)
            t = sample["t"].unsqueeze(0).to(device)
            dt = sample["dt"].unsqueeze(0).to(device) if "dt" in sample else None
            pos = int(sample["pos"].item())

            # Fixed negatives
            cands = np.concatenate(([pos], neg_pool[idx]))
            cand_t = torch.tensor(cands, dtype=torch.long, device=device)

            seq_len = int((x > 0).sum().item())

            # Baseline
            u_base = base_model(x)
            s_base = torch.matmul(u_base, base_model.get_item_embedding(cand_t).t())
            n_base = compute_ndcg_per_user(s_base, k=args.topk)

            # ODE (Euler)
            u_ode = ode_model(x, t, dt=dt)
            s_ode = torch.matmul(u_ode, ode_model.get_item_embedding(cand_t).t())
            n_ode = compute_ndcg_per_user(s_ode, k=args.topk)

            results.append({
                "len": seq_len,
                "ndcg_base": n_base,
                "ndcg_ode": n_ode,
                "diff": n_ode - n_base,
            })

    df = pd.DataFrame(results)
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Overall Paired t-test
    print("\n" + "=" * 50)
    print(" 1. Statistical Significance Test (Paired t-test)")
    print("=" * 50)
    t_stat, p_val = paired_t_test(df["ndcg_ode"], df["ndcg_base"])
    print(f"Mean NDCG (Base) : {df['ndcg_base'].mean():.4f}")
    print(f"Mean NDCG (Euler): {df['ndcg_ode'].mean():.4f}")
    print(f"T-statistic: {t_stat:.4f}")
    print(f"P-value: {p_val:.4e}")

    if p_val < 0.05:
        print(" RESULT: Statistically Significant (p < 0.05)!")
    elif p_val < 0.1:
        print(" RESULT: Marginally Significant (0.05 < p < 0.1)")
    else:
        print(" RESULT: Not significant.")

    # 2. By Sequence Length
    print("\n" + "=" * 50)
    print(" 2. Performance by Sequence Length")
    print("=" * 50)
    df["len_group"] = pd.cut(
        df["len"], bins=[0, 10, 15, np.inf],
        labels=["Short (<=10)", "Medium (11-15)", "Long (>15)"],
    )
    summary = df.groupby("len_group", observed=False)[["ndcg_base", "ndcg_ode"]].mean()
    summary["improvement_pct"] = (
        (summary["ndcg_ode"] - summary["ndcg_base"]) / (summary["ndcg_base"] + 1e-12) * 100
    )

    # Per-group significance
    group_sig = []
    for group_name in ["Short (<=10)", "Medium (11-15)", "Long (>15)"]:
        g = df[df["len_group"] == group_name]
        if len(g) > 1:
            t_g, p_g = paired_t_test(g["ndcg_ode"], g["ndcg_base"])
            group_sig.append({"group": group_name, "t_stat": t_g, "p_value": p_g, "n": len(g)})
    group_sig_df = pd.DataFrame(group_sig)

    print(summary)
    print("\nPer-group significance:")
    print(group_sig_df.to_string(index=False))

    # Save CSVs
    sig_path = os.path.join(args.output_dir, "significance_detailed.csv")
    pd.DataFrame([{
        "mean_ndcg_baseline": df["ndcg_base"].mean(),
        "mean_ndcg_euler": df["ndcg_ode"].mean(),
        "t_statistic": t_stat,
        "p_value": p_val,
        "n_users": len(df),
    }]).to_csv(sig_path, index=False)

    len_path = os.path.join(args.output_dir, "significance_by_length.csv")
    summary.reset_index().to_csv(len_path, index=False)

    group_sig_path = os.path.join(args.output_dir, "significance_by_length_pval.csv")
    group_sig_df.to_csv(group_sig_path, index=False)

    print(f"\nSaved to: {sig_path}, {len_path}, {group_sig_path}")


if __name__ == "__main__":
    run_analysis()
