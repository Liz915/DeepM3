#!/usr/bin/env python3
"""
Irregularity Analysis — Evaluates how well DeepM3 vs GRU performs
on users with varying temporal irregularity (Coefficient of Variation of inter-event gaps).
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


def compute_ndcg(scores, k=10):
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


def analyze_metrics():
    parser = argparse.ArgumentParser(description="Irregularity analysis")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--baseline_ckpt", default="")
    parser.add_argument("--ode_ckpt", default="")
    parser.add_argument("--ode_solver", default="euler", choices=["euler", "rk4", "none"])
    # Backward-compatible alias for old command lines.
    parser.add_argument("--rk4_ckpt", default="")
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

    base_ckpt = args.baseline_ckpt if args.baseline_ckpt else get_ckpt(ds, dim, "gru_baseline")

    # Load Baseline
    print(f" Loading Baseline (GRU) from {base_ckpt}...")
    base_state = torch.load(base_ckpt, map_location=device, weights_only=False)
    base_dim = infer_hidden_dim(base_state) or config["model"]["hidden_dim"]
    base_model = GRUBaseline(n_items=dataset.n_items, hidden_dim=base_dim).to(device)
    base_model.load_state_dict(base_state, strict=True)
    base_model.eval()

    ode_ckpt = args.ode_ckpt.strip()
    if not ode_ckpt:
        if args.rk4_ckpt.strip():
            ode_ckpt = args.rk4_ckpt.strip()
            args.ode_solver = "rk4"
        else:
            ode_ckpt = get_ckpt(ds, dim, args.ode_solver)

    # Load ODE model
    print(f" Loading DeepM3 ({args.ode_solver}) from {ode_ckpt}...")
    ode_state = torch.load(ode_ckpt, map_location=device, weights_only=False)
    ode_dim = infer_hidden_dim(ode_state) or config["model"]["hidden_dim"]
    local_cfg = copy.deepcopy(config)
    local_cfg["model"]["hidden_dim"] = ode_dim
    ode_model = DeepM3Model(local_cfg, n_items=dataset.n_items, solver=args.ode_solver).to(device)
    ode_model.load_state_dict(ode_state, strict=True)
    ode_model.eval()

    results = []

    print(" Running analysis...")
    with torch.no_grad():
        for idx in tqdm(range(len(dataset))):
            sample = dataset[idx]
            x = sample["x"].unsqueeze(0).to(device)
            t = sample["t"].unsqueeze(0).to(device)
            dt = sample["dt"].unsqueeze(0).to(device) if "dt" in sample else None
            pos = int(sample["pos"].item())

            # Compute CV of inter-event gaps
            if dt is None:
                dt_local = torch.zeros_like(t)
                dt_local[:, 1:] = t[:, 1:] - t[:, :-1]
            else:
                dt_local = dt
            dt_valid = dt_local[:, 1:] if dt_local.size(1) > 1 else dt_local
            t_std = torch.std(dt_valid).item()
            t_mean = torch.mean(dt_valid).item()
            cv = t_std / (t_mean + 1e-6)

            # Fixed negatives
            cands = np.concatenate(([pos], neg_pool[idx]))
            cand_t = torch.tensor(cands, dtype=torch.long, device=device)

            # Baseline
            u_base = base_model(x)
            i_base = base_model.get_item_embedding(cand_t)
            s_base = torch.matmul(u_base, i_base.t())
            n_base = compute_ndcg(s_base, k=args.topk)

            # ODE
            u_ode = ode_model(x, t, dt=dt)
            i_ode = ode_model.get_item_embedding(cand_t)
            s_ode = torch.matmul(u_ode, i_ode.t())
            n_ode = compute_ndcg(s_ode, k=args.topk)

            results.append({
                "cv": cv,
                "ndcg_base": n_base,
                "ndcg_ode": n_ode,
            })

    df = pd.DataFrame(results)
    os.makedirs(args.output_dir, exist_ok=True)

    # Group by CV tertiles
    df["bucket_cv"] = pd.qcut(
        df["cv"], q=3,
        labels=["Regular (Low CV)", "Normal", "Irregular (High CV)"],
        duplicates="drop",
    )

    print("\n Analysis by Coefficient of Variation (CV):")
    summary = df.groupby("bucket_cv", observed=False)[["ndcg_base", "ndcg_ode"]].mean()
    summary["lift_pct"] = (
        (summary["ndcg_ode"] - summary["ndcg_base"]) / (summary["ndcg_base"] + 1e-12) * 100
    )
    print(summary)

    # Per-group significance
    group_sig = []
    for group_name in ["Regular (Low CV)", "Normal", "Irregular (High CV)"]:
        g = df[df["bucket_cv"] == group_name]
        if len(g) > 1:
            t_g, p_g = paired_t_test(g["ndcg_ode"], g["ndcg_base"])
            group_sig.append({
                "group": group_name,
                "n": len(g),
                "ndcg_base_mean": g["ndcg_base"].mean(),
                "ndcg_ode_mean": g["ndcg_ode"].mean(),
                "lift_pct": (g["ndcg_ode"].mean() - g["ndcg_base"].mean()) / (g["ndcg_base"].mean() + 1e-12) * 100,
                "t_stat": t_g,
                "p_value": p_g,
            })
    group_sig_df = pd.DataFrame(group_sig)

    # Highlight irregular group
    irr_group = df[df["bucket_cv"] == "Irregular (High CV)"]
    base_mean = irr_group["ndcg_base"].mean()
    ode_mean = irr_group["ndcg_ode"].mean()
    lift = (ode_mean - base_mean) / (base_mean + 1e-12) * 100
    print(f"\n Lift in Irregular Group: +{lift:.2f}%")

    if not group_sig_df.empty:
        print("\nPer-group significance:")
        print(group_sig_df.to_string(index=False))

    # Save CSVs
    irr_path = os.path.join(args.output_dir, "irregularity_analysis.csv")
    summary.reset_index().to_csv(irr_path, index=False)

    irr_sig_path = os.path.join(args.output_dir, "irregularity_significance.csv")
    group_sig_df.to_csv(irr_sig_path, index=False)

    print(f"\nSaved to: {irr_path}, {irr_sig_path}")


if __name__ == "__main__":
    analyze_metrics()
