#!/usr/bin/env python3
"""
Comprehensive evaluation for ALL models on a given dataset.
Generates table1_main.csv with all 5 metrics + significance tests.
Supports: GRU, SASRec, TiSASRec, DeepM3 (none/euler/rk4).
"""
import argparse
import copy
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.append(os.getcwd())
from src.data.dataset import MovieLensDataset
from src.dynamics.modeling import DeepM3Model
from src.dynamics.gru_baseline import GRUBaseline
from src.dynamics.baselines import SASRec, TiSASRec


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


def compute_rank_metrics(scores, k=10):
    n_items = scores.size(-1)
    _, indices = torch.topk(scores, k, dim=-1)
    row = indices[0]

    hr, ndcg, mrr = 0.0, 0.0, 0.0
    if 0 in row:
        rank = (row == 0).nonzero(as_tuple=True)[0].item()
        hr = 1.0
        ndcg = 1.0 / np.log2(rank + 2)
        mrr = 1.0 / (rank + 1)

    _, top5 = torch.topk(scores, min(5, n_items), dim=-1)
    recall5 = 1.0 if 0 in top5[0] else 0.0

    pos_score = scores[0, 0].item()
    neg_scores = scores[0, 1:]
    auc = float((neg_scores < pos_score).float().mean().item())

    return {"hr": hr, "ndcg": ndcg, "mrr": mrr, "recall5": recall5, "auc": auc}


def build_neg_pool(dataset, num_neg, seed):
    rng = np.random.default_rng(seed)
    neg_pool = []
    for sample in dataset.samples:
        blocked = {int(sample["y"])}
        blocked.update(int(v) for v in sample["x"] if int(v) > 0)
        negs = []
        while len(negs) < num_neg:
            c = int(rng.integers(1, dataset.n_items))
            if c not in blocked:
                negs.append(c)
        neg_pool.append(np.asarray(negs, dtype=np.int64))
    return neg_pool


def load_model(model_name, solver, ckpt_path, n_items, hidden_dim, config, device):
    if not os.path.exists(ckpt_path):
        return None

    state = torch.load(ckpt_path, map_location=device, weights_only=False)

    cfg = copy.deepcopy(config)
    cfg["model"]["hidden_dim"] = hidden_dim

    if model_name == "gru":
        model = GRUBaseline(n_items=n_items, hidden_dim=hidden_dim)
    elif model_name == "sasrec":
        model = SASRec(n_items=n_items, hidden_dim=hidden_dim,
                       max_len=cfg["data"].get("max_seq_len", 20))
    elif model_name == "tisasrec":
        model = TiSASRec(n_items=n_items, hidden_dim=hidden_dim,
                         max_len=cfg["data"].get("max_seq_len", 20))
    elif model_name == "deepm3":
        model = DeepM3Model(cfg, n_items=n_items, solver=solver)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


def uses_time(model_name):
    return model_name in ("tisasrec", "deepm3")


def evaluate_model(model, model_name, dataset, neg_pool, device, topk=10):
    metrics = {"hr": [], "ndcg": [], "mrr": [], "recall5": [], "auc": [], "latency": []}

    with torch.no_grad():
        for idx in range(len(dataset)):
            sample = dataset[idx]
            x = sample["x"].unsqueeze(0).to(device)
            t = sample["t"].unsqueeze(0).to(device)
            dt = sample["dt"].unsqueeze(0).to(device) if "dt" in sample else None
            pos = int(sample["pos"].item())

            cands = np.concatenate(([pos], neg_pool[idx]))
            cands_t = torch.tensor(cands, dtype=torch.long, device=device)

            start = time.time()
            if model_name == "deepm3":
                user_emb = model(x, t, dt=dt)
            elif uses_time(model_name):
                user_emb = model(x, t)
            else:
                user_emb = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elif device.type == "mps":
                torch.mps.synchronize()
            elapsed = (time.time() - start) * 1000.0

            cand_emb = model.get_item_embedding(cands_t)
            scores = torch.matmul(user_emb, cand_emb.t())
            m = compute_rank_metrics(scores, k=topk)
            for k_name in ("hr", "ndcg", "mrr", "recall5", "auc"):
                metrics[k_name].append(m[k_name])
            metrics["latency"].append(elapsed)

    return {k: np.asarray(v, dtype=np.float64) for k, v in metrics.items()}


def paired_t_test(x, y):
    try:
        from scipy import stats
        t, p = stats.ttest_rel(x, y)
        return float(t), float(p)
    except Exception:
        diff = x - y
        n = diff.size
        if n < 2:
            return float("nan"), float("nan")
        d_mean = float(np.mean(diff))
        d_std = float(np.std(diff, ddof=1))
        if d_std == 0:
            return float("inf"), 0.0
        t = d_mean / (d_std / math.sqrt(n))
        p = math.erfc(abs(t) / math.sqrt(2.0))
        return t, p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ml1m")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_neg", type=int, default=100)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--data_dir", type=str, default="")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    device = pick_device(args.device)
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_dir = args.data_dir or ("data/amazon" if args.dataset == "amazon" else "data")
    dataset = MovieLensDataset(mode="test", data_dir=data_dir)
    print(f"Test: {len(dataset)} samples | n_items={dataset.n_items} | device={device}")

    neg_pool = build_neg_pool(dataset, args.num_neg, args.seed)

    dim = args.hidden_dim
    ds = args.dataset

    # Model specs: (display_name, model_type, solver, checkpoint_path)
    model_specs = [
        ("GRU", "gru", "baseline", f"checkpoints/{ds}_gru_d{dim}.pth"),
        ("SASRec", "sasrec", "none", f"checkpoints/{ds}_sasrec_d{dim}.pth"),
        ("TiSASRec", "tisasrec", "none", f"checkpoints/{ds}_tisasrec_d{dim}.pth"),
        ("DeepM3 (none)", "deepm3", "none", f"checkpoints/{ds}_deepm3_none_d{dim}.pth"),
        ("DeepM3 (Euler)", "deepm3", "euler", f"checkpoints/{ds}_deepm3_euler_d{dim}.pth"),
        ("DeepM3 (RK4)", "deepm3", "rk4", f"checkpoints/{ds}_deepm3_rk4_d{dim}.pth"),
    ]

    rows = []
    all_ndcg = {}

    for display, model_type, solver, ckpt in model_specs:
        if not os.path.exists(ckpt):
            print(f"Skip {display}: {ckpt} not found")
            continue

        print(f"Evaluating {display} from {ckpt}...")
        model = load_model(model_type, solver, ckpt, dataset.n_items, dim, cfg, device)
        if model is None:
            continue

        metrics = evaluate_model(model, model_type, dataset, neg_pool, device, args.topk)
        all_ndcg[display] = metrics["ndcg"]

        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

        rows.append({
            "method": display,
            "params": param_count,
            "latency_ms": float(np.mean(metrics["latency"])),
            "hr@10": float(np.mean(metrics["hr"])),
            "ndcg@10": float(np.mean(metrics["ndcg"])),
            "mrr": float(np.mean(metrics["mrr"])),
            "recall@5": float(np.mean(metrics["recall5"])),
            "auc": float(np.mean(metrics["auc"])),
        })
        print(f"  HR@10={rows[-1]['hr@10']:.4f} NDCG@10={rows[-1]['ndcg@10']:.4f} "
              f"MRR={rows[-1]['mrr']:.4f} R@5={rows[-1]['recall@5']:.4f} AUC={rows[-1]['auc']:.4f}")

    if not rows:
        print("No models found! Train them first.")
        return

    df = pd.DataFrame(rows)

    # Compute lifts vs GRU baseline
    if "GRU" in df["method"].values:
        base = df[df["method"] == "GRU"].iloc[0]
        for metric in ["hr@10", "ndcg@10", "mrr", "recall@5", "auc"]:
            df[f"{metric}_lift%"] = (
                (df[metric] - base[metric]) / (base[metric] + 1e-12) * 100.0
            )

    out_dir = Path(f"results/{args.dataset}")
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "full_comparison.csv", index=False)
    print(f"\nFull comparison saved to {out_dir}/full_comparison.csv")

    # Significance tests: each model vs GRU
    if "GRU" in all_ndcg:
        sig_rows = []
        gru_ndcg = all_ndcg["GRU"]
        for name, ndcg_arr in all_ndcg.items():
            if name == "GRU":
                continue
            t_stat, p_val = paired_t_test(ndcg_arr, gru_ndcg)
            sig_rows.append({
                "comparison": f"{name} vs GRU",
                "t_statistic": t_stat,
                "p_value": p_val,
                "significant_005": p_val < 0.05 if not np.isnan(p_val) else False,
            })
        sig_df = pd.DataFrame(sig_rows)
        sig_df.to_csv(out_dir / "significance_all.csv", index=False)
        print(f"\nSignificance tests:")
        print(sig_df.to_string(index=False))

    print(f"\nDone! Results in {out_dir}/")


if __name__ == "__main__":
    main()
