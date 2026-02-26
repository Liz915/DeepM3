#!/usr/bin/env python3
"""
Model Efficiency Benchmark — measures inference latency, parameter count,
and recommendation quality for all model variants.

Outputs results/ml1m/efficiency.csv for use in the paper's efficiency table.
"""
import argparse
import copy
import glob
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

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


def infer_hidden_dim(state_dict):
    w = state_dict.get("item_emb.weight")
    if w is not None and getattr(w, "ndim", 0) == 2:
        return int(w.shape[1])
    gru_w = state_dict.get("gru.weight_ih_l0")
    if gru_w is not None and getattr(gru_w, "ndim", 0) == 2:
        return int(gru_w.shape[0] // 3)
    return None


def compute_hr_ndcg(scores, k=10):
    _, indices = torch.topk(scores, k, dim=-1)
    row = indices[0]
    if 0 in row:
        rank = (row == 0).nonzero(as_tuple=True)[0].item()
        return 1.0, 1.0 / np.log2(rank + 2)
    return 0.0, 0.0


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


def load_model(model_type, solver, ckpt_path, n_items, config, device):
    """Load a model checkpoint. Returns (model, params_count) or None."""
    if not os.path.exists(ckpt_path):
        return None

    state_dict = torch.load(ckpt_path, map_location=device, weights_only=False)
    hidden_dim = infer_hidden_dim(state_dict)
    if hidden_dim is None:
        hidden_dim = config["model"]["hidden_dim"]

    local_cfg = copy.deepcopy(config)
    local_cfg["model"]["hidden_dim"] = hidden_dim

    if model_type == "gru":
        model = GRUBaseline(n_items=n_items, hidden_dim=hidden_dim)
    elif model_type == "deepm3":
        model = DeepM3Model(local_cfg, n_items=n_items, solver=solver)
    elif model_type == "sasrec":
        max_len = local_cfg["data"].get("max_seq_len", 20)
        model = SASRec(n_items=n_items, hidden_dim=hidden_dim, max_len=max_len)
    elif model_type == "tisasrec":
        max_len = local_cfg["data"].get("max_seq_len", 20)
        model = TiSASRec(n_items=n_items, hidden_dim=hidden_dim, max_len=max_len)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return model, params


def needs_time(model_type, solver):
    """Whether forward pass requires time inputs."""
    if model_type == "gru":
        return False
    if model_type == "sasrec":
        return False
    if model_type == "tisasrec":
        return True
    if model_type == "deepm3":
        return solver != "baseline"
    return False


def pick_tuned_checkpoint(model_name, output_dir, dataset):
    """
    Pick the tuned best checkpoint for SASRec/TiSASRec if available.
    Priority:
      1) baseline_tuning_best.csv under output_dir/baseline_tuning
      2) first checkpoint match under checkpoints/tuning (fallback)
    """
    best_csv = os.path.join(output_dir, "baseline_tuning", "baseline_tuning_best.csv")
    if os.path.exists(best_csv):
        try:
            df = pd.read_csv(best_csv)
            row = df[df["model"] == model_name]
            if not row.empty and "checkpoint" in row.columns:
                ckpt = str(row.iloc[0]["checkpoint"])
                if os.path.exists(ckpt):
                    return ckpt
        except Exception:
            pass

    pattern = os.path.join("checkpoints", "tuning", f"tune_{dataset}_{model_name}_*.pth")
    ckpts = sorted(glob.glob(pattern))
    return ckpts[0] if ckpts else None


def benchmark_model(model, model_type, solver, dataset, neg_pool, device,
                    topk=10, warmup=50):
    """Run full evaluation with latency measurement."""
    use_time = needs_time(model_type, solver)
    hrs, ndcgs, latencies = [], [], []

    with torch.no_grad():
        desc = f"Bench {model_type} ({solver})"
        for idx in tqdm(range(len(dataset)), desc=desc, leave=False):
            sample = dataset[idx]
            x = sample["x"].unsqueeze(0).to(device)
            t = sample["t"].unsqueeze(0).to(device)
            dt = sample["dt"].unsqueeze(0).to(device) if "dt" in sample else None
            pos = int(sample["pos"].item())

            cands = np.concatenate(([pos], neg_pool[idx]))
            cand_t = torch.tensor(cands, dtype=torch.long, device=device)

            st = time.time()
            if model_type == "deepm3" and solver != "baseline":
                user_emb = model(x, t, dt=dt)
            elif use_time:
                user_emb = model(x, t)
            else:
                user_emb = model(x)
            # Synchronize for accurate timing
            if device.type == "cuda":
                torch.cuda.synchronize()
            elif device.type == "mps":
                torch.mps.synchronize()
            elapsed = (time.time() - st) * 1000.0

            cand_emb = model.get_item_embedding(cand_t)
            scores = torch.matmul(user_emb, cand_emb.t())
            h, n = compute_hr_ndcg(scores, k=topk)
            hrs.append(h)
            ndcgs.append(n)

            # Skip warmup samples for latency stats
            if idx >= warmup:
                latencies.append(elapsed)

    return {
        "hr@10": float(np.mean(hrs)),
        "ndcg@10": float(np.mean(ndcgs)),
        "latency_mean_ms": float(np.mean(latencies)) if latencies else 0.0,
        "latency_std_ms": float(np.std(latencies)) if latencies else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Model efficiency benchmark")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_neg", type=int, default=100)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--output_dir", default="results/ml1m")
    parser.add_argument("--dataset", type=str, default="ml1m", choices=["ml1m", "amazon"])
    parser.add_argument("--data_dir", type=str, default="")
    parser.add_argument("--hidden_dim", type=int, default=128)
    args = parser.parse_args()

    device = pick_device(args.device)
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    data_dir = args.data_dir or ("data/amazon" if args.dataset == "amazon" else "data")
    dataset = MovieLensDataset(mode="test", data_dir=data_dir)
    print(f"Test samples: {len(dataset)} | device={device}")
    neg_pool = build_neg_pool(dataset, args.num_neg, args.seed)

    ds = args.dataset
    dim = args.hidden_dim

    def get_ckpt(tag):
        if ds == "ml1m":
            legacy = {
                "baseline": "checkpoints/model_gru_baseline.pth",
                "none": "checkpoints/model_ode_none.pth",
                "euler": "checkpoints/model_ode_euler.pth",
                "rk4": "checkpoints/model_ode_rk4.pth",
            }
            if os.path.exists(legacy[tag]):
                return legacy[tag]
        if tag == "baseline":
            return f"checkpoints/{ds}_gru_d{dim}.pth"
        return f"checkpoints/{ds}_deepm3_{tag}_d{dim}.pth"

    # Define all models to benchmark
    # (display_name, model_type, solver, checkpoint_path)
    candidates = [
        ("Baseline (GRU)", "gru", "baseline", get_ckpt("baseline")),
        ("DeepM3 (none)", "deepm3", "none", get_ckpt("none")),
        ("DeepM3 (Euler)", "deepm3", "euler", get_ckpt("euler")),
        ("DeepM3 (RK4)", "deepm3", "rk4", get_ckpt("rk4")),
    ]

    # Also look for tuned baseline checkpoints (best SASRec / TiSASRec)
    for model_name in ["sasrec", "tisasrec"]:
        ckpt = pick_tuned_checkpoint(
            model_name=model_name,
            output_dir=args.output_dir,
            dataset=args.dataset,
        )
        if ckpt:
            candidates.append((
                f"{model_name.upper()}",
                model_name,
                "n/a",
                ckpt,
            ))

    rows = []
    for display_name, model_type, solver, ckpt in candidates:
        result = load_model(model_type, solver, ckpt, dataset.n_items, cfg, device)
        if result is None:
            print(f"  Skip {display_name}: {ckpt} not found")
            continue

        model, param_count = result
        print(f"  Benchmarking {display_name} ({param_count:,} params)...", end=" ", flush=True)
        metrics = benchmark_model(
            model, model_type, solver, dataset, neg_pool, device, topk=args.topk
        )
        print(
            f"HR@10={metrics['hr@10']:.4f} NDCG@10={metrics['ndcg@10']:.4f} "
            f"Lat={metrics['latency_mean_ms']:.2f}±{metrics['latency_std_ms']:.2f}ms"
        )

        rows.append({
            "model": display_name,
            "type": model_type,
            "solver": solver,
            "params": param_count,
            "hr@10": metrics["hr@10"],
            "ndcg@10": metrics["ndcg@10"],
            "latency_mean_ms": metrics["latency_mean_ms"],
            "latency_std_ms": metrics["latency_std_ms"],
        })

    if not rows:
        print("No models found!")
        return

    df = pd.DataFrame(rows)
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "efficiency.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
