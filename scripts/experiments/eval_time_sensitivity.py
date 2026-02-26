#!/usr/bin/env python3
"""
E6: Time Sensitivity - Original vs Shuffled timestamps.
Proves the model exploits temporal structure, not just sequence order.
Outputs CSV + prints results.
"""
import argparse
import copy
import os
import sys
import torch
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm

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


def eval_model(model, dataset, device, solver, shuffle_time=False, num_neg=100, seed=42, topk=10):
    rng = np.random.default_rng(seed)
    torch_gen = torch.Generator(device="cpu")
    torch_gen.manual_seed(seed)
    ndcg_list = []
    model.eval()

    with torch.no_grad():
        desc = f"Eval {solver} (shuf={shuffle_time})"
        for idx in tqdm(range(len(dataset)), desc=desc, leave=False):
            sample = dataset[idx]
            x = sample["x"].unsqueeze(0).to(device)
            t = sample["t"].unsqueeze(0).to(device)
            dt = sample["dt"].unsqueeze(0).to(device) if "dt" in sample else None
            pos = int(sample["pos"].item())

            # Fixed negatives for reproducibility
            blocked = {pos}
            blocked.update(int(v) for v in sample["x"] if int(v) > 0)
            negs = []
            while len(negs) < num_neg:
                c = int(rng.integers(1, dataset.n_items))
                if c not in blocked:
                    negs.append(c)
            cand = torch.tensor([pos] + negs, dtype=torch.long, device=device)

            if shuffle_time:
                # Shuffle inter-event gaps while keeping timestamps monotonic.
                if dt is None:
                    dt = torch.zeros_like(t)
                    dt[:, 1:] = t[:, 1:] - t[:, :-1]

                if dt.size(1) > 1:
                    gap_perm = torch.randperm(dt.size(1) - 1, generator=torch_gen).to(dt.device)
                    shuffled_tail = dt[:, 1:][:, gap_perm]
                    dt = torch.cat([dt[:, :1], shuffled_tail], dim=1)
                    t = torch.cumsum(dt, dim=1)

            if solver == "baseline":
                user_emb = model(x)
            else:
                user_emb = model(x, t, dt=dt)

            cand_emb = model.get_item_embedding(cand)
            scores = torch.matmul(user_emb, cand_emb.t())
            ndcg_list.append(compute_ndcg(scores, k=topk))

    return np.mean(ndcg_list)


def main():
    parser = argparse.ArgumentParser(description="Time sensitivity analysis")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_neg", type=int, default=100)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--output_dir", default="results/ml1m")
    parser.add_argument("--dataset", type=str, default="ml1m", choices=["ml1m", "amazon"])
    parser.add_argument("--data_dir", type=str, default="")
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--baseline_ckpt", type=str, default="")
    parser.add_argument("--euler_ckpt", type=str, default="")
    parser.add_argument("--rk4_ckpt", type=str, default="")
    args = parser.parse_args()

    device = pick_device(args.device)
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    data_dir = args.data_dir or ("data/amazon" if args.dataset == "amazon" else "data")
    dataset = MovieLensDataset(mode="test", data_dir=data_dir)
    print(f"Test samples: {len(dataset)} | device={device}")

    results = []

    ds = args.dataset
    dim = args.hidden_dim

    def get_ckpt(tag):
        if ds == "ml1m":
            legacy = {
                "baseline": "checkpoints/model_gru_baseline.pth",
                "euler": "checkpoints/model_ode_euler.pth",
                "rk4": "checkpoints/model_ode_rk4.pth",
            }
            if os.path.exists(legacy[tag]):
                return legacy[tag]
        if tag == "baseline":
            return f"checkpoints/{ds}_gru_d{dim}.pth"
        return f"checkpoints/{ds}_deepm3_{tag}_d{dim}.pth"

    ckpt_map = {
        "baseline": args.baseline_ckpt.strip() or get_ckpt("baseline"),
        "euler": args.euler_ckpt.strip() or get_ckpt("euler"),
        "rk4": args.rk4_ckpt.strip() or get_ckpt("rk4"),
    }

    # Test ODE variants (euler & rk4): they should degrade when timestamps are shuffled.
    for solver, ckpt_path in [("euler", ckpt_map["euler"]), ("rk4", ckpt_map["rk4"])]:
        if not os.path.exists(ckpt_path):
            print(f"Skip {solver}: {ckpt_path} not found")
            continue

        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        hidden_dim = infer_hidden_dim(state) or cfg["model"]["hidden_dim"]
        local_cfg = copy.deepcopy(cfg)
        local_cfg["model"]["hidden_dim"] = hidden_dim
        model = DeepM3Model(local_cfg, n_items=dataset.n_items, solver=solver).to(device)
        model.load_state_dict(state, strict=True)

        ndcg_orig = eval_model(
            model, dataset, device, solver,
            shuffle_time=False, num_neg=args.num_neg, seed=args.seed, topk=args.topk
        )
        ndcg_shuf = eval_model(
            model, dataset, device, solver,
            shuffle_time=True, num_neg=args.num_neg, seed=args.seed, topk=args.topk
        )
        drop = (ndcg_orig - ndcg_shuf) / (ndcg_orig + 1e-12) * 100

        results.append({
            "model": f"DeepM3 ({solver})",
            "ndcg_original": ndcg_orig,
            "ndcg_shuffled": ndcg_shuf,
            "drop_pct": drop,
        })
        print(f"{solver}: orig={ndcg_orig:.4f} shuf={ndcg_shuf:.4f} drop={drop:.2f}%")

    # GRU baseline should NOT degrade (it ignores timestamps).
    ckpt_b = ckpt_map["baseline"]
    if os.path.exists(ckpt_b):
        state = torch.load(ckpt_b, map_location=device, weights_only=False)
        hidden_dim = infer_hidden_dim(state) or cfg["model"]["hidden_dim"]
        model_b = GRUBaseline(n_items=dataset.n_items, hidden_dim=hidden_dim).to(device)
        model_b.load_state_dict(state, strict=True)
        ndcg_b1 = eval_model(
            model_b, dataset, device, "baseline",
            shuffle_time=False, num_neg=args.num_neg, seed=args.seed, topk=args.topk
        )
        ndcg_b2 = eval_model(
            model_b, dataset, device, "baseline",
            shuffle_time=True, num_neg=args.num_neg, seed=args.seed, topk=args.topk
        )
        drop_b = (ndcg_b1 - ndcg_b2) / (ndcg_b1 + 1e-12) * 100
        results.append({
            "model": "Baseline (GRU)",
            "ndcg_original": ndcg_b1,
            "ndcg_shuffled": ndcg_b2,
            "drop_pct": drop_b,
        })
        print(f"baseline: orig={ndcg_b1:.4f} shuf={ndcg_b2:.4f} drop={drop_b:.2f}%")

    df = pd.DataFrame(results)
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "time_sensitivity.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
