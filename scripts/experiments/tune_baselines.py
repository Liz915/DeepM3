#!/usr/bin/env python3
"""
Grid-search tuner for SASRec/TiSASRec baselines.

Runs train_unified.py for each config, evaluates on fixed negatives, and
exports full + best-per-model CSVs for reproducible baseline fairness.
"""
import argparse
import copy
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.append(os.getcwd())
from src.data.dataset import MovieLensDataset
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


def parse_csv_list(raw, cast_fn):
    return [cast_fn(x.strip()) for x in raw.split(",") if x.strip()]


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


def compute_hr_ndcg(scores, k=10):
    _, indices = torch.topk(scores, k, dim=-1)
    row = indices[0]
    if 0 in row:
        rank = (row == 0).nonzero(as_tuple=True)[0].item()
        return 1.0, 1.0 / np.log2(rank + 2)
    return 0.0, 0.0


def load_baseline(model_name, ckpt_path, n_items, hidden_dim, config, device):
    cfg = copy.deepcopy(config)
    cfg["model"]["hidden_dim"] = hidden_dim
    max_len = cfg["data"].get("max_seq_len", 20)

    if model_name == "sasrec":
        model = SASRec(n_items=n_items, hidden_dim=hidden_dim, max_len=max_len)
    elif model_name == "tisasrec":
        model = TiSASRec(n_items=n_items, hidden_dim=hidden_dim, max_len=max_len)
    else:
        raise ValueError(f"Unsupported model for tuning: {model_name}")

    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


def evaluate_baseline(model, model_name, dataset, neg_pool, device, topk=10):
    hr, ndcg, latency_ms = [], [], []
    use_time = model_name == "tisasrec"

    with torch.no_grad():
        for idx in range(len(dataset)):
            sample = dataset[idx]
            x = sample["x"].unsqueeze(0).to(device)
            t = sample["t"].unsqueeze(0).to(device)
            pos = int(sample["pos"].item())

            cands = np.concatenate(([pos], neg_pool[idx]))
            cand_t = torch.tensor(cands, dtype=torch.long, device=device)

            st = time.time()
            if use_time:
                user_emb = model(x, t)
            else:
                user_emb = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elif device.type == "mps":
                torch.mps.synchronize()
            latency_ms.append((time.time() - st) * 1000.0)

            cand_emb = model.get_item_embedding(cand_t)
            scores = torch.matmul(user_emb, cand_emb.t())
            h, n = compute_hr_ndcg(scores, k=topk)
            hr.append(h)
            ndcg.append(n)

    return {
        "hr@10": float(np.mean(hr)),
        "ndcg@10": float(np.mean(ndcg)),
        "latency_ms": float(np.mean(latency_ms)),
    }


def run_train(train_cmd):
    print(" ".join(train_cmd))
    subprocess.run(train_cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, default="sasrec,tisasrec")
    parser.add_argument("--lrs", type=str, default="1e-4,3e-4,5e-4")
    parser.add_argument("--epochs_list", type=str, default="50,100,150")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dataset", type=str, default="ml1m", choices=["ml1m", "amazon"])
    parser.add_argument("--data_dir", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_neg", type=int, default=100)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--train_script", type=str, default="scripts/train/train_unified.py")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    models = parse_csv_list(args.models, str)
    lrs = parse_csv_list(args.lrs, float)
    epochs_list = parse_csv_list(args.epochs_list, int)

    data_dir = args.data_dir or ("data/amazon" if args.dataset == "amazon" else "data")
    out_dir = Path(args.output_dir or f"results/{args.dataset}/baseline_tuning")
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path("checkpoints/tuning")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = pick_device(args.device)
    test_dataset = MovieLensDataset(mode="test", data_dir=data_dir)
    print(f"Test samples: {len(test_dataset)} | n_items={test_dataset.n_items} | device={device}")
    neg_pool = build_neg_pool(test_dataset, args.num_neg, args.seed)

    rows = []
    total_runs = len(models) * len(lrs) * len(epochs_list)
    run_id = 0

    for model_name in models:
        for lr in lrs:
            for epochs in epochs_list:
                run_id += 1
                lr_tag = str(lr).replace(".", "p")
                ckpt_name = (
                    f"tune_{args.dataset}_{model_name}_d{args.hidden_dim}_"
                    f"lr{lr_tag}_e{epochs}_s{args.seed}.pth"
                )
                ckpt_rel = Path("tuning") / ckpt_name
                ckpt_path = Path("checkpoints") / ckpt_rel

                print("\n" + "-" * 72)
                print(
                    f"[{run_id}/{total_runs}] model={model_name} "
                    f"lr={lr} epochs={epochs} dim={args.hidden_dim}"
                )

                cmd = [
                    "python",
                    args.train_script,
                    "--model", model_name,
                    "--epochs", str(epochs),
                    "--seed", str(args.seed),
                    "--hidden_dim", str(args.hidden_dim),
                    "--dataset", args.dataset,
                    "--save_name", str(ckpt_rel),
                    "--lr", str(lr),
                    "--device", args.device,
                    "--config", args.config,
                ]
                if args.data_dir:
                    cmd.extend(["--data_dir", args.data_dir])
                run_train(cmd)

                model = load_baseline(
                    model_name=model_name,
                    ckpt_path=str(ckpt_path),
                    n_items=test_dataset.n_items,
                    hidden_dim=args.hidden_dim,
                    config=cfg,
                    device=device,
                )
                metrics = evaluate_baseline(
                    model=model,
                    model_name=model_name,
                    dataset=test_dataset,
                    neg_pool=neg_pool,
                    device=device,
                    topk=args.topk,
                )

                row = {
                    "model": model_name,
                    "lr": lr,
                    "epochs": epochs,
                    "hidden_dim": args.hidden_dim,
                    "seed": args.seed,
                    "checkpoint": str(ckpt_path),
                    **metrics,
                }
                rows.append(row)
                print(
                    f"Result: HR@10={metrics['hr@10']:.4f} "
                    f"NDCG@10={metrics['ndcg@10']:.4f} "
                    f"Latency={metrics['latency_ms']:.2f}ms"
                )

    if not rows:
        print("No runs executed.")
        return

    df = pd.DataFrame(rows).sort_values(["model", "ndcg@10"], ascending=[True, False])
    best = df.groupby("model", as_index=False).head(1).reset_index(drop=True)

    full_path = out_dir / "baseline_tuning_all.csv"
    best_path = out_dir / "baseline_tuning_best.csv"
    df.to_csv(full_path, index=False)
    best.to_csv(best_path, index=False)

    print("\n" + "=" * 72)
    print("Best settings per model:")
    print(best[["model", "lr", "epochs", "hidden_dim", "hr@10", "ndcg@10", "latency_ms"]].to_string(index=False))
    print(f"\nSaved full results: {full_path}")
    print(f"Saved best results: {best_path}")


if __name__ == "__main__":
    main()
