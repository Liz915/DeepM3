#!/usr/bin/env python3
"""
Unified training script for ALL models (GRU, SASRec, TiSASRec, DeepM3 variants).
Supports both ML-1M and Amazon-Books datasets.
"""
import argparse
import os
import sys
import time
import copy

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml

sys.path.append(os.getcwd())
from src.data.dataset import MovieLensDataset
from src.dynamics.modeling import DeepM3Model
from src.dynamics.gru_baseline import GRUBaseline
from src.dynamics.baselines import SASRec, TiSASRec
from src.utils.seeder import set_seed
import numpy as np

def compute_ndcg_per_user(scores, k=10):
    _, top_indices = torch.topk(scores, k=k, dim=-1)
    pos_in_topk = (top_indices == 0).float()
    ranks = torch.arange(1, k + 1, device=scores.device).float()
    weights = 1.0 / torch.log2(ranks + 1)
    ndcg = (pos_in_topk * weights).sum(dim=-1).item()
    return ndcg

def evaluate_val(model, dataset, device, solver="euler", args=None, val_indices=None):
    model.eval()
    ndcg_list = []
    # Build a simple fixed negative pool for val to be fast and reproducible.
    rng = np.random.default_rng(42)
    num_neg = args.val_num_neg if args is not None else 100
    n_items = getattr(dataset, "n_items", None)
    if n_items is None and hasattr(dataset, "dataset"):
        n_items = getattr(dataset.dataset, "n_items", None)
    if n_items is None:
        raise ValueError("Cannot infer n_items for validation dataset.")

    index_iter = val_indices if val_indices is not None else range(len(dataset))
    with torch.no_grad():
        for idx in index_iter:
            sample = dataset[idx]
            x = sample["x"].unsqueeze(0).to(device)
            t = sample["t"].unsqueeze(0).to(device)
            dt = sample["dt"].unsqueeze(0).to(device) if "dt" in sample else None
            pos = int(sample["pos"].item())

            blocked = {pos}
            blocked.update(int(v) for v in sample["x"].tolist() if int(v) > 0)
            negs = []
            while len(negs) < num_neg:
                c = int(rng.integers(1, n_items))
                if c not in blocked:
                    negs.append(c)
            cands = np.asarray([pos] + negs, dtype=np.int64)
            cand_t = torch.tensor(cands, dtype=torch.long, device=device)
            
            if args.model == "deepm3":
                t_in, dt_in = t, dt
                if hasattr(args, "time_ablation") and args.time_ablation != "full":
                    seq_len = t.size(1)
                    t_seq = torch.arange(1, seq_len+1, dtype=torch.float32, device=t.device).unsqueeze(0).expand_as(t)
                    dt_ones = torch.ones_like(t)
                    if args.time_ablation == "none":
                        t_in, dt_in = t_seq, dt_ones
                    elif args.time_ablation == "t_only":
                        dt_in = dt_ones
                    elif args.time_ablation == "dt_only":
                        t_in = t_seq
                u = model(x, t_in, dt=dt_in)
            elif model_needs_time(args.model):
                u = model(x, t)
            else:
                u = model(x)
                
            i_emb = model.get_item_embedding(cand_t)
            s = torch.matmul(u, i_emb.t())
            ndcg_list.append(compute_ndcg_per_user(s, k=10))
    return np.mean(ndcg_list)


def bpr_loss(user_emb, pos_item_emb, neg_item_emb):
    pos_score = torch.sum(user_emb * pos_item_emb, dim=-1)
    neg_score = torch.sum(user_emb * neg_item_emb, dim=-1)
    loss = -torch.mean(torch.nn.functional.logsigmoid(pos_score - neg_score))
    return loss


def build_model(model_name, n_items, hidden_dim, config, solver="euler"):
    """Build model by name."""
    if model_name == "gru":
        return GRUBaseline(n_items=n_items, hidden_dim=hidden_dim)
    elif model_name == "sasrec":
        return SASRec(n_items=n_items, hidden_dim=hidden_dim,
                      max_len=config["data"].get("max_seq_len", 20))
    elif model_name == "tisasrec":
        return TiSASRec(n_items=n_items, hidden_dim=hidden_dim,
                        max_len=config["data"].get("max_seq_len", 20))
    elif model_name == "deepm3":
        cfg = copy.deepcopy(config)
        cfg["model"]["hidden_dim"] = hidden_dim
        return DeepM3Model(cfg, n_items=n_items, solver=solver)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def model_needs_time(model_name):
    """Whether the model uses timestamp input."""
    return model_name in ("tisasrec", "deepm3")


def augment_time_inputs(t, dt, jitter_std=0.0, drop_prob=0.0):
    """
    Time augmentation for DeepM3 training:
    - jitter on inter-event gaps
    - random gap dropout
    """
    if dt is None:
        dt = torch.zeros_like(t)
        dt[:, 1:] = t[:, 1:] - t[:, :-1]

    dt_aug = dt.clone()
    if dt_aug.size(1) <= 1:
        return t, dt_aug

    if jitter_std > 0:
        noise = torch.randn_like(dt_aug[:, 1:]) * jitter_std
        dt_aug[:, 1:] = torch.clamp(dt_aug[:, 1:] + noise, min=0.0)

    if drop_prob > 0:
        keep_mask = (torch.rand_like(dt_aug[:, 1:]) > drop_prob).float()
        dt_aug[:, 1:] = dt_aug[:, 1:] * keep_mask

    t_aug = torch.cumsum(dt_aug, dim=1)
    return t_aug, dt_aug


def train(args):
    set_seed(args.seed)
    
    # Device selection
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"Training {args.model} on {device} | dim={args.hidden_dim} | dataset={args.dataset}")
    if args.model == "deepm3" and (args.time_jitter_std > 0 or args.time_drop_prob > 0):
        print(
            f"Time augmentation enabled: jitter_std={args.time_jitter_std}, "
            f"drop_prob={args.time_drop_prob}"
        )

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    if args.hidden_dim > 0:
        config["model"]["hidden_dim"] = args.hidden_dim
    hidden_dim = config["model"]["hidden_dim"]

    data_dir = args.data_dir or ("data/amazon" if args.dataset == "amazon" else "data")

    # Dataset
    if args.dataset == "ml1m":
        train_dataset = MovieLensDataset(mode="train", data_dir=data_dir)
    elif args.dataset == "amazon":
        train_dataset = MovieLensDataset(mode="train", data_dir=data_dir)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=True,
        num_workers=0,
    )
    print(f"Train Samples: {len(train_dataset)} | n_items: {train_dataset.n_items}")

    model = build_model(
        args.model, train_dataset.n_items, hidden_dim, config, solver=args.solver
    ).to(device)
    
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {param_count:,}")

    # Model-specific LR: Transformers need lower LR
    default_lr = config["train"].get("learning_rate", 1e-3)
    if args.lr > 0:
        lr = args.lr
    elif args.model in ("sasrec", "tisasrec"):
        lr = 1e-4  # Transformers are unstable with lr=1e-3
    else:
        lr = default_lr
    print(f"Learning rate: {lr}")

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)

    # Validation split for early stopping:
    # prefer explicit ml1m_val.pkl; otherwise use a deterministic subset of train.
    val_indices = None
    val_path = os.path.join(data_dir, "ml1m_val.pkl")
    if os.path.exists(val_path):
        val_dataset = MovieLensDataset(mode="val", data_dir=data_dir)
        print(f"Val Samples: {len(val_dataset)} (explicit split)")
    else:
        val_dataset = train_dataset
        rng = np.random.default_rng(args.seed + 17)
        all_indices = np.arange(len(train_dataset))
        rng.shuffle(all_indices)
        val_count = int(len(train_dataset) * args.val_ratio)
        val_count = max(1, min(len(train_dataset), args.val_max_samples, max(256, val_count)))
        val_indices = all_indices[:val_count].tolist()
        print(
            f"Validation file missing. Using train subset for early stopping: "
            f"{val_count}/{len(train_dataset)} samples."
        )

    best_ndcg = -1.0
    patience_counter = 0
    patience_limit = 3

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        start = time.time()

        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()

            x = batch["x"].to(device)
            t = batch["t"].to(device)
            dt = batch["dt"].to(device) if "dt" in batch else None
            pos = batch["pos"].to(device)
            neg = batch["neg"].to(device)

            # Forward
            if args.model == "deepm3":
                t_in, dt_in = t, dt
                if args.time_jitter_std > 0 or args.time_drop_prob > 0:
                    t_in, dt_in = augment_time_inputs(
                        t=t,
                        dt=dt,
                        jitter_std=args.time_jitter_std,
                        drop_prob=args.time_drop_prob,
                    )
                # Ablation overrides overrides jitter/drop
                if hasattr(args, "time_ablation") and args.time_ablation != "full":
                    seq_len = t.size(1)
                    t_seq = torch.arange(1, seq_len+1, dtype=torch.float32, device=t.device).unsqueeze(0).expand_as(t)
                    dt_ones = torch.ones_like(t)
                    if args.time_ablation == "none":
                        t_in, dt_in = t_seq, dt_ones
                    elif args.time_ablation == "t_only":
                        dt_in = dt_ones
                    elif args.time_ablation == "dt_only":
                        t_in = t_seq
                
                user_emb = model(x, t_in, dt=dt_in)
            elif model_needs_time(args.model):
                user_emb = model(x, t)
            else:
                user_emb = model(x)

            pos_emb = model.get_item_embedding(pos)
            neg_emb = model.get_item_embedding(neg)
            loss = bpr_loss(user_emb, pos_emb, neg_emb)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

            if i % 100 == 0:
                print(f"\rStep {i}/{len(train_loader)} Loss: {loss.item():.4f}", end="")

        elapsed = time.time() - start
        
        # Validation Eval
        val_ndcg = evaluate_val(
            model=model,
            dataset=val_dataset,
            device=device,
            solver=args.solver,
            args=args,
            val_indices=val_indices,
        )
        print(f"\nEpoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f} | Val NDCG@10: {val_ndcg:.4f} | Time: {elapsed:.1f}s")

        save_path = os.path.join("checkpoints", args.save_name)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        if val_ndcg > best_ndcg:
            best_ndcg = val_ndcg
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"  [+] Best model saved to {save_path}")
        else:
            patience_counter += 1
            print(f"  [-] No improvement. Patience: {patience_counter}/{patience_limit}")
            if patience_counter >= patience_limit:
                print("Early stopping triggered.")
                break

    print(f"Training Complete. Best Val NDCG: {best_ndcg:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=str, default="deepm3",
                        choices=["gru", "sasrec", "tisasrec", "deepm3"])
    parser.add_argument("--solver", type=str, default="euler",
                        choices=["baseline", "rk4", "euler", "none"])
    parser.add_argument("--save_name", type=str, default="model.pth")
    parser.add_argument("--hidden_dim", type=int, default=0)
    parser.add_argument("--dataset", type=str, default="ml1m",
                        choices=["ml1m", "amazon"])
    parser.add_argument("--data_dir", type=str, default="")
    parser.add_argument("--lr", type=float, default=0, help="Override learning rate (0=auto)")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--time_jitter_std", type=float, default=0.0)
    parser.add_argument("--time_drop_prob", type=float, default=0.0)
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--val_max_samples", type=int, default=2000)
    parser.add_argument("--val_num_neg", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--time_ablation", type=str, default="full", choices=["full", "none", "t_only", "dt_only"])
    args = parser.parse_args()
    train(args)
