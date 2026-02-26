import argparse
import copy
import os
import sys
import time

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.getcwd())
from src.data.dataset import MovieLensDataset
from src.dynamics.gru_baseline import GRUBaseline
from src.dynamics.modeling import DeepM3Model


def pick_device(device_name: str) -> torch.device:
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if device_name == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def infer_hidden_dim(state_dict) -> int | None:
    weight = state_dict.get("item_emb.weight")
    if weight is not None and getattr(weight, "ndim", 0) == 2:
        return int(weight.shape[1])
    gru_w = state_dict.get("gru.weight_ih_l0")
    if gru_w is not None and getattr(gru_w, "ndim", 0) == 2:
        # GRU weight_ih shape: [3*hidden_dim, input_dim]
        return int(gru_w.shape[0] // 3)
    return None


def sample_negatives(rng, n_items: int, positive: int, history_items, num_neg: int):
    blocked = {int(positive)}
    blocked.update(int(v) for v in history_items if int(v) > 0)
    negatives = []
    while len(negatives) < num_neg:
        cand = int(rng.integers(1, n_items))
        if cand in blocked:
            continue
        negatives.append(cand)
    return negatives


def compute_metrics(scores, k=10):
    _, indices = torch.topk(scores, k, dim=-1)
    row = indices[0]
    if 0 in row:
        rank = (row == 0).nonzero(as_tuple=True)[0].item()
        return 1.0, 1.0 / np.log2(rank + 2)
    return 0.0, 0.0


def build_model(cfg, n_items: int, solver: str, state_dict, device):
    hidden_dim = infer_hidden_dim(state_dict)
    local_cfg = copy.deepcopy(cfg)
    if hidden_dim is not None:
        local_cfg.setdefault("model", {})
        local_cfg["model"]["hidden_dim"] = hidden_dim

    if solver == "baseline":
        model = GRUBaseline(
            n_items=n_items,
            hidden_dim=local_cfg["model"]["hidden_dim"],
        ).to(device)
    else:
        model = DeepM3Model(local_cfg, n_items=n_items, solver=solver).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def evaluate(args):
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model not found: {args.model_path}")

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = pick_device(args.device)
    dataset = MovieLensDataset(mode="test")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    rng = np.random.default_rng(args.seed)

    state_dict = torch.load(args.model_path, map_location=device, weights_only=False)
    model = build_model(cfg, dataset.n_items, args.solver, state_dict, device)

    iterator = tqdm(loader) if args.verbose else loader
    total_hr = 0.0
    total_ndcg = 0.0
    latencies = []

    with torch.no_grad():
        for batch in iterator:
            x = batch["x"].to(device)
            t = batch["t"].to(device)
            dt = batch["dt"].to(device) if "dt" in batch else None
            pos = batch["pos"].to(device)

            negatives = sample_negatives(
                rng=rng,
                n_items=dataset.n_items,
                positive=int(pos.item()),
                history_items=batch["x"][0].tolist(),
                num_neg=args.num_neg,
            )
            neg_tensor = torch.tensor(negatives, dtype=torch.long, device=device)
            candidates = torch.cat([pos, neg_tensor], dim=0)

            start = time.time()
            if args.solver == "baseline":
                user_emb = model(x)
            else:
                user_emb = model(x, t, dt=dt)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elif device.type == "mps":
                torch.mps.synchronize()
            end = time.time()

            cand_emb = model.get_item_embedding(candidates)
            scores = torch.matmul(user_emb, cand_emb.t())
            hr, ndcg = compute_metrics(scores, k=args.topk)
            total_hr += hr
            total_ndcg += ndcg
            latencies.append((end - start) * 1000.0)

    n_samples = len(dataset)
    avg_hr = total_hr / n_samples
    avg_ndcg = total_ndcg / n_samples
    avg_lat = float(np.mean(latencies))
    print(f"{args.model_path},{avg_hr:.4f},{avg_ndcg:.4f},{avg_lat:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--model_path", required=True, help="Path to .pth checkpoint")
    parser.add_argument(
        "--solver",
        required=True,
        choices=["baseline", "none", "euler", "rk4"],
        help="Model solver to instantiate for this checkpoint",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_neg", type=int, default=100)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    evaluate(args)
