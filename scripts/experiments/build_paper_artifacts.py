import argparse
import copy
import glob
import math
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

import sys

sys.path.append(os.getcwd())
from src.data.dataset import MovieLensDataset
from src.dynamics.gru_baseline import GRUBaseline
from src.dynamics.modeling import DeepM3Model


LEN_GROUP_ORDER = ["Short (<=10)", "Medium (11-15)", "Long (>15)"]
CV_GROUP_ORDER = ["Regular (Low CV)", "Normal (Mid CV)", "Irregular (High CV)"]


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


def first_existing(paths):
    for path in paths:
        if path and os.path.exists(path):
            return path
    return None


def infer_hidden_dim(state_dict) -> int | None:
    weight = state_dict.get("item_emb.weight")
    if weight is not None and getattr(weight, "ndim", 0) == 2:
        return int(weight.shape[1])
    gru_w = state_dict.get("gru.weight_ih_l0")
    if gru_w is not None and getattr(gru_w, "ndim", 0) == 2:
        return int(gru_w.shape[0] // 3)
    return None


def compute_rank_metrics(scores, k=10):
    """Compute HR@K, NDCG@K, MRR, Recall@5, and AUC.
    Assumes index 0 in scores is the positive item."""
    n_items = scores.size(-1)
    _, indices = torch.topk(scores, k, dim=-1)
    row = indices[0]

    hr, ndcg, mrr = 0.0, 0.0, 0.0
    if 0 in row:
        rank = (row == 0).nonzero(as_tuple=True)[0].item()
        hr = 1.0
        ndcg = 1.0 / np.log2(rank + 2)
        mrr = 1.0 / (rank + 1)

    # Recall@5
    _, top5 = torch.topk(scores, min(5, n_items), dim=-1)
    recall5 = 1.0 if 0 in top5[0] else 0.0

    # AUC: fraction of negatives scored below the positive
    pos_score = scores[0, 0].item()
    neg_scores = scores[0, 1:]  # all negatives
    auc = float((neg_scores < pos_score).float().mean().item())

    return {"hr": hr, "ndcg": ndcg, "mrr": mrr, "recall5": recall5, "auc": auc}


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


def build_fixed_negatives(dataset, num_neg: int, seed: int):
    rng = np.random.default_rng(seed)
    neg_pool = []
    for sample in dataset.samples:
        neg = sample_negatives(
            rng=rng,
            n_items=dataset.n_items,
            positive=int(sample["y"]),
            history_items=sample["x"],
            num_neg=num_neg,
        )
        neg_pool.append(np.asarray(neg, dtype=np.int64))
    return neg_pool


def build_user_metadata(dataset):
    rows = []
    for idx, sample in enumerate(dataset.samples):
        x = np.asarray(sample["x"], dtype=np.int64)
        seq_len = int(np.sum(x > 0))

        if "dt" in sample:
            dt = np.asarray(sample["dt"], dtype=np.float64)
        else:
            t = np.asarray(sample["t"], dtype=np.float64)
            dt = np.zeros_like(t)
            if len(t) > 1:
                dt[1:] = t[1:] - t[:-1]

        # Use inter-event gaps for irregularity; drop the first artificial zero-gap.
        dt_used = dt[:seq_len]
        if dt_used.size > 1:
            dt_used = dt_used[1:]
        dt_mean = float(np.mean(dt_used)) if dt_used.size > 0 else 0.0
        dt_std = float(np.std(dt_used)) if dt_used.size > 0 else 0.0
        cv = dt_std / (abs(dt_mean) + 1e-8)
        rows.append({"user_idx": idx, "seq_len": seq_len, "cv": cv})
    return pd.DataFrame(rows)


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


def evaluate_model(model, solver, dataset, neg_pool, device, topk=10):
    metric_lists = {"hr": [], "ndcg": [], "mrr": [], "recall5": [], "auc": [], "latency": []}

    with torch.no_grad():
        for idx in range(len(dataset)):
            sample = dataset[idx]
            x = sample["x"].unsqueeze(0).to(device)
            t = sample["t"].unsqueeze(0).to(device)
            dt = sample["dt"].unsqueeze(0).to(device) if "dt" in sample else None
            pos = int(sample["pos"].item())

            candidates = np.concatenate(([pos], neg_pool[idx]))
            candidates_t = torch.tensor(candidates, dtype=torch.long, device=device)

            start = time.time()
            if solver == "baseline":
                user_emb = model(x)
            else:
                user_emb = model(x, t, dt=dt)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elif device.type == "mps":
                torch.mps.synchronize()
            end = time.time()

            cand_emb = model.get_item_embedding(candidates_t)
            scores = torch.matmul(user_emb, cand_emb.t())
            m = compute_rank_metrics(scores, k=topk)
            for key in ("hr", "ndcg", "mrr", "recall5", "auc"):
                metric_lists[key].append(m[key])
            metric_lists["latency"].append((end - start) * 1000.0)

    return {k: np.asarray(v, dtype=np.float64) for k, v in metric_lists.items()}


def paired_t_test(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    try:
        from scipy import stats  # pylint: disable=import-outside-toplevel

        t_stat, p_val = stats.ttest_rel(x, y)
        return float(t_stat), float(p_val)
    except Exception:
        diff = x - y
        n = diff.size
        if n < 2:
            return float("nan"), float("nan")
        diff_mean = float(np.mean(diff))
        diff_std = float(np.std(diff, ddof=1))
        if diff_std == 0:
            return float("inf"), 0.0
        t_stat = diff_mean / (diff_std / math.sqrt(n))
        # Normal approximation fallback if scipy is unavailable.
        p_val = math.erfc(abs(t_stat) / math.sqrt(2.0))
        return float(t_stat), float(p_val)


def compute_seq_len_table(per_user_df):
    seq_df = per_user_df.copy()
    seq_df["len_group"] = pd.cut(
        seq_df["seq_len"],
        bins=[0, 10, 15, np.inf],
        labels=LEN_GROUP_ORDER,
        right=True,
    )
    out = (
        seq_df.groupby("len_group", observed=False)[["ndcg_baseline", "ndcg_euler"]]
        .mean()
        .reset_index()
    )
    out["lift_pct"] = (
        (out["ndcg_euler"] - out["ndcg_baseline"]) / (out["ndcg_baseline"] + 1e-12) * 100.0
    )
    return out


def compute_irregular_table(per_user_df):
    cv_df = per_user_df.copy()
    cv_df["cv_group"] = pd.qcut(
        cv_df["cv"], q=3, labels=CV_GROUP_ORDER, duplicates="drop"
    )
    out = (
        cv_df.groupby("cv_group", observed=False)[["ndcg_baseline", "ndcg_euler"]]
        .mean()
        .reset_index()
    )
    out["lift_pct"] = (
        (out["ndcg_euler"] - out["ndcg_baseline"]) / (out["ndcg_baseline"] + 1e-12) * 100.0
    )
    return out


def plot_fig3_seq_len(seq_df, save_path):
    seq_df = seq_df.copy()
    seq_df["len_group"] = pd.Categorical(
        seq_df["len_group"], categories=LEN_GROUP_ORDER, ordered=True
    )
    seq_df = seq_df.sort_values("len_group")

    x = np.arange(len(seq_df))
    width = 0.35
    plt.figure(figsize=(8, 5))
    plt.bar(
        x - width / 2,
        seq_df["ndcg_baseline"].values,
        width,
        label="Baseline (GRU)",
        color="#AAAAAA",
        alpha=0.8,
    )
    plt.bar(
        x + width / 2,
        seq_df["ndcg_euler"].values,
        width,
        label="DeepM3 (Euler)",
        color="#D62728",
        alpha=0.9,
    )

    for i, lift in enumerate(seq_df["lift_pct"].values):
        y = max(seq_df["ndcg_baseline"].values[i], seq_df["ndcg_euler"].values[i]) + 0.01
        plt.text(i, y, f"{lift:+.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.ylabel("NDCG@10", fontsize=12)
    plt.title("Performance by Sequence Length", fontsize=14)
    plt.xticks(x, seq_df["len_group"].astype(str).values, fontsize=11)
    ymax = max(float(seq_df["ndcg_baseline"].max()), float(seq_df["ndcg_euler"].max())) + 0.08
    plt.ylim(0, ymax)
    plt.legend(fontsize=11)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close()


def plot_fig4_sensitivity(sens_df, save_path):
    sens_df = sens_df.sort_values("hidden_dim")
    plt.figure(figsize=(7, 5))
    plt.plot(
        sens_df["hidden_dim"].values,
        sens_df["ndcg@10"].values,
        marker="o",
        linestyle="-",
        linewidth=2,
        markersize=8,
        color="#1f77b4",
        label="DeepM3 (Euler)",
    )

    best_idx = sens_df["ndcg@10"].idxmax()
    best_dim = int(sens_df.loc[best_idx, "hidden_dim"])
    best_ndcg = float(sens_df.loc[best_idx, "ndcg@10"])
    plt.annotate(
        f"Peak: {best_ndcg:.4f}",
        xy=(best_dim, best_ndcg),
        xytext=(best_dim, best_ndcg + 0.01),
        ha="center",
        arrowprops=dict(arrowstyle="->", color="black"),
    )

    plt.xscale("log", base=2)
    plt.xticks(
        sens_df["hidden_dim"].values,
        labels=[str(int(v)) for v in sens_df["hidden_dim"].values],
        fontsize=11,
    )
    plt.xlabel("Hidden Dimension ($d$)", fontsize=12)
    plt.ylabel("NDCG@10", fontsize=12)
    plt.title("Hyperparameter Sensitivity", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close()


def evaluate_sensitivity(cfg, dataset, neg_pool, device, topk, checkpoint_paths):
    rows = []
    for ckpt in checkpoint_paths:
        if not os.path.exists(ckpt):
            continue
        state_dict = torch.load(ckpt, map_location=device, weights_only=False)
        hidden_dim = infer_hidden_dim(state_dict)
        if hidden_dim is None:
            print(f" Skip sensitivity ckpt (cannot infer hidden_dim): {ckpt}")
            continue
        try:
            model = build_model(cfg, dataset.n_items, "euler", state_dict, device)
        except RuntimeError as e:
            # Old/incompatible checkpoints are common when architecture evolves.
            print(f" Skip incompatible sensitivity ckpt: {ckpt}")
            print(f"   Reason: {str(e).splitlines()[0]}")
            continue
        metrics = evaluate_model(
            model=model,
            solver="euler",
            dataset=dataset,
            neg_pool=neg_pool,
            device=device,
            topk=topk,
        )
        rows.append(
            {
                "checkpoint": ckpt,
                "hidden_dim": hidden_dim,
                "ndcg@10": float(np.mean(metrics["ndcg"])),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["hidden_dim", "ndcg@10"])
    raw_df = pd.DataFrame(rows).sort_values(["hidden_dim", "checkpoint"])
    agg_df = raw_df.groupby("hidden_dim", as_index=False)["ndcg@10"].mean()
    return agg_df


def main(args):
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    Path(args.assets_dir).mkdir(parents=True, exist_ok=True)

    device = pick_device(args.device)
    dataset = MovieLensDataset(mode="test")
    print(f" Test samples: {len(dataset)} | device={device}")

    print(" Building fixed negative candidates...")
    neg_pool = build_fixed_negatives(dataset, num_neg=args.num_neg, seed=args.seed)
    per_user_df = build_user_metadata(dataset)

    ckpt_baseline = first_existing(
        [args.baseline_ckpt, "checkpoints/model_gru_baseline.pth", "checkpoints/model_baseline_gru.pth"]
    )
    ckpt_none = first_existing([args.none_ckpt, "checkpoints/model_ode_none.pth"])
    ckpt_euler = first_existing([args.euler_ckpt, "checkpoints/model_ode_euler.pth"])
    ckpt_rk4 = first_existing([args.rk4_ckpt, "checkpoints/model_ode_rk4.pth"])

    model_specs = [
        ("Baseline", "baseline", ckpt_baseline),
        ("DeepM3", "none", ckpt_none),
        ("DeepM3", "euler", ckpt_euler),
        ("DeepM3", "rk4", ckpt_rk4),
    ]

    overall_rows = []
    model_outputs = {}
    for method, solver, ckpt in model_specs:
        if not ckpt:
            print(f" Skip {method}/{solver}: checkpoint not found.")
            continue
        print(f" Evaluating {method}/{solver} from {ckpt}")
        state_dict = torch.load(ckpt, map_location=device, weights_only=False)
        model = build_model(cfg, dataset.n_items, solver, state_dict, device)
        metrics = evaluate_model(
            model=model,
            solver=solver,
            dataset=dataset,
            neg_pool=neg_pool,
            device=device,
            topk=args.topk,
        )
        model_outputs[solver] = metrics
        per_user_df[f"hr_{solver}"] = metrics["hr"]
        per_user_df[f"ndcg_{solver}"] = metrics["ndcg"]
        overall_rows.append(
            {
                "method": method,
                "solver": solver,
                "time_per_sample_ms": float(np.mean(metrics["latency"])),
                "hr@10": float(np.mean(metrics["hr"])),
                "ndcg@10": float(np.mean(metrics["ndcg"])),
                "mrr": float(np.mean(metrics["mrr"])),
                "recall@5": float(np.mean(metrics["recall5"])),
                "auc": float(np.mean(metrics["auc"])),
                "checkpoint": ckpt,
            }
        )

    if not overall_rows:
        raise RuntimeError("No valid checkpoints found. Please provide at least one checkpoint path.")

    overall_df = pd.DataFrame(overall_rows)
    order = pd.CategoricalDtype(["baseline", "none", "euler", "rk4"], ordered=True)
    overall_df["solver"] = overall_df["solver"].astype(order)
    overall_df = overall_df.sort_values("solver").reset_index(drop=True)
    overall_df.to_csv(out_dir / "overall_metrics.csv", index=False)

    table1_cols = ["method", "solver", "time_per_sample_ms", "hr@10", "ndcg@10", "mrr", "recall@5", "auc"]
    table1_df = overall_df[[c for c in table1_cols if c in overall_df.columns]].copy()
    if "baseline" in overall_df["solver"].astype(str).values:
        base_row = overall_df[overall_df["solver"].astype(str) == "baseline"].iloc[0]
        for metric in ["hr@10", "ndcg@10", "mrr", "recall@5", "auc"]:
            if metric in table1_df.columns:
                table1_df[f"{metric}_lift_pct"] = (
                    (table1_df[metric] - base_row[metric]) / (base_row[metric] + 1e-12) * 100.0
                )
    table1_df.to_csv(out_dir / "table1_main.csv", index=False)

    per_user_df.to_csv(out_dir / "per_user_metrics.csv", index=False)

    if "baseline" in model_outputs and "euler" in model_outputs:
        t_stat, p_val = paired_t_test(model_outputs["euler"]["ndcg"], model_outputs["baseline"]["ndcg"])
        sig_df = pd.DataFrame(
            [
                {
                    "mean_ndcg_baseline": float(np.mean(model_outputs["baseline"]["ndcg"])),
                    "mean_ndcg_euler": float(np.mean(model_outputs["euler"]["ndcg"])),
                    "t_statistic": t_stat,
                    "p_value": p_val,
                }
            ]
        )
        sig_df.to_csv(out_dir / "significance.csv", index=False)

        seq_df = compute_seq_len_table(per_user_df)
        seq_df.to_csv(out_dir / "fig3_seq_len.csv", index=False)
        plot_fig3_seq_len(seq_df, os.path.join(args.assets_dir, "Fig3_SeqLen.pdf"))

        irr_df = compute_irregular_table(per_user_df)
        irr_df.to_csv(out_dir / "table2_irregular.csv", index=False)
    else:
        print(" Skip sequence/irregular/significance outputs: baseline and euler are both required.")

    sensitivity_paths = sorted(glob.glob(args.sensitivity_glob)) if args.sensitivity_glob else []
    if ckpt_euler and ckpt_euler not in sensitivity_paths:
        sensitivity_paths.append(ckpt_euler)
    if sensitivity_paths:
        print(f" Evaluating sensitivity on {len(sensitivity_paths)} checkpoints...")
        sens_df = evaluate_sensitivity(
            cfg=cfg,
            dataset=dataset,
            neg_pool=neg_pool,
            device=device,
            topk=args.topk,
            checkpoint_paths=sensitivity_paths,
        )
        if not sens_df.empty:
            sens_df.to_csv(out_dir / "fig4_sensitivity.csv", index=False)
            plot_fig4_sensitivity(sens_df, os.path.join(args.assets_dir, "Fig4_Sensitivity.pdf"))
        else:
            print(" Skip Fig4: no valid sensitivity checkpoints.")
    else:
        print(" Skip Fig4: no checkpoint matched sensitivity glob.")

    print(f" Artifacts generated in: {out_dir}")
    print(f" Figures generated in: {args.assets_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--output_dir", type=str, default="results/ml1m")
    parser.add_argument("--assets_dir", type=str, default="assets")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_neg", type=int, default=100)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])

    parser.add_argument("--baseline_ckpt", type=str, default="")
    parser.add_argument("--none_ckpt", type=str, default="")
    parser.add_argument("--euler_ckpt", type=str, default="")
    parser.add_argument("--rk4_ckpt", type=str, default="")
    parser.add_argument("--sensitivity_glob", type=str, default="")

    main(parser.parse_args())
