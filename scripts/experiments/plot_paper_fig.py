import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LEN_GROUP_ORDER = ["Short (<=10)", "Medium (11-15)", "Long (>15)"]


def plot_fig3_seq_len_performance(fig3_csv: str, save_path: str):
    if not os.path.exists(fig3_csv):
        raise FileNotFoundError(f"Figure 3 source CSV not found: {fig3_csv}")

    df = pd.read_csv(fig3_csv)
    df["len_group"] = pd.Categorical(df["len_group"], categories=LEN_GROUP_ORDER, ordered=True)
    df = df.sort_values("len_group")

    x = np.arange(len(df))
    width = 0.35
    plt.figure(figsize=(8, 5))
    plt.bar(
        x - width / 2,
        df["ndcg_baseline"].values,
        width,
        label="Baseline (GRU)",
        color="#AAAAAA",
        alpha=0.8,
    )
    plt.bar(
        x + width / 2,
        df["ndcg_euler"].values,
        width,
        label="DeepM3 (Euler)",
        color="#D62728",
        alpha=0.9,
    )

    for i, lift in enumerate(df["lift_pct"].values):
        y = max(df["ndcg_baseline"].values[i], df["ndcg_euler"].values[i]) + 0.01
        plt.text(i, y, f"{lift:+.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.ylabel("NDCG@10", fontsize=12)
    plt.title("Performance by Sequence Length", fontsize=14)
    plt.xticks(x, df["len_group"].astype(str).values, fontsize=11)
    ymax = max(float(df["ndcg_baseline"].max()), float(df["ndcg_euler"].max())) + 0.08
    plt.ylim(0, ymax)
    plt.legend(fontsize=11)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close()
    print(f" Figure 3 saved to {save_path}")


def plot_fig4_sensitivity(fig4_csv: str, save_path: str):
    if not os.path.exists(fig4_csv):
        raise FileNotFoundError(f"Figure 4 source CSV not found: {fig4_csv}")

    df = pd.read_csv(fig4_csv).sort_values("hidden_dim")
    plt.figure(figsize=(7, 5))
    plt.plot(
        df["hidden_dim"].values,
        df["ndcg@10"].values,
        marker="o",
        linestyle="-",
        linewidth=2,
        markersize=8,
        color="#1f77b4",
        label="DeepM3 (Euler)",
    )

    best_idx = df["ndcg@10"].idxmax()
    best_dim = int(df.loc[best_idx, "hidden_dim"])
    best_ndcg = float(df.loc[best_idx, "ndcg@10"])
    plt.annotate(
        f"Peak: {best_ndcg:.4f}",
        xy=(best_dim, best_ndcg),
        xytext=(best_dim, best_ndcg + 0.01),
        ha="center",
        arrowprops=dict(arrowstyle="->", color="black"),
    )

    plt.xscale("log", base=2)
    plt.xticks(
        df["hidden_dim"].values,
        labels=[str(int(v)) for v in df["hidden_dim"].values],
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
    print(f" Figure 4 saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fig3_csv", default="results/paper/fig3_seq_len.csv")
    parser.add_argument("--fig4_csv", default="results/paper/fig4_sensitivity.csv")
    parser.add_argument("--fig3_out", default="assets/Fig3_SeqLen.pdf")
    parser.add_argument("--fig4_out", default="assets/Fig4_Sensitivity.pdf")
    args = parser.parse_args()

    plot_fig3_seq_len_performance(args.fig3_csv, args.fig3_out)
    plot_fig4_sensitivity(args.fig4_csv, args.fig4_out)
