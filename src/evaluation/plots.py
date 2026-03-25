"""Visualization module for experiment results.

Generates all figures needed for the paper / report:
  1. Training curves (train vs val loss)
  2. Sparsity-accuracy frontier
  3. Cumulative tracking deviation
  4. Portfolio weight heatmap
  5. Regime-conditioned performance
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Consistent style
plt.rcParams.update({
    "figure.figsize": (10, 6),
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "legend.fontsize": 11,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
})


def plot_training_curves(
    history: dict,
    title: str = "Training Curves",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot train and validation loss curves."""
    fig, ax = plt.subplots()
    epochs = range(1, len(history["train_losses"]) + 1)

    ax.plot(epochs, history["train_losses"], "b-", label="Train Loss", linewidth=2)
    ax.plot(epochs, history["val_losses"], "r--", label="Val Loss", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)

    if save_path:
        fig.savefig(save_path)
    return fig


def plot_sparsity_frontier(
    results: list[dict],
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot tracking error vs sparsity (K) across methods.

    Expects results with keys: K, tracking_error, loss, model.
    """
    df = pd.DataFrame(results)
    fig, ax = plt.subplots()

    for (model, loss), group in df.groupby(["model", "loss"]):
        agg = group.groupby("K")["tracking_error"].agg(["mean", "std"]).reset_index()
        label = f"{model} ({loss})"
        ax.errorbar(
            agg["K"], agg["mean"], yerr=agg["std"],
            marker="o", capsize=4, linewidth=2, label=label,
        )

    ax.set_xlabel("Sparsity Level K (# stocks)")
    ax.set_ylabel("Annualized Tracking Error")
    ax.set_title("Sparsity-Accuracy Frontier")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.invert_xaxis()  # fewer stocks → harder → left side

    if save_path:
        fig.savefig(save_path)
    return fig


def plot_cumulative_tracking(
    portfolio_returns: np.ndarray,
    index_returns: np.ndarray,
    dates: pd.DatetimeIndex | None = None,
    title: str = "Cumulative Returns",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot cumulative returns of portfolio vs index."""
    cum_port = np.cumprod(1 + portfolio_returns)
    cum_idx = np.cumprod(1 + index_returns)
    cum_diff = np.cumsum(portfolio_returns - index_returns)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[2, 1])

    x = dates if dates is not None else np.arange(len(portfolio_returns))

    ax1.plot(x, cum_port, "b-", label="Portfolio", linewidth=1.5)
    ax1.plot(x, cum_idx, "k--", label="Index", linewidth=1.5)
    ax1.set_ylabel("Cumulative Return")
    ax1.set_title(title)
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.fill_between(x, cum_diff, 0, alpha=0.3, color="red")
    ax2.plot(x, cum_diff, "r-", linewidth=1)
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.set_ylabel("Cumulative Tracking Diff")
    ax2.set_xlabel("Date" if dates is not None else "Day")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


def plot_weight_heatmap(
    weights_history: list[np.ndarray],
    tickers: list[str] | None = None,
    dates: list | None = None,
    top_k: int = 20,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Heatmap of portfolio weights over time (top K stocks by avg weight)."""
    W = np.array(weights_history)  # (T_rebal, N)
    n = W.shape[1]

    # Select top-K by average weight
    avg_w = W.mean(axis=0)
    top_idx = np.argsort(avg_w)[-top_k:][::-1]
    W_top = W[:, top_idx]

    labels = [tickers[i] for i in top_idx] if tickers else [f"Stock {i}" for i in top_idx]
    date_labels = [str(d)[:10] for d in dates] if dates else [str(i) for i in range(W.shape[0])]

    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(
        W_top.T,
        xticklabels=date_labels,
        yticklabels=labels,
        cmap="YlOrRd",
        ax=ax,
        cbar_kws={"label": "Weight"},
    )
    ax.set_xlabel("Rebalancing Date")
    ax.set_ylabel("Stock")
    ax.set_title(f"Portfolio Weights (Top {top_k} Stocks)")

    # Rotate x-axis labels
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path)
    return fig


def plot_regime_performance(
    results: list[dict],
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Bar chart of tracking error by market regime (bull/bear/sideways)."""
    df = pd.DataFrame(results)
    if "regime" not in df.columns:
        raise ValueError("Results must contain 'regime' column for regime analysis")

    fig, ax = plt.subplots()

    regime_order = ["bull", "sideways", "bear"]
    colors = {"bull": "#2ecc71", "sideways": "#f39c12", "bear": "#e74c3c"}

    pivot = df.groupby(["regime", "loss"])["tracking_error"].mean().unstack()
    pivot = pivot.reindex(regime_order)
    pivot.plot(kind="bar", ax=ax, color=["#3498db", "#e67e22"], edgecolor="black")

    ax.set_xlabel("Market Regime")
    ax.set_ylabel("Annualized Tracking Error")
    ax.set_title("Tracking Error by Market Regime")
    ax.legend(title="Loss Type")
    ax.grid(alpha=0.3, axis="y")
    plt.xticks(rotation=0)

    if save_path:
        fig.savefig(save_path)
    return fig


def plot_method_comparison(
    results: list[dict],
    metric: str = "tracking_error",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Box plot comparing methods across folds for a given metric."""
    df = pd.DataFrame(results)
    df["method"] = df["model"] + " + " + df["loss"]

    fig, ax = plt.subplots()
    methods = df["method"].unique()
    data = [df[df["method"] == m][metric].values for m in methods]

    bp = ax.boxplot(data, labels=methods, patch_artist=True)
    colors = plt.cm.Set2(np.linspace(0, 1, len(methods)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"{metric.replace('_', ' ').title()} by Method")
    ax.grid(alpha=0.3, axis="y")

    if save_path:
        fig.savefig(save_path)
    return fig


def save_all_figures(output_dir: str | Path = "figures") -> None:
    """Close all open figures and remind user to call individual plot functions."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    plt.close("all")
