"""Generate all publication-quality figures and statistical analysis.

Usage:
    python scripts/generate_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

# ── Style ──
plt.rcParams.update({
    "figure.figsize": (10, 6),
    "font.size": 12,
    "font.family": "serif",
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "legend.fontsize": 11,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
})

RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)

METHOD_STYLES = {
    "factor + task": {"color": "#2563eb", "marker": "o", "ls": "-"},
    "factor + mse":  {"color": "#93c5fd", "marker": "s", "ls": "--"},
    "neural + task": {"color": "#dc2626", "marker": "^", "ls": "-"},
    "neural + mse":  {"color": "#fca5a5", "marker": "D", "ls": "--"},
}


def load_all_results() -> pd.DataFrame:
    """Load and merge all result JSON files."""
    all_results = []
    for path in RESULTS_DIR.glob("*_results.json"):
        with open(path) as f:
            data = json.load(f)
        all_results.extend(data["results"])

    df = pd.DataFrame(all_results)
    df["method"] = df["model"] + " + " + df["loss"]
    return df


# ── Figure 1: Sparsity-Accuracy Frontier ──
def fig_sparsity_frontier(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots()
    for method, style in METHOD_STYLES.items():
        sub = df[df["method"] == method]
        if sub.empty:
            continue
        agg = sub.groupby("K")["tracking_error"].agg(["mean", "std"]).reset_index()
        ax.errorbar(
            agg["K"], agg["mean"], yerr=agg["std"],
            marker=style["marker"], color=style["color"], ls=style["ls"],
            capsize=4, linewidth=2, label=method, markersize=8,
        )

    ax.set_xlabel("Sparsity Level K (# stocks held)")
    ax.set_ylabel("Annualized Tracking Error")
    ax.set_title("Sparsity-Accuracy Frontier: DFL vs Two-Stage")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(FIGURES_DIR / "fig1_sparsity_frontier.pdf")
    fig.savefig(FIGURES_DIR / "fig1_sparsity_frontier.png")
    plt.close(fig)
    print("  [saved] fig1_sparsity_frontier")


# ── Figure 2: Turnover vs K ──
def fig_turnover_frontier(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots()
    for method, style in METHOD_STYLES.items():
        sub = df[df["method"] == method]
        if sub.empty:
            continue
        agg = sub.groupby("K")["avg_turnover"].agg(["mean", "std"]).reset_index()
        ax.errorbar(
            agg["K"], agg["mean"], yerr=agg["std"],
            marker=style["marker"], color=style["color"], ls=style["ls"],
            capsize=4, linewidth=2, label=method, markersize=8,
        )

    ax.set_xlabel("Sparsity Level K (# stocks held)")
    ax.set_ylabel("Average Turnover per Rebalance")
    ax.set_title("Portfolio Turnover vs Sparsity")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(FIGURES_DIR / "fig2_turnover_frontier.pdf")
    fig.savefig(FIGURES_DIR / "fig2_turnover_frontier.png")
    plt.close(fig)
    print("  [saved] fig2_turnover_frontier")


# ── Figure 3: Box Plot of TE at K=20 ──
def fig_te_boxplot(df: pd.DataFrame, K: int = 20) -> None:
    sub = df[df["K"] == K].copy()
    if sub.empty:
        print(f"  [skip] No data for K={K}")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    methods = [m for m in METHOD_STYLES if m in sub["method"].unique()]
    data = [sub[sub["method"] == m]["tracking_error"].values for m in methods]
    colors = [METHOD_STYLES[m]["color"] for m in methods]

    bp = ax.boxplot(data, labels=methods, patch_artist=True, widths=0.5)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)

    ax.set_ylabel("Annualized Tracking Error")
    ax.set_title(f"Tracking Error Distribution (K={K}, 9 Folds)")
    ax.grid(alpha=0.3, axis="y")
    plt.xticks(rotation=15)
    fig.savefig(FIGURES_DIR / f"fig3_te_boxplot_K{K}.pdf")
    fig.savefig(FIGURES_DIR / f"fig3_te_boxplot_K{K}.png")
    plt.close(fig)
    print(f"  [saved] fig3_te_boxplot_K{K}")


# ── Figure 4: DFL Advantage Heatmap ──
def fig_dfl_advantage(df: pd.DataFrame) -> None:
    """Heatmap: % improvement of DFL over MSE for each model × K."""
    fig, ax = plt.subplots(figsize=(8, 4))

    models = ["factor", "neural"]
    K_vals = sorted(df["K"].unique())
    advantage = np.full((len(models), len(K_vals)), np.nan)

    for i, model in enumerate(models):
        for j, K in enumerate(K_vals):
            dfl = df[(df["model"] == model) & (df["loss"] == "task") & (df["K"] == K)]["tracking_error"]
            mse = df[(df["model"] == model) & (df["loss"] == "mse") & (df["K"] == K)]["tracking_error"]
            if len(dfl) > 0 and len(mse) > 0:
                advantage[i, j] = (mse.mean() - dfl.mean()) / mse.mean() * 100

    im = ax.imshow(advantage, cmap="RdYlGn", aspect="auto", vmin=-10, vmax=20)
    ax.set_xticks(range(len(K_vals)))
    ax.set_xticklabels(K_vals)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([m.title() for m in models])
    ax.set_xlabel("Sparsity Level K")
    ax.set_title("DFL Advantage over MSE (% TE Reduction)")

    for i in range(len(models)):
        for j in range(len(K_vals)):
            if not np.isnan(advantage[i, j]):
                ax.text(j, i, f"{advantage[i, j]:+.1f}%",
                        ha="center", va="center", fontweight="bold", fontsize=12)

    fig.colorbar(im, label="% TE Reduction")
    fig.savefig(FIGURES_DIR / "fig4_dfl_advantage.pdf")
    fig.savefig(FIGURES_DIR / "fig4_dfl_advantage.png")
    plt.close(fig)
    print("  [saved] fig4_dfl_advantage")


# ── Figure 5: Cumulative Tracking Chart ──
def fig_cumulative_tracking(df: pd.DataFrame, K: int = 20) -> None:
    """Cumulative return of portfolio vs index for each method at a given K."""
    sub = df[df["K"] == K].copy()
    if sub.empty or "portfolio_returns" not in sub.columns:
        print(f"  [skip] No portfolio_returns data for K={K}")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, metric_label, ret_key, idx_key in [
        (axes[0], "Cumulative Return", "portfolio_returns", "index_returns"),
        (axes[1], "Cumulative Tracking Difference", None, None),
    ]:
        for method, style in METHOD_STYLES.items():
            rows = sub[sub["method"] == method]
            if rows.empty:
                continue
            # Use longest fold for visualization
            best_row = rows.loc[rows["portfolio_returns"].apply(len).idxmax()]
            port_r = np.array(best_row["portfolio_returns"])
            idx_r = np.array(best_row["index_returns"])

            if ret_key is not None:
                cum_port = np.cumprod(1 + port_r)
                cum_idx = np.cumprod(1 + idx_r)
                ax.plot(cum_port, color=style["color"], ls=style["ls"],
                        linewidth=1.5, label=f"{method} (portfolio)")
                if method == list(METHOD_STYLES.keys())[0]:
                    ax.plot(cum_idx, color="black", ls=":", linewidth=1.5,
                            label="Index", alpha=0.7)
            else:
                cum_diff = np.cumsum(port_r - idx_r) * 100
                ax.plot(cum_diff, color=style["color"], ls=style["ls"],
                        linewidth=1.5, label=method)

        if ret_key is not None:
            ax.set_ylabel("Cumulative Return (Growth of $1)")
            ax.set_title(f"Portfolio vs Index (K={K})")
        else:
            ax.set_ylabel("Cumulative Tracking Diff (bps)")
            ax.set_title(f"Cumulative Tracking Error (K={K})")
            ax.axhline(0, color="black", ls=":", alpha=0.5)

        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_xlabel("Trading Days")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"fig5_cumulative_tracking_K{K}.pdf")
    fig.savefig(FIGURES_DIR / f"fig5_cumulative_tracking_K{K}.png")
    plt.close(fig)
    print(f"  [saved] fig5_cumulative_tracking_K{K}")


# ── Statistical Significance ──
def statistical_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Paired t-test: DFL vs MSE for each model × K."""
    rows = []
    models = df["model"].unique()
    K_vals = sorted(df["K"].unique())

    for model in models:
        for K in K_vals:
            dfl = df[(df["model"] == model) & (df["loss"] == "task") & (df["K"] == K)]
            mse = df[(df["model"] == model) & (df["loss"] == "mse") & (df["K"] == K)]

            # Align by fold
            dfl_te = dfl.sort_values("fold")["tracking_error"].values
            mse_te = mse.sort_values("fold")["tracking_error"].values

            if len(dfl_te) < 3 or len(mse_te) < 3 or len(dfl_te) != len(mse_te):
                continue

            t_stat, p_val = ttest_rel(dfl_te, mse_te)
            mean_diff = dfl_te.mean() - mse_te.mean()
            pct_diff = mean_diff / mse_te.mean() * 100
            dfl_wins = sum(d < m for d, m in zip(dfl_te, mse_te))

            rows.append({
                "model": model,
                "K": K,
                "DFL_TE": f"{dfl_te.mean():.4f}",
                "MSE_TE": f"{mse_te.mean():.4f}",
                "diff": f"{mean_diff:+.4f}",
                "pct": f"{pct_diff:+.1f}%",
                "t_stat": f"{t_stat:.2f}",
                "p_value": f"{p_val:.4f}",
                "sig": "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.1 else "ns",
                "DFL_wins": f"{dfl_wins}/{len(dfl_te)}",
            })

    results_df = pd.DataFrame(rows)
    return results_df


# ── Summary Table ──
def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Generate mean ± std table for all methods."""
    agg = df.groupby(["method", "K"]).agg(
        TE_mean=("tracking_error", "mean"),
        TE_std=("tracking_error", "std"),
        TO_mean=("avg_turnover", "mean"),
        TO_std=("avg_turnover", "std"),
        MTD_mean=("max_tracking_deviation", "mean"),
    ).reset_index()
    agg["TE"] = agg.apply(lambda r: f"{r['TE_mean']:.4f} ± {r['TE_std']:.4f}", axis=1)
    agg["Turnover"] = agg.apply(lambda r: f"{r['TO_mean']:.4f} ± {r['TO_std']:.4f}", axis=1)
    agg["MTD"] = agg["MTD_mean"].apply(lambda x: f"{x:.4f}")
    return agg[["method", "K", "TE", "Turnover", "MTD"]]


# ── Figure 6: Turnover Penalty Ablation ──
def fig_turnover_ablation() -> None:
    """Bar chart comparing Neural+DFL at λ=0, 0.01, 0.1 vs MSE baseline."""
    ablation_dir = Path("results/ablation")
    if not ablation_dir.exists():
        print("  [skip] No ablation results")
        return

    configs = [
        ("λ=0", ablation_dir / "neural_task_lambda0.0_results.json", "#dc2626"),
        ("λ=0.01", ablation_dir / "neural_task_lambda0.01_results.json", "#f97316"),
        ("λ=0.1", ablation_dir / "neural_task_lambda0.1_results.json", "#fbbf24"),
        ("MSE (baseline)", Path("results/neural_mse_results.json"), "#93c5fd"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    K_vals = [5, 10, 20, 30, 50]
    x = np.arange(len(K_vals))
    width = 0.2

    for ax_idx, (metric, ylabel) in enumerate([
        ("tracking_error", "Annualized Tracking Error"),
        ("avg_turnover", "Average Turnover"),
    ]):
        ax = axes[ax_idx]
        for i, (label, path, color) in enumerate(configs):
            if not path.exists():
                continue
            with open(path) as f:
                data = json.load(f)["results"]
            means = []
            stds = []
            for K in K_vals:
                vals = [r[metric] for r in data if r["K"] == K]
                means.append(np.mean(vals) if vals else 0)
                stds.append(np.std(vals) if vals else 0)
            ax.bar(x + i * width, means, width, yerr=stds, label=label,
                   color=color, capsize=3, alpha=0.85)

        ax.set_xlabel("Sparsity Level K")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x + 1.5 * width)
        ax.set_xticklabels(K_vals)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3, axis="y")

    axes[0].set_title("Tracking Error: Turnover Penalty Ablation (Neural)")
    axes[1].set_title("Turnover: Turnover Penalty Ablation (Neural)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig6_turnover_ablation.pdf")
    fig.savefig(FIGURES_DIR / "fig6_turnover_ablation.png")
    plt.close(fig)
    print("  [saved] fig6_turnover_ablation")


def main():
    print("Loading results...")
    df = load_all_results()
    print(f"  {len(df)} result rows: {df['method'].value_counts().to_dict()}")

    print("\nGenerating figures...")
    fig_sparsity_frontier(df)
    fig_turnover_frontier(df)
    fig_te_boxplot(df, K=20)
    fig_dfl_advantage(df)
    fig_cumulative_tracking(df, K=20)
    fig_turnover_ablation()

    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    table = summary_table(df)
    print(table.to_string(index=False))

    print("\n" + "=" * 70)
    print("STATISTICAL SIGNIFICANCE (Paired t-test: DFL vs MSE)")
    print("=" * 70)
    stats = statistical_tests(df)
    if not stats.empty:
        print(stats.to_string(index=False))
    else:
        print("  Not enough data for paired tests.")

    # Save tables
    table.to_csv(FIGURES_DIR / "summary_table.csv", index=False)
    if not stats.empty:
        stats.to_csv(FIGURES_DIR / "statistical_tests.csv", index=False)
    print(f"\nAll outputs saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
