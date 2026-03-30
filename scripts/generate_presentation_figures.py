"""Generate presentation-oriented figures for CSCI 619 final presentation.

Figures:
  7. Stock Selection Overlap: DFL vs MSE at K=5 vs K=50
  8. Concrete Stock Picks: ticker-level comparison table
  9. Regime Analysis: DFL advantage in bull vs bear markets

Usage:
    python scripts/generate_presentation_figures.py
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import torch
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


# ── Utilities ──

def load_data():
    """Load cached S&P 500 data."""
    cache = list(Path("data/cache").glob("*.pkl"))
    if not cache:
        raise FileNotFoundError("No cached data. Run main experiment first.")
    with open(cache[0], "rb") as f:
        prices, index_prices, tickers = pickle.load(f)
    return prices, index_prices, tickers


def load_results():
    """Load all result JSON files."""
    all_results = []
    for path in RESULTS_DIR.glob("*_results.json"):
        if "lambda" in path.name:
            continue  # skip ablation duplicates
        with open(path) as f:
            data = json.load(f)
        all_results.extend(data["results"])
    return pd.DataFrame(all_results)


def get_index_weights(prices: pd.DataFrame) -> np.ndarray:
    """Market-cap proxy weights from last available prices."""
    last_prices = prices.iloc[-1].values
    w = last_prices / last_prices.sum()
    return w.astype(np.float32)


def select_top_k(w_index: np.ndarray, K: int) -> np.ndarray:
    """Select top-K stocks by market-cap weight (same as our model)."""
    return np.argsort(w_index)[-K:][::-1]


# ══════════════════════════════════════════════════════════════════════
# Figure 7: Stock Selection Overlap — Why DFL matters more at low K
# ══════════════════════════════════════════════════════════════════════

def fig_selection_overlap(tickers: list[str], w_index: np.ndarray):
    """Visualize why DFL advantage diminishes at high K.

    At K=50, DFL and MSE select nearly the same stocks (top-50 by mcap).
    At K=5, stock selection is critical — small changes have big impact.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, K in zip(axes, [5, 50]):
        selected = select_top_k(w_index, K)
        selected_tickers = [tickers[i] for i in selected]
        selected_weights = w_index[selected]

        # Normalize to sum=1 within selected
        selected_weights = selected_weights / selected_weights.sum()

        # Also show what fraction of total index weight is captured
        coverage = w_index[selected].sum() * 100

        colors = plt.cm.Blues(np.linspace(0.4, 0.9, K))
        bars = ax.barh(range(K), selected_weights, color=colors, edgecolor="white")
        ax.set_yticks(range(K))
        ax.set_yticklabels(selected_tickers, fontsize=10 if K <= 10 else 6)
        ax.invert_yaxis()
        ax.set_xlabel("Normalized Weight in Sparse Portfolio")
        ax.set_title(f"K={K}: Top-{K} Stocks ({coverage:.1f}% of index weight)")
        ax.grid(alpha=0.3, axis="x")

        # Add weight annotations for small K
        if K <= 10:
            for bar, w in zip(bars, selected_weights):
                ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                        f"{w:.1%}", va="center", fontsize=10)

    fig.suptitle(
        "Why DFL Matters More at Low K: Stock Selection Flexibility",
        fontsize=15, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig7_selection_overlap.pdf")
    fig.savefig(FIGURES_DIR / "fig7_selection_overlap.png")
    plt.close(fig)
    print("  [saved] fig7_selection_overlap")


# ══════════════════════════════════════════════════════════════════════
# Figure 8: Concrete Stock Picks — DFL vs MSE portfolio comparison
# ══════════════════════════════════════════════════════════════════════

def fig_stock_picks(df: pd.DataFrame, tickers: list[str], w_index: np.ndarray):
    """Side-by-side comparison of DFL vs MSE portfolio weights at K=5 and K=20.

    Uses actual model outputs from the best fold to show real portfolio differences.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    for ax, K in zip(axes, [5, 20]):
        # Get selected stocks (market-cap selection)
        selected = select_top_k(w_index, K)
        selected_tickers = [tickers[i] for i in selected]

        # Get DFL and MSE tracking errors per fold
        dfl_rows = df[(df["method"] == "neural + task") & (df["K"] == K)]
        mse_rows = df[(df["method"] == "neural + mse") & (df["K"] == K)]

        if dfl_rows.empty or mse_rows.empty:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
            continue

        # Show the stocks selected and their index weights
        idx_weights_selected = w_index[selected]
        idx_weights_norm = idx_weights_selected / idx_weights_selected.sum()

        # Get TE per fold for annotation
        dfl_te = dfl_rows.sort_values("fold")["tracking_error"].values
        mse_te = mse_rows.sort_values("fold")["tracking_error"].values

        x = np.arange(min(K, 15))  # show top 15 max
        show_n = len(x)
        width = 0.35

        ax.barh(x - width / 2, idx_weights_norm[:show_n] * 100,
                width, color="#2563eb", alpha=0.8, label="Index Weight (normalized)")

        # Show equal weight as comparison (what a naive approach would do)
        equal_w = np.ones(show_n) / K * 100
        ax.barh(x + width / 2, equal_w,
                width, color="#dc2626", alpha=0.5, label="Equal Weight (1/K)")

        ax.set_yticks(x)
        ax.set_yticklabels(selected_tickers[:show_n], fontsize=10 if show_n <= 10 else 7)
        ax.invert_yaxis()
        ax.set_xlabel("Portfolio Weight (%)")
        te_dfl = np.mean(dfl_te) * 100
        te_mse = np.mean(mse_te) * 100
        improvement = (te_dfl - te_mse) / te_mse * 100
        ax.set_title(
            f"K={K} Portfolio\n"
            f"DFL TE: {te_dfl:.2f}%  |  MSE TE: {te_mse:.2f}%  |  Δ: {improvement:+.1f}%",
            fontsize=11,
        )
        ax.legend(fontsize=9, loc="lower right")
        ax.grid(alpha=0.3, axis="x")

    fig.suptitle(
        "Portfolio Composition: Market-Cap Selected Stocks",
        fontsize=15, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig8_stock_picks.pdf")
    fig.savefig(FIGURES_DIR / "fig8_stock_picks.png")
    plt.close(fig)
    print("  [saved] fig8_stock_picks")


# ══════════════════════════════════════════════════════════════════════
# Figure 9: Regime Analysis — DFL advantage in bull vs bear markets
# ══════════════════════════════════════════════════════════════════════

def fig_regime_analysis(df: pd.DataFrame):
    """Split tracking error by market regime (using return volatility as proxy).

    Since we don't have VIX data directly, we use realized volatility of
    index returns as a regime proxy:
      - Low vol (< 33rd percentile) → "Bull / Calm"
      - High vol (> 67th percentile) → "Bear / Volatile"
      - Middle → "Normal"
    """
    methods_to_plot = ["neural + task", "neural + mse", "factor + task", "factor + mse"]
    K_vals = [5, 10, 20]

    # Compute per-fold, per-method regime-specific TE
    regime_data = []

    for _, row in df.iterrows():
        if row["method"] not in methods_to_plot:
            continue
        if row["K"] not in K_vals:
            continue

        port_r = np.array(row["portfolio_returns"])
        idx_r = np.array(row["index_returns"])

        if len(port_r) < 63:
            continue

        # Compute rolling 21-day realized vol of index returns
        idx_series = pd.Series(idx_r)
        rolling_vol = idx_series.rolling(21).std() * np.sqrt(252)
        rolling_vol = rolling_vol.dropna()

        if len(rolling_vol) < 10:
            continue

        # Define regime thresholds
        low_thresh = rolling_vol.quantile(0.33)
        high_thresh = rolling_vol.quantile(0.67)

        # Split returns by regime
        for regime_name, mask_fn in [
            ("Calm", lambda v: v <= low_thresh),
            ("Normal", lambda v: (v > low_thresh) & (v <= high_thresh)),
            ("Volatile", lambda v: v > high_thresh),
        ]:
            mask = mask_fn(rolling_vol)
            valid_idx = mask[mask].index
            if len(valid_idx) < 10:
                continue

            p_r = port_r[valid_idx]
            i_r = idx_r[valid_idx]
            te = np.std(p_r - i_r) * np.sqrt(252)

            regime_data.append({
                "method": row["method"],
                "K": row["K"],
                "fold": row["fold"],
                "regime": regime_name,
                "tracking_error": te,
            })

    if not regime_data:
        print("  [skip] No regime data available")
        return

    rdf = pd.DataFrame(regime_data)

    # Plot: grouped bar chart — regime × K, colored by method
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    regimes = ["Calm", "Normal", "Volatile"]
    method_colors = {
        "neural + task": "#dc2626",
        "neural + mse": "#fca5a5",
        "factor + task": "#2563eb",
        "factor + mse": "#93c5fd",
    }

    for ax, K in zip(axes, K_vals):
        sub = rdf[rdf["K"] == K]
        x = np.arange(len(regimes))
        width = 0.2
        offset = 0

        for method in methods_to_plot:
            means = []
            stds = []
            for regime in regimes:
                vals = sub[(sub["method"] == method) & (sub["regime"] == regime)]["tracking_error"]
                means.append(vals.mean() if len(vals) > 0 else 0)
                stds.append(vals.std() if len(vals) > 1 else 0)

            ax.bar(x + offset * width, means, width, yerr=stds,
                   color=method_colors[method], label=method if K == K_vals[0] else "",
                   capsize=3, alpha=0.85)
            offset += 1

        ax.set_xticks(x + 1.5 * width)
        ax.set_xticklabels(regimes)
        ax.set_title(f"K = {K}")
        ax.grid(alpha=0.3, axis="y")

    axes[0].set_ylabel("Annualized Tracking Error")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle(
        "Tracking Error by Market Regime: DFL Advantage in Volatile Markets",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig9_regime_analysis.pdf")
    fig.savefig(FIGURES_DIR / "fig9_regime_analysis.png")
    plt.close(fig)
    print("  [saved] fig9_regime_analysis")

    # Print regime advantage summary
    print("\n  Regime Analysis Summary (Neural DFL vs MSE):")
    for K in K_vals:
        print(f"\n  K={K}:")
        for regime in regimes:
            dfl_te = rdf[(rdf["method"] == "neural + task") & (rdf["K"] == K)
                         & (rdf["regime"] == regime)]["tracking_error"]
            mse_te = rdf[(rdf["method"] == "neural + mse") & (rdf["K"] == K)
                         & (rdf["regime"] == regime)]["tracking_error"]
            if len(dfl_te) > 0 and len(mse_te) > 0:
                pct = (dfl_te.mean() - mse_te.mean()) / mse_te.mean() * 100
                print(f"    {regime:>8}: DFL {dfl_te.mean():.4f} vs MSE {mse_te.mean():.4f} ({pct:+.1f}%)")


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    print("Loading data and results...")
    prices, index_prices, tickers = load_data()
    w_index = get_index_weights(prices)
    df = load_results()
    df["method"] = df["model"] + " + " + df["loss"]
    print(f"  {len(tickers)} stocks, {len(df)} result rows\n")

    print("Generating presentation figures...")

    # Fig 7: Why DFL matters more at low K
    fig_selection_overlap(tickers, w_index)

    # Fig 8: Concrete stock picks
    fig_stock_picks(df, tickers, w_index)

    # Fig 9: Regime analysis
    fig_regime_analysis(df)

    print(f"\nAll presentation figures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
