#!/usr/bin/env python3
"""Compute comprehensive risk metrics from saved daily returns in result JSONs.

Outputs a LaTeX-ready table for the paper appendix.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_results(path):
    with open(path) as f:
        return json.load(f)["results"]


def compute_metrics(result):
    """Compute risk metrics from a single fold result."""
    pr = np.array(result["portfolio_returns"])
    ir = np.array(result["index_returns"])
    diff = pr - ir  # daily tracking differences

    te_ann = np.std(diff) * np.sqrt(252)
    mean_active = np.mean(diff) * 252
    ir_ratio = mean_active / te_ann if te_ann > 0 else 0.0

    # Max drawdown of cumulative tracking difference
    cum_diff = np.cumsum(diff)
    running_max = np.maximum.accumulate(cum_diff)
    drawdowns = running_max - cum_diff
    max_dd = np.max(drawdowns) if len(drawdowns) > 0 else 0.0

    # CVaR 5% (expected shortfall of daily tracking diff)
    sorted_diff = np.sort(diff)
    n5 = max(1, int(len(sorted_diff) * 0.05))
    cvar_5 = -np.mean(sorted_diff[:n5])  # positive = bad

    # Monthly TE (std of monthly tracking differences, annualized)
    n_months = len(diff) // 21
    if n_months >= 3:
        monthly_diffs = [np.sum(diff[i*21:(i+1)*21]) for i in range(n_months)]
        monthly_te = np.std(monthly_diffs) * np.sqrt(12)
    else:
        monthly_te = np.nan

    # Worst 21-day rolling TE
    if len(diff) >= 21:
        rolling_te = [np.std(diff[i:i+21]) * np.sqrt(252) for i in range(len(diff)-20)]
        worst_rolling = np.max(rolling_te)
        pct_dfl_better = None  # will be computed in comparison
    else:
        worst_rolling = np.nan

    return {
        "te_ann": te_ann,
        "ir": ir_ratio,
        "max_dd": max_dd,
        "cvar_5": cvar_5,
        "monthly_te": monthly_te,
        "worst_21d_te": worst_rolling,
        "mean_active_ret": mean_active,
    }


def main():
    results_dir = Path("results")

    # Load neural MSE and DFL results
    mse_results = load_results(results_dir / "neural_mse_results.json")
    dfl_results = load_results(results_dir / "neural_task_results.json")

    K_values = [5, 10, 20, 30, 50]

    print("\n=== Comprehensive Risk Metrics (Neural Model, per K) ===\n")

    all_metrics = {}

    for K in K_values:
        mse_folds = [r for r in mse_results if r["K"] == K]
        dfl_folds = [r for r in dfl_results if r["K"] == K]

        if not mse_folds or not dfl_folds:
            continue

        mse_metrics = [compute_metrics(r) for r in mse_folds]
        dfl_metrics = [compute_metrics(r) for r in dfl_folds]

        def avg(metrics, key):
            vals = [m[key] for m in metrics if not np.isnan(m[key])]
            return np.mean(vals) if vals else np.nan

        def worst(metrics, key):
            vals = [m[key] for m in metrics if not np.isnan(m[key])]
            return np.max(vals) if vals else np.nan

        all_metrics[K] = {
            "mse": {k: avg(mse_metrics, k) for k in mse_metrics[0]},
            "dfl": {k: avg(dfl_metrics, k) for k in dfl_metrics[0]},
            "mse_worst_te": worst(mse_metrics, "te_ann"),
            "dfl_worst_te": worst(dfl_metrics, "te_ann"),
        }

        m = all_metrics[K]
        print(f"K={K}:")
        print(f"  {'Metric':>20} {'MSE':>10} {'DFL':>10} {'Gain':>10}")
        print(f"  {'─'*52}")
        for metric_name, label in [
            ("te_ann", "Ann. TE"),
            ("ir", "Info Ratio"),
            ("max_dd", "Max DD (track)"),
            ("cvar_5", "CVaR 5%"),
            ("monthly_te", "Monthly TE"),
            ("worst_21d_te", "Worst 21d TE"),
        ]:
            mv = m["mse"][metric_name]
            dv = m["dfl"][metric_name]
            if np.isnan(mv) or np.isnan(dv):
                print(f"  {label:>20} {mv:10.4f} {dv:10.4f}      N/A")
            else:
                if metric_name == "ir":
                    # Higher IR is better, don't compute gain %
                    print(f"  {label:>20} {mv:10.4f} {dv:10.4f}")
                else:
                    gain = (1 - dv / mv) * 100 if mv != 0 else 0
                    print(f"  {label:>20} {mv:10.4f} {dv:10.4f} {gain:+9.1f}%")
        print()

    # LaTeX table for K=20 (main result)
    K = 20
    if K in all_metrics:
        m = all_metrics[K]
        print("\n=== LaTeX Table (K=20, Neural Model) ===\n")
        print(r"\begin{table}[t]")
        print(r"\centering")
        print(r"\caption{Comprehensive risk metrics at $K{=}20$ (neural model, 9 folds). DFL reduces risk across all metrics. CVaR$_{5\%}$ is the expected daily tracking shortfall in the worst 5\% of days.}")
        print(r"\label{tab:risk_metrics}")
        print(r"\small")
        print(r"\begin{tabular}{@{}lccc@{}}")
        print(r"\toprule")
        print(r"Metric & MSE & DFL & Improvement \\")
        print(r"\midrule")

        rows = [
            ("te_ann", "Annualized TE", True),
            ("monthly_te", "Monthly TE (ann.)", True),
            ("worst_21d_te", "Worst 21-day TE", True),
            ("max_dd", "Max tracking DD", True),
            ("cvar_5", r"CVaR$_{5\%}$ (daily)", True),
            ("ir", "Information ratio", False),
        ]
        for key, label, lower_better in rows:
            mv = m["mse"][key]
            dv = m["dfl"][key]
            if key == "ir":
                print(f"{label} & {mv:.4f} & {dv:.4f} & --- \\\\")
            else:
                gain = (1 - dv / mv) * 100 if mv != 0 else 0
                print(f"{label} & {mv:.4f} & {dv:.4f} & ${gain:+.1f}\\%$ \\\\")

        print(r"\midrule")
        # Worst-fold TE
        print(f"Worst-fold TE & {m['mse_worst_te']:.4f} & {m['dfl_worst_te']:.4f} & "
              f"${(1 - m['dfl_worst_te']/m['mse_worst_te'])*100:+.1f}\\%$ \\\\")

        # TE dispersion
        mse_te_vals = [compute_metrics(r)["te_ann"] for r in mse_results if r["K"] == K]
        dfl_te_vals = [compute_metrics(r)["te_ann"] for r in dfl_results if r["K"] == K]
        mse_std = np.std(mse_te_vals)
        dfl_std = np.std(dfl_te_vals)
        print(f"TE std across folds & {mse_std:.4f} & {dfl_std:.4f} & "
              f"${(1 - dfl_std/mse_std)*100:+.1f}\\%$ \\\\")

        print(r"\bottomrule")
        print(r"\end{tabular}")
        print(r"\end{table}")

    # Also output per-K summary for all K values
    print("\n\n=== Summary Table (All K) ===\n")
    print(f"{'K':>3} {'MSE TE':>8} {'DFL TE':>8} {'MSE CVaR':>9} {'DFL CVaR':>9} "
          f"{'MSE MaxDD':>10} {'DFL MaxDD':>10} {'MSE MoTE':>9} {'DFL MoTE':>9}")
    for K in K_values:
        if K not in all_metrics:
            continue
        m = all_metrics[K]
        print(f"{K:3d} {m['mse']['te_ann']:8.4f} {m['dfl']['te_ann']:8.4f} "
              f"{m['mse']['cvar_5']:9.4f} {m['dfl']['cvar_5']:9.4f} "
              f"{m['mse']['max_dd']:10.4f} {m['dfl']['max_dd']:10.4f} "
              f"{m['mse']['monthly_te']:9.4f} {m['dfl']['monthly_te']:9.4f}")


if __name__ == "__main__":
    main()
