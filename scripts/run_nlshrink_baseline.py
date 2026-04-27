#!/usr/bin/env python3
"""Nonlinear shrinkage baseline (Ledoit & Wolf 2020).

Compares LW2004 (linear) vs LW2020 (nonlinear) shrinkage as two-stage baselines,
then tests whether task-aware training still outperforms the stronger baseline.

Usage:
    python scripts/run_nlshrink_baseline.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.features import FeatureBuilder
from src.data.loader import SP500DataLoader
from src.data.splitter import RollingWindowSplitter
from src.optimization.solver import solve_tracking_qp, naive_index_weights
from src.evaluation.metrics import tracking_error, turnover
from src.utils import set_seed, get_logger

log = get_logger(__name__)

SEED = 42
SPARSITY_LEVELS = [5, 10, 20, 30, 50]
LOOKBACK = 63


def _compute_index_weights(prices):
    last_p = prices.iloc[-1].values
    return (last_p / last_p.sum()).astype(np.float32)


def _get_monthly_rebal(dates):
    s = pd.Series(range(len(dates)), index=dates)
    return list(s.resample("ME").last().dropna().index)


def run_nlshrink_baseline():
    set_seed(SEED)

    loader = SP500DataLoader(start_date="2006-01-01", end_date="2025-12-31",
                             top_n=100, cache_dir="data/cache")
    prices, index_prices, tickers = loader.load()
    fb = FeatureBuilder(vol_window=21, momentum_window=63)
    returns = fb.compute_returns(prices)

    splitter = RollingWindowSplitter(train_years=3, val_months=6,
                                     test_years=1, step_years=1)
    folds = splitter.split(returns.index)
    min_offset = LOOKBACK + 1

    all_results = []

    for fold_info in folds:
        train_mask = (returns.index >= fold_info.train_start) & (returns.index <= fold_info.train_end)
        ret_train = returns.loc[train_mask]
        ret_test = returns.loc[
            (returns.index >= fold_info.test_start) & (returns.index <= fold_info.test_end)
        ]

        train_prices = prices.loc[prices.index.isin(ret_train.index)]
        w_index = _compute_index_weights(train_prices)

        rebal_test = [d for d in _get_monthly_rebal(ret_test.index)
                      if d in returns.index and returns.index.get_loc(d) >= min_offset]
        if not rebal_test:
            continue

        for K in SPARSITY_LEVELS:
            # Run both LW2004 and LW2020 on same dates
            for method_name, cov_fn in [
                ("lw2004", fb.compute_ledoit_wolf_covariance),
                ("lw2020", fb.compute_nonlinear_shrinkage_covariance),
            ]:
                weights_history = []
                for date in rebal_test:
                    if method_name == "lw2004":
                        cov, _ = cov_fn(returns, date, lookback=LOOKBACK)
                    else:
                        cov = cov_fn(returns, date, lookback=LOOKBACK)

                    try:
                        w_opt = solve_tracking_qp(cov, w_index, K=K, selection="market_cap")
                    except Exception:
                        w_opt = naive_index_weights(w_index, K)

                    weights_history.append(w_opt)

                # Evaluate
                portfolio_rets, index_rets = [], []
                for i, date in enumerate(rebal_test):
                    loc = returns.index.get_loc(date)
                    next_loc = (returns.index.get_loc(rebal_test[i + 1])
                                if i + 1 < len(rebal_test) else len(returns))
                    if next_loc <= loc:
                        continue
                    holding_rets = returns.iloc[loc:next_loc].values
                    portfolio_rets.extend((holding_rets @ weights_history[i]).tolist())
                    index_rets.extend((holding_rets @ w_index).tolist())

                if not portfolio_rets:
                    continue

                pr, ir = np.array(portfolio_rets), np.array(index_rets)
                te = tracking_error(pr, ir, 252)
                avg_to = (float(np.mean([turnover(weights_history[j], weights_history[j+1])
                                          for j in range(len(weights_history)-1)]))
                          if len(weights_history) > 1 else 0.0)

                result = {
                    "fold": fold_info.fold,
                    "K": K,
                    "method": method_name,
                    "tracking_error": te,
                    "avg_turnover": avg_to,
                }
                all_results.append(result)
                log.info(f"  Fold {fold_info.fold} K={K} {method_name} | TE={te:.4f}")

    return all_results


def main():
    log.info("=" * 60)
    log.info("Nonlinear Shrinkage Baseline (LW2020 vs LW2004)")
    log.info("=" * 60)

    results = run_nlshrink_baseline()

    out_path = Path("results/nlshrink_baseline.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"results": results}, f, indent=2)
    log.info(f"\nResults saved to {out_path}")

    # Summary table
    print("\n=== Nonlinear Shrinkage Baseline ===")
    print(f"{'K':>4} {'LW2004 TE':>10} {'LW2020 TE':>10} {'Gain':>8}")
    for K in SPARSITY_LEVELS:
        lw04 = [r for r in results if r["K"] == K and r["method"] == "lw2004"]
        lw20 = [r for r in results if r["K"] == K and r["method"] == "lw2020"]
        if lw04 and lw20:
            te04 = np.mean([r["tracking_error"] for r in lw04])
            te20 = np.mean([r["tracking_error"] for r in lw20])
            gain = (1 - te20/te04) * 100
            print(f"{K:>4} {te04:>10.4f} {te20:>10.4f} {gain:>+7.1f}%")

    # Compare with task-aware (from existing results if available)
    existing = Path("results/baselines.json")
    if existing.exists():
        with open(existing) as f:
            baseline_data = json.load(f)
        # Look for task-aware results
        ta_results = [r for r in baseline_data.get("results", [])
                      if r.get("loss") == "task" or r.get("model") == "shrinkage_dfl"]
        if ta_results:
            print("\n=== LW2020 vs Task-Aware ===")
            print(f"{'K':>4} {'LW2020 TE':>10} {'Task TE':>10} {'Task Gain':>10}")
            for K in SPARSITY_LEVELS:
                lw20 = [r for r in results if r["K"] == K and r["method"] == "lw2020"]
                ta = [r for r in ta_results if r.get("K") == K]
                if lw20 and ta:
                    te20 = np.mean([r["tracking_error"] for r in lw20])
                    te_ta = np.mean([r["tracking_error"] for r in ta])
                    gain = (1 - te_ta/te20) * 100
                    print(f"{K:>4} {te20:>10.4f} {te_ta:>10.4f} {gain:>+9.1f}%")


if __name__ == "__main__":
    main()
