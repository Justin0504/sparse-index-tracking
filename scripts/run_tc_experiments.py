"""Run transaction cost experiments: DFL vs Val-Tuned with TC in the QP.

This is the key experiment for ARR: with transaction costs in the optimization,
DFL should outperform grid search because the optimal covariance depends on
the current portfolio state, which a single α cannot capture.

Usage:
    .venv/bin/python scripts/run_tc_experiments.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import SP500DataLoader, FeatureBuilder, RollingWindowSplitter
from src.optimization.solver import solve_tracking_qp_tc, naive_index_weights
from src.evaluation.metrics import tracking_error, turnover, information_ratio
from src.utils import set_seed, get_logger

log = get_logger(__name__)

SEED = 42
TC_COSTS = [0.001, 0.005, 0.01]  # 10, 50, 100 bps per unit turnover
K_VALUES = [10, 20]
LOOKBACK = 63


def _compute_index_weights(prices: pd.DataFrame) -> np.ndarray:
    last_prices = prices.iloc[-1].values
    w = last_prices / last_prices.sum()
    return w.astype(np.float32)


def _get_monthly_rebal(dates: pd.DatetimeIndex) -> list[pd.Timestamp]:
    s = pd.Series(range(len(dates)), index=dates)
    return list(s.resample("ME").last().dropna().index)


def run_val_tuned_with_tc(prices, returns, folds, fb):
    """Val-tuned shrinkage baseline WITH transaction costs in the QP.

    Grid search α on validation set, but the QP includes TC penalty.
    The key question: can a single α handle the TC-TE trade-off?
    """
    log.info("\n" + "=" * 60)
    log.info("BASELINE: Val-Tuned Shrinkage + TC-aware QP")
    log.info("=" * 60)

    all_results = []
    min_offset = max(21, 63) + 1
    n_stocks = returns.shape[1]
    alphas_grid = np.linspace(0.01, 0.99, 50)

    for fold_info in folds:
        train_mask = (returns.index >= fold_info.train_start) & (returns.index <= fold_info.train_end)
        val_mask = (returns.index >= fold_info.val_start) & (returns.index <= fold_info.val_end)
        test_mask = (returns.index >= fold_info.test_start) & (returns.index <= fold_info.test_end)

        ret_train = returns.loc[train_mask]
        ret_val = returns.loc[val_mask]
        ret_test = returns.loc[test_mask]

        train_prices = prices.loc[prices.index.isin(ret_train.index)]
        w_index = _compute_index_weights(train_prices)

        rebal_val = [d for d in _get_monthly_rebal(ret_val.index)
                     if d in returns.index and returns.index.get_loc(d) >= min_offset]
        rebal_test = [d for d in _get_monthly_rebal(ret_test.index)
                      if d in returns.index and returns.index.get_loc(d) >= min_offset]

        if not rebal_val or not rebal_test:
            continue

        for K in K_VALUES:
            for tc in TC_COSTS:
                log.info(f"\nFold {fold_info.fold} K={K} tc={tc}: Grid searching α...")

                # Grid search on validation set with TC
                best_alpha = 0.5
                best_val_cost = float("inf")
                forward_window = 21

                for alpha in alphas_grid:
                    val_costs = []
                    w_prev_val = None

                    for date in rebal_val:
                        loc = returns.index.get_loc(date)
                        if loc + forward_window > len(returns):
                            continue

                        cov_np = fb.compute_realized_covariance(returns, date, lookback=LOOKBACK)
                        mu = np.trace(cov_np) / n_stocks
                        cov_shrunk = (1 - alpha) * cov_np + alpha * mu * np.eye(n_stocks)

                        # Annualize covariance so TC cost is on the same scale
                        cov_annual = 252.0 * cov_shrunk
                        try:
                            w_opt = solve_tracking_qp_tc(cov_annual, w_index, K=K,
                                                         w_prev=w_prev_val, tc_cost=tc)
                        except Exception:
                            continue

                        # Net cost = realized TE + TC * turnover
                        future_ret = returns.iloc[loc: loc + forward_window].values
                        diff = future_ret @ w_opt - future_ret @ w_index
                        te_proxy = 252.0 * np.mean(diff ** 2)

                        to = np.sum(np.abs(w_opt - w_prev_val)) / 2 if w_prev_val is not None else 0
                        net = te_proxy + tc * to
                        val_costs.append(net)
                        w_prev_val = w_opt

                    if val_costs and np.mean(val_costs) < best_val_cost:
                        best_val_cost = np.mean(val_costs)
                        best_alpha = alpha

                log.info(f"  Best α={best_alpha:.3f} (val net cost={best_val_cost:.6f})")

                # Evaluate on test with best alpha + TC
                weights_history = []
                portfolio_rets = []
                index_rets = []

                for i, date in enumerate(rebal_test):
                    loc = returns.index.get_loc(date)
                    next_loc = (returns.index.get_loc(rebal_test[i + 1])
                                if i + 1 < len(rebal_test) else len(returns))
                    if next_loc <= loc:
                        continue

                    cov_np = fb.compute_realized_covariance(returns, date, lookback=LOOKBACK)
                    mu = np.trace(cov_np) / n_stocks
                    cov_shrunk = (1 - best_alpha) * cov_np + best_alpha * mu * np.eye(n_stocks)
                    cov_annual = 252.0 * cov_shrunk

                    w_prev_np = weights_history[-1] if weights_history else None
                    try:
                        w_opt = solve_tracking_qp_tc(cov_annual, w_index, K=K,
                                                     w_prev=w_prev_np, tc_cost=tc)
                    except Exception:
                        w_opt = naive_index_weights(w_index, K)

                    weights_history.append(w_opt)
                    holding_rets = returns.iloc[loc:next_loc].values
                    portfolio_rets.extend((holding_rets @ w_opt).tolist())
                    index_rets.extend((holding_rets @ w_index).tolist())

                if not portfolio_rets:
                    continue

                pr = np.array(portfolio_rets)
                ir_arr = np.array(index_rets)
                te = tracking_error(pr, ir_arr, 252)
                avg_to = (float(np.mean([turnover(weights_history[j], weights_history[j + 1])
                                         for j in range(len(weights_history) - 1)]))
                          if len(weights_history) > 1 else 0.0)

                result = {
                    "fold": fold_info.fold,
                    "K": K,
                    "tc_cost": tc,
                    "model": "val_tuned_tc",
                    "best_alpha": float(best_alpha),
                    "tracking_error": te,
                    "avg_turnover": avg_to,
                    "net_cost": te + tc * avg_to,
                    "test_start": str(fold_info.test_start.date()),
                    "test_end": str(fold_info.test_end.date()),
                }
                all_results.append(result)
                log.info(
                    f"Fold {fold_info.fold} K={K} tc={tc} | TE={te:.4f} | "
                    f"TO={avg_to:.4f} | Net={te + tc * avg_to:.4f} | α={best_alpha:.3f}"
                )

    return all_results


def run_dfl_with_tc():
    """Run DFL experiments with TC via hydra config overrides."""
    results = {}
    for K in K_VALUES:
        for tc in TC_COSTS:
            for loss in ["mse", "task"]:
                label = f"neural_{loss}_K{K}_tc{tc}"
                log.info(f"\n>>> Running {label}...")
                cmd = [
                    sys.executable, "-m", "src.main",
                    "model.name=neural",
                    f"training.loss={loss}",
                    f"optimization.sparsity_levels=[{K}]",
                    f"optimization.tc_cost={tc}",
                    "training.device=cpu",
                    "+run_naive_baseline=false",
                ]
                subprocess.run(cmd, check=True,
                               cwd=str(Path(__file__).resolve().parent.parent))
                results[label] = "done"
    return results


def main():
    set_seed(SEED)

    # Load data
    loader = SP500DataLoader(
        start_date="2006-01-01", end_date="2025-12-31",
        top_n=100, cache_dir="data/cache",
    )
    prices, index_prices, tickers = loader.load()
    fb = FeatureBuilder(vol_window=21, momentum_window=63)
    returns = fb.compute_returns(prices)
    splitter = RollingWindowSplitter(train_years=3, val_months=6, test_years=1, step_years=1)
    folds = splitter.split(returns.index)

    log.info(f"Loaded {len(tickers)} stocks, {len(folds)} folds")

    out_dir = Path("results/tc_experiments")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Val-tuned baseline with TC
    vt_results = run_val_tuned_with_tc(prices, returns, folds, fb)
    with open(out_dir / "val_tuned_tc_results.json", "w") as f:
        json.dump({"results": vt_results}, f, indent=2)
    log.info(f"\nVal-tuned TC results saved ({len(vt_results)} entries)")

    # 2. DFL with TC (via hydra)
    log.info("\n>>> Running DFL TC experiments via hydra...")
    run_dfl_with_tc()

    # Summary
    log.info("\n" + "=" * 60)
    log.info("SUMMARY: Val-Tuned + TC")
    log.info("=" * 60)
    for K in K_VALUES:
        for tc in TC_COSTS:
            entries = [r for r in vt_results if r["K"] == K and r["tc_cost"] == tc]
            if entries:
                mean_te = np.mean([r["tracking_error"] for r in entries])
                mean_to = np.mean([r["avg_turnover"] for r in entries])
                mean_net = np.mean([r["net_cost"] for r in entries])
                log.info(f"K={K} tc={tc}: TE={mean_te:.4f} TO={mean_to:.4f} Net={mean_net:.4f}")


if __name__ == "__main__":
    main()
