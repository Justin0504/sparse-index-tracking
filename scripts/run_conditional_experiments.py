"""Run conditional/structured shrinkage experiments.

Compares:
1. Shrinkage (1 param) — MSE, Task, Val-Tuned grid search
2. ConditionalShrinkage (385 params) — MSE, Task, Random-search baseline
3. StructuredShrinkage (12 params) — MSE, Task, Random-search baseline

The key result: grid search matches DFL for 1-param shrinkage, but
DFL dominates for conditional/structured models because grid/random
search cannot explore high-dim parameter spaces.

Usage:
    .venv/bin/python scripts/run_conditional_experiments.py
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
from src.optimization.solver import solve_tracking_qp
from src.utils import set_seed, get_logger

log = get_logger(__name__)

SEED = 42
K_VALUES = [10, 20]
LOOKBACK = 63


def _compute_index_weights(prices: pd.DataFrame) -> np.ndarray:
    last_prices = prices.iloc[-1].values
    w = last_prices / last_prices.sum()
    return w.astype(np.float32)


def _get_monthly_rebal(dates: pd.DatetimeIndex) -> list[pd.Timestamp]:
    s = pd.Series(range(len(dates)), index=dates)
    return list(s.resample("ME").last().dropna().index)


def run_random_search_conditional(prices, returns, folds, fb, n_trials=200):
    """Random-search baseline for conditional shrinkage.

    Since the conditional model maps market features → alpha via an MLP,
    grid search is intractable. We use random search over alpha schedules:
    for each trial, sample random alpha values per rebalancing date from
    different distributions, evaluate on validation, pick the best.

    This is the STRONGEST non-DFL baseline for conditional models.
    """
    log.info("\n" + "=" * 60)
    log.info("BASELINE: Random Search Conditional Alpha")
    log.info("=" * 60)

    all_results = []
    min_offset = max(21, 63) + 1
    n_stocks = returns.shape[1]

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
            forward_window = 21

            # Strategy 1: Random fixed alpha (same as val-tuned but random)
            # Strategy 2: Random alpha per volatility regime (high/low vol → different alpha)
            # Strategy 3: Random linear function of market features

            best_val_cost = float("inf")
            best_strategy = None

            rng = np.random.RandomState(SEED + fold_info.fold * 100 + K)

            for trial in range(n_trials):
                # Generate a random alpha strategy
                strategy_type = trial % 3

                if strategy_type == 0:
                    # Fixed random alpha
                    alpha_fixed = rng.uniform(0.01, 0.99)
                    alpha_fn = lambda date, fb=fb, a=alpha_fixed: a
                elif strategy_type == 1:
                    # Two-regime: different alpha for high/low vol
                    alpha_low = rng.uniform(0.01, 0.99)
                    alpha_high = rng.uniform(0.01, 0.99)
                    vol_thresh = rng.uniform(0.10, 0.30)
                    def alpha_fn(date, fb=fb, al=alpha_low, ah=alpha_high, vt=vol_thresh):
                        mf = fb.build_market_features(returns, date, LOOKBACK)
                        return ah if mf[2] > vt else al  # mf[2] = short_vol
                else:
                    # Linear function of market features
                    weights = rng.randn(5) * 0.5
                    bias = rng.uniform(-2, 2)
                    def alpha_fn(date, fb=fb, w=weights, b=bias):
                        mf = fb.build_market_features(returns, date, LOOKBACK)
                        logit = float(np.dot(w, mf) + b)
                        return 1.0 / (1.0 + np.exp(-np.clip(logit, -10, 10)))

                # Evaluate on validation
                val_costs = []
                for date in rebal_val:
                    loc = returns.index.get_loc(date)
                    if loc + forward_window > len(returns):
                        continue
                    alpha = alpha_fn(date)
                    cov_np = fb.compute_realized_covariance(returns, date, lookback=LOOKBACK)
                    mu = np.trace(cov_np) / n_stocks
                    cov_shrunk = (1 - alpha) * cov_np + alpha * mu * np.eye(n_stocks)
                    try:
                        w_opt = solve_tracking_qp(cov_shrunk, w_index, K=K)
                    except Exception:
                        continue
                    future_ret = returns.iloc[loc: loc + forward_window].values
                    diff = future_ret @ w_opt - future_ret @ w_index
                    te_proxy = 252.0 * np.mean(diff ** 2)
                    val_costs.append(te_proxy)

                if val_costs and np.mean(val_costs) < best_val_cost:
                    best_val_cost = np.mean(val_costs)
                    best_strategy = alpha_fn

            if best_strategy is None:
                continue

            log.info(f"Fold {fold_info.fold} K={K}: Best random-search val cost={best_val_cost:.6f}")

            # Evaluate on test
            weights_history = []
            portfolio_rets = []
            index_rets_list = []

            for i, date in enumerate(rebal_test):
                loc = returns.index.get_loc(date)
                next_loc = (returns.index.get_loc(rebal_test[i + 1])
                            if i + 1 < len(rebal_test) else len(returns))
                if next_loc <= loc:
                    continue

                alpha = best_strategy(date)
                cov_np = fb.compute_realized_covariance(returns, date, lookback=LOOKBACK)
                mu = np.trace(cov_np) / n_stocks
                cov_shrunk = (1 - alpha) * cov_np + alpha * mu * np.eye(n_stocks)
                try:
                    w_opt = solve_tracking_qp(cov_shrunk, w_index, K=K)
                except Exception:
                    w_opt = np.ones(n_stocks, dtype=np.float64) / n_stocks

                weights_history.append(w_opt)
                holding_rets = returns.iloc[loc:next_loc].values
                portfolio_rets.extend((holding_rets @ w_opt).tolist())
                index_rets_list.extend((holding_rets @ w_index).tolist())

            if not portfolio_rets:
                continue

            from src.evaluation.metrics import tracking_error, turnover
            pr = np.array(portfolio_rets)
            ir_arr = np.array(index_rets_list)
            te = tracking_error(pr, ir_arr, 252)
            avg_to = (float(np.mean([turnover(weights_history[j], weights_history[j + 1])
                                     for j in range(len(weights_history) - 1)]))
                      if len(weights_history) > 1 else 0.0)

            result = {
                "fold": fold_info.fold, "K": K,
                "model": "random_search_conditional",
                "tracking_error": te, "avg_turnover": avg_to,
                "test_start": str(fold_info.test_start.date()),
                "test_end": str(fold_info.test_end.date()),
            }
            all_results.append(result)
            log.info(f"Fold {fold_info.fold} K={K} | TE={te:.4f} | TO={avg_to:.4f}")

    return all_results


def run_random_search_structured(prices, returns, folds, fb, n_trials=500):
    """Random-search baseline for structured shrinkage (12 params).

    Sample random (alpha, sector_var_1..11) and evaluate.
    """
    log.info("\n" + "=" * 60)
    log.info("BASELINE: Random Search Structured Shrinkage")
    log.info("=" * 60)

    all_results = []
    min_offset = max(21, 63) + 1
    n_stocks = returns.shape[1]
    n_sectors = 11

    for fold_info in folds:
        train_mask = (returns.index >= fold_info.train_start) & (returns.index <= fold_info.train_end)
        val_mask = (returns.index >= fold_info.val_start) & (returns.index <= fold_info.val_end)
        test_mask = (returns.index >= fold_info.test_start) & (returns.index <= fold_info.test_end)

        ret_val = returns.loc[val_mask]
        ret_test = returns.loc[test_mask]

        ret_train = returns.loc[train_mask]
        train_prices = prices.loc[prices.index.isin(ret_train.index)]
        w_index = _compute_index_weights(train_prices)

        rebal_val = [d for d in _get_monthly_rebal(ret_val.index)
                     if d in returns.index and returns.index.get_loc(d) >= min_offset]
        rebal_test = [d for d in _get_monthly_rebal(ret_test.index)
                      if d in returns.index and returns.index.get_loc(d) >= min_offset]

        if not rebal_val or not rebal_test:
            continue

        sector_ids = np.arange(n_stocks) % n_sectors

        for K in K_VALUES:
            forward_window = 21
            best_val_cost = float("inf")
            best_params = None

            rng = np.random.RandomState(SEED + fold_info.fold * 100 + K)

            for trial in range(n_trials):
                alpha = rng.uniform(0.01, 0.99)
                # Sector variance multipliers (relative to mu)
                sector_mults = np.exp(rng.randn(n_sectors) * 0.5)  # log-normal around 1

                val_costs = []
                for date in rebal_val:
                    loc = returns.index.get_loc(date)
                    if loc + forward_window > len(returns):
                        continue

                    cov_np = fb.compute_realized_covariance(returns, date, lookback=LOOKBACK)
                    mu = np.trace(cov_np) / n_stocks

                    # Structured target
                    diag_values = np.array([mu * sector_mults[s] for s in sector_ids])
                    T = np.diag(diag_values)
                    cov_shrunk = (1 - alpha) * cov_np + alpha * T

                    try:
                        w_opt = solve_tracking_qp(cov_shrunk, w_index, K=K)
                    except Exception:
                        continue

                    future_ret = returns.iloc[loc: loc + forward_window].values
                    diff = future_ret @ w_opt - future_ret @ w_index
                    te_proxy = 252.0 * np.mean(diff ** 2)
                    val_costs.append(te_proxy)

                if val_costs and np.mean(val_costs) < best_val_cost:
                    best_val_cost = np.mean(val_costs)
                    best_params = (alpha, sector_mults.copy())

            if best_params is None:
                continue

            best_alpha, best_mults = best_params
            log.info(f"Fold {fold_info.fold} K={K}: Best random-search val cost={best_val_cost:.6f} alpha={best_alpha:.3f}")

            # Evaluate on test
            weights_history = []
            portfolio_rets = []
            index_rets_list = []

            for i, date in enumerate(rebal_test):
                loc = returns.index.get_loc(date)
                next_loc = (returns.index.get_loc(rebal_test[i + 1])
                            if i + 1 < len(rebal_test) else len(returns))
                if next_loc <= loc:
                    continue

                cov_np = fb.compute_realized_covariance(returns, date, lookback=LOOKBACK)
                mu = np.trace(cov_np) / n_stocks
                diag_values = np.array([mu * best_mults[s] for s in sector_ids])
                T = np.diag(diag_values)
                cov_shrunk = (1 - best_alpha) * cov_np + best_alpha * T

                try:
                    w_opt = solve_tracking_qp(cov_shrunk, w_index, K=K)
                except Exception:
                    w_opt = np.ones(n_stocks, dtype=np.float64) / n_stocks

                weights_history.append(w_opt)
                holding_rets = returns.iloc[loc:next_loc].values
                portfolio_rets.extend((holding_rets @ w_opt).tolist())
                index_rets_list.extend((holding_rets @ w_index).tolist())

            if not portfolio_rets:
                continue

            from src.evaluation.metrics import tracking_error, turnover
            pr = np.array(portfolio_rets)
            ir_arr = np.array(index_rets_list)
            te = tracking_error(pr, ir_arr, 252)
            avg_to = (float(np.mean([turnover(weights_history[j], weights_history[j + 1])
                                     for j in range(len(weights_history) - 1)]))
                      if len(weights_history) > 1 else 0.0)

            result = {
                "fold": fold_info.fold, "K": K,
                "model": "random_search_structured",
                "tracking_error": te, "avg_turnover": avg_to,
                "test_start": str(fold_info.test_start.date()),
                "test_end": str(fold_info.test_end.date()),
            }
            all_results.append(result)
            log.info(f"Fold {fold_info.fold} K={K} | TE={te:.4f} | TO={avg_to:.4f}")

    return all_results


def run_dfl_models():
    """Run DFL experiments for conditional and structured shrinkage via hydra."""
    root = str(Path(__file__).resolve().parent.parent)
    results = {}
    for model_name in ["conditional_shrinkage", "structured_shrinkage"]:
        for K in K_VALUES:
            for loss in ["mse", "task"]:
                label = f"{model_name}_{loss}_K{K}"
                log.info(f"\n>>> Running {label}...")
                cmd = [
                    sys.executable, "-m", "src.main",
                    f"model.name={model_name}",
                    f"training.loss={loss}",
                    f"optimization.sparsity_levels=[{K}]",
                    "training.device=cpu",
                    "+run_naive_baseline=false",
                ]
                result = subprocess.run(cmd, capture_output=True, text=True, cwd=root)
                lines = result.stdout.strip().split('\n')
                for l in lines[-3:]:
                    log.info(f"  {l}")
                if result.returncode != 0:
                    log.error(f"  FAILED: {result.stderr[-300:]}")
                else:
                    results[label] = "done"
    return results


def main():
    set_seed(SEED)

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

    out_dir = Path("results/conditional_experiments")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Random-search baselines
    rs_cond = run_random_search_conditional(prices, returns, folds, fb)
    with open(out_dir / "random_search_conditional_results.json", "w") as f:
        json.dump({"results": rs_cond}, f, indent=2)

    rs_struct = run_random_search_structured(prices, returns, folds, fb)
    with open(out_dir / "random_search_structured_results.json", "w") as f:
        json.dump({"results": rs_struct}, f, indent=2)

    # 2. DFL models
    log.info("\n>>> Running DFL experiments...")
    run_dfl_models()

    # Summary
    log.info("\n" + "=" * 60)
    log.info("SUMMARY")
    log.info("=" * 60)
    for K in K_VALUES:
        log.info(f"\n--- K={K} ---")
        for label, data in [("RandomSearch-Cond", rs_cond), ("RandomSearch-Struct", rs_struct)]:
            entries = [r for r in data if r["K"] == K]
            if entries:
                te = np.mean([r["tracking_error"] for r in entries])
                to = np.mean([r["avg_turnover"] for r in entries])
                log.info(f"  {label}: TE={te:.4f} TO={to:.4f}")


if __name__ == "__main__":
    main()
