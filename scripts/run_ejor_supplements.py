#!/usr/bin/env python3
"""EJOR supplementary experiments:
1. Tracking-score selection vs market-cap selection baseline
2. Portfolio characteristics (HHI, effective N, max position)
3. Computational time comparison
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.features import FeatureBuilder
from src.data.loader import SP500DataLoader
from src.data.splitter import RollingWindowSplitter
from src.models import NeuralCovarianceModel, ShrinkageCovarianceModel
from src.optimization.solver import solve_tracking_qp, naive_index_weights, compute_tracking_scores_np
from src.evaluation.metrics import tracking_error, turnover
from src.training import Trainer
from src.utils import set_seed, get_logger

log = get_logger(__name__)

SEED = 42
LOOKBACK = 63
K_VALUES = [5, 10, 20, 30, 50]


def load_data():
    set_seed(SEED)
    fb = FeatureBuilder(vol_window=21, momentum_window=63,
                        use_sector=True, use_mcap_quintile=True)
    loader = SP500DataLoader(start_date="2006-01-01", end_date="2025-12-31", top_n=100)
    prices, index_prices, tickers = loader.load()
    returns = fb.compute_returns(prices)
    splitter = RollingWindowSplitter(train_years=3, val_months=6,
                                     test_years=1, step_years=1)
    folds = splitter.split(returns.index)
    return prices, returns, tickers, folds, fb


def compute_portfolio_characteristics(weights_list):
    """Compute HHI, effective N, max position from a list of weight vectors."""
    hhis, eff_ns, max_positions, n_active = [], [], [], []
    for w in weights_list:
        w_nonzero = w[w > 1e-6]
        hhi = float(np.sum(w_nonzero ** 2))
        hhis.append(hhi)
        eff_ns.append(1.0 / hhi if hhi > 0 else 0)
        max_positions.append(float(np.max(w)))
        n_active.append(int(np.sum(w > 1e-6)))
    return {
        "hhi": float(np.mean(hhis)),
        "effective_n": float(np.mean(eff_ns)),
        "max_position": float(np.mean(max_positions)),
        "n_active": float(np.mean(n_active)),
    }


# ═══════════════════════════════════════════
# Experiment 1: Selection method comparison
# ═══════════════════════════════════════════

def run_selection_comparison(prices, returns, folds, fb):
    """Compare market-cap vs tracking-score selection with Ledoit-Wolf covariance."""
    log.info("\n=== Selection Method Comparison ===")

    min_offset = 64
    all_results = []

    for fold_info in folds:
        train_mask = (returns.index >= fold_info.train_start) & (returns.index <= fold_info.train_end)
        test_mask = (returns.index >= fold_info.test_start) & (returns.index <= fold_info.test_end)

        ret_train = returns.loc[train_mask]
        ret_test = returns.loc[test_mask]

        train_prices = prices.loc[prices.index.isin(ret_train.index)]
        w_index = (train_prices.iloc[-1].values / train_prices.iloc[-1].values.sum()).astype(np.float32)

        s = pd.Series(range(len(ret_test.index)), index=ret_test.index)
        monthly_idx = s.resample("ME").last().dropna()
        rebal_test = []
        for d in monthly_idx.index:
            closest = ret_test.index[ret_test.index <= d]
            if len(closest) > 0 and returns.index.get_loc(closest[-1]) >= min_offset:
                rebal_test.append(closest[-1])

        if not rebal_test:
            continue

        for K in K_VALUES:
            for sel_method in ["market_cap", "tracking_score"]:
                weights_history = []
                portfolio_rets, index_rets = [], []

                for i, date in enumerate(rebal_test):
                    loc = returns.index.get_loc(date)
                    next_loc = (returns.index.get_loc(rebal_test[i + 1])
                                if i + 1 < len(rebal_test) else len(returns))
                    if next_loc <= loc:
                        continue

                    cov_lw, _ = fb.compute_ledoit_wolf_covariance(returns, date, lookback=LOOKBACK)

                    try:
                        w_opt = solve_tracking_qp(cov_lw, w_index, K=K, selection=sel_method)
                    except Exception:
                        w_opt = naive_index_weights(w_index, K)

                    weights_history.append(w_opt)
                    holding_rets = returns.iloc[loc:next_loc].values
                    portfolio_rets.extend((holding_rets @ w_opt).tolist())
                    index_rets.extend((holding_rets @ w_index).tolist())

                if not portfolio_rets:
                    continue

                pr = np.array(portfolio_rets)
                ir = np.array(index_rets)
                te = tracking_error(pr, ir, 252)
                avg_to = float(np.mean([turnover(weights_history[j], weights_history[j+1])
                                         for j in range(len(weights_history)-1)])) if len(weights_history) > 1 else 0.0
                chars = compute_portfolio_characteristics(weights_history)

                result = {
                    "fold": fold_info.fold, "K": K, "selection": sel_method,
                    "tracking_error": te, "avg_turnover": avg_to, **chars,
                }
                all_results.append(result)
                log.info(f"  Fold {fold_info.fold} K={K} sel={sel_method} | TE={te:.4f} HHI={chars['hhi']:.3f}")

    return all_results


# ═══════════════════════════════════════════
# Experiment 2: Portfolio characteristics for existing results
# ═══════════════════════════════════════════

def run_portfolio_characteristics(prices, returns, folds, fb):
    """Compute detailed portfolio characteristics for task-aware vs two-stage at K=20."""
    log.info("\n=== Portfolio Characteristics (K=20) ===")

    min_offset = 64
    n_stocks = returns.shape[1]
    sample_date = returns.index[min_offset + 1]
    n_features = fb.build_features(returns, sample_date).shape[1]
    all_results = []

    for fold_info in folds:
        train_mask = (returns.index >= fold_info.train_start) & (returns.index <= fold_info.train_end)
        val_mask = (returns.index >= fold_info.val_start) & (returns.index <= fold_info.val_end)
        test_mask = (returns.index >= fold_info.test_start) & (returns.index <= fold_info.test_end)

        ret_train = returns.loc[train_mask]
        ret_val = returns.loc[val_mask]
        ret_test = returns.loc[test_mask]

        train_prices = prices.loc[prices.index.isin(ret_train.index)]
        w_index = (train_prices.iloc[-1].values / train_prices.iloc[-1].values.sum()).astype(np.float32)

        def _get_rebal(ret_df):
            dates = [d for d in ret_df.index if d.weekday() == 4]
            return [d for d in dates if returns.index.get_loc(d) >= min_offset]

        rebal_train = _get_rebal(ret_train)
        rebal_val = _get_rebal(ret_val)

        s = pd.Series(range(len(ret_test.index)), index=ret_test.index)
        monthly_idx = s.resample("ME").last().dropna()
        rebal_test = []
        for d in monthly_idx.index:
            closest = ret_test.index[ret_test.index <= d]
            if len(closest) > 0 and returns.index.get_loc(closest[-1]) >= min_offset:
                rebal_test.append(closest[-1])

        if not rebal_train or not rebal_test:
            continue

        K = 20

        for loss_type in ["mse", "task"]:
            model = NeuralCovarianceModel(n_stocks=n_stocks, n_features=n_features,
                                           hidden_dims=[64, 64])
            trainer = Trainer(
                model=model, feature_builder=fb, loss_type=loss_type,
                lr=1e-3, weight_decay=1e-4, epochs=50, patience=10,
                device="cpu", sparsity_K=K, selection="market_cap",
            )
            trainer.train(
                returns_train=ret_train.values, returns_val=ret_val.values,
                rebalance_dates_train=rebal_train, rebalance_dates_val=rebal_val,
                returns_df_train=returns, returns_df_val=returns,
                w_index=w_index, lookback=LOOKBACK,
            )

            model.eval()
            weights_history = []
            portfolio_rets, index_rets = [], []

            for i, date in enumerate(rebal_test):
                loc = returns.index.get_loc(date)
                next_loc = (returns.index.get_loc(rebal_test[i + 1])
                            if i + 1 < len(rebal_test) else len(returns))
                if next_loc <= loc:
                    continue

                features_np = fb.build_features(returns, date)
                with torch.no_grad():
                    features_t = torch.tensor(features_np, dtype=torch.float32)
                    cov_pred = model(features_t).cpu().numpy()

                try:
                    w_opt = solve_tracking_qp(cov_pred, w_index, K=K, selection="market_cap")
                except Exception:
                    w_opt = np.ones(n_stocks) / n_stocks

                weights_history.append(w_opt)
                holding_rets = returns.iloc[loc:next_loc].values
                portfolio_rets.extend((holding_rets @ w_opt).tolist())
                index_rets.extend((holding_rets @ w_index).tolist())

            if not portfolio_rets:
                continue

            pr = np.array(portfolio_rets)
            ir_arr = np.array(index_rets)
            te = tracking_error(pr, ir_arr, 252)
            chars = compute_portfolio_characteristics(weights_history)

            result = {"fold": fold_info.fold, "K": K, "loss": loss_type,
                      "tracking_error": te, **chars}
            all_results.append(result)
            log.info(f"  Fold {fold_info.fold} {loss_type} | TE={te:.4f} HHI={chars['hhi']:.3f} "
                     f"maxW={chars['max_position']:.3f} effN={chars['effective_n']:.1f}")

    return all_results


# ═══════════════════════════════════════════
# Experiment 3: Computational time comparison
# ═══════════════════════════════════════════

def run_timing_comparison(prices, returns, folds, fb):
    """Time each method on one fold to produce a wall-clock comparison table."""
    log.info("\n=== Computational Time Comparison ===")

    min_offset = 64
    n_stocks = returns.shape[1]
    sample_date = returns.index[min_offset + 1]
    n_features = fb.build_features(returns, sample_date).shape[1]

    fold_info = folds[0]  # Use first fold for timing
    train_mask = (returns.index >= fold_info.train_start) & (returns.index <= fold_info.train_end)
    val_mask = (returns.index >= fold_info.val_start) & (returns.index <= fold_info.val_end)

    ret_train = returns.loc[train_mask]
    ret_val = returns.loc[val_mask]

    train_prices = prices.loc[prices.index.isin(ret_train.index)]
    w_index = (train_prices.iloc[-1].values / train_prices.iloc[-1].values.sum()).astype(np.float32)

    def _get_rebal(ret_df):
        dates = [d for d in ret_df.index if d.weekday() == 4]
        return [d for d in dates if returns.index.get_loc(d) >= min_offset]

    rebal_train = _get_rebal(ret_train)
    rebal_val = _get_rebal(ret_val)

    K = 20
    timings = {}

    # 1. Ledoit-Wolf (no learning)
    t0 = time.time()
    for date in rebal_train[:20]:
        cov_lw, _ = fb.compute_ledoit_wolf_covariance(returns, date, lookback=LOOKBACK)
        solve_tracking_qp(cov_lw, w_index, K=K)
    timings["ledoit_wolf"] = time.time() - t0
    log.info(f"  Ledoit-Wolf: {timings['ledoit_wolf']:.1f}s (20 dates)")

    # 2. Grid search (50 alphas on val)
    alphas_grid = np.linspace(0.01, 0.99, 50)
    s = pd.Series(range(len(ret_val.index)), index=ret_val.index)
    monthly_idx = s.resample("ME").last().dropna()
    rebal_val_monthly = []
    for d in monthly_idx.index:
        closest = ret_val.index[ret_val.index <= d]
        if len(closest) > 0 and returns.index.get_loc(closest[-1]) >= min_offset:
            rebal_val_monthly.append(closest[-1])

    t0 = time.time()
    for alpha in alphas_grid:
        for date in rebal_val_monthly:
            cov_np = fb.compute_realized_covariance(returns, date, lookback=LOOKBACK)
            mu = np.trace(cov_np) / n_stocks
            cov_shrunk = (1 - alpha) * cov_np + alpha * mu * np.eye(n_stocks)
            solve_tracking_qp(cov_shrunk, w_index, K=K)
    timings["grid_search"] = time.time() - t0
    log.info(f"  Grid search (50 alphas): {timings['grid_search']:.1f}s")

    # 3. Two-stage MSE training (neural)
    model_mse = NeuralCovarianceModel(n_stocks=n_stocks, n_features=n_features,
                                       hidden_dims=[64, 64])
    trainer_mse = Trainer(
        model=model_mse, feature_builder=fb, loss_type="mse",
        lr=1e-3, weight_decay=1e-4, epochs=50, patience=10,
        device="cpu", sparsity_K=K, selection="market_cap",
    )
    t0 = time.time()
    trainer_mse.train(
        returns_train=ret_train.values, returns_val=ret_val.values,
        rebalance_dates_train=rebal_train, rebalance_dates_val=rebal_val,
        returns_df_train=returns, returns_df_val=returns,
        w_index=w_index, lookback=LOOKBACK,
    )
    timings["neural_mse"] = time.time() - t0
    log.info(f"  Neural MSE training: {timings['neural_mse']:.1f}s")

    # 4. Task-aware training (neural)
    model_task = NeuralCovarianceModel(n_stocks=n_stocks, n_features=n_features,
                                        hidden_dims=[64, 64])
    trainer_task = Trainer(
        model=model_task, feature_builder=fb, loss_type="task",
        lr=1e-3, weight_decay=1e-4, epochs=50, patience=10,
        device="cpu", sparsity_K=K, selection="market_cap",
    )
    t0 = time.time()
    trainer_task.train(
        returns_train=ret_train.values, returns_val=ret_val.values,
        rebalance_dates_train=rebal_train, rebalance_dates_val=rebal_val,
        returns_df_train=returns, returns_df_val=returns,
        w_index=w_index, lookback=LOOKBACK,
    )
    timings["neural_task"] = time.time() - t0
    log.info(f"  Neural task-aware training: {timings['neural_task']:.1f}s")

    return timings


def main():
    prices, returns, tickers, folds, fb = load_data()
    log.info(f"Loaded {len(tickers)} stocks, {len(folds)} folds")

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    # 1. Selection comparison
    sel_results = run_selection_comparison(prices, returns, folds, fb)
    with open(out_dir / "selection_comparison.json", "w") as f:
        json.dump({"results": sel_results}, f, indent=2)

    # Print selection summary
    print("\n=== Selection Method Comparison (Ledoit-Wolf + QP) ===")
    print(f"{'K':>3} {'MarketCap TE':>12} {'TrackScore TE':>14} {'MC HHI':>8} {'TS HHI':>8}")
    for K in K_VALUES:
        mc = [r for r in sel_results if r["K"] == K and r["selection"] == "market_cap"]
        ts = [r for r in sel_results if r["K"] == K and r["selection"] == "tracking_score"]
        if mc and ts:
            mc_te = np.mean([r["tracking_error"] for r in mc])
            ts_te = np.mean([r["tracking_error"] for r in ts])
            mc_hhi = np.mean([r["hhi"] for r in mc])
            ts_hhi = np.mean([r["hhi"] for r in ts])
            print(f"{K:3d} {mc_te:12.4f} {ts_te:14.4f} {mc_hhi:8.3f} {ts_hhi:8.3f}")

    # 2. Portfolio characteristics
    char_results = run_portfolio_characteristics(prices, returns, folds, fb)
    with open(out_dir / "portfolio_characteristics.json", "w") as f:
        json.dump({"results": char_results}, f, indent=2)

    # Print characteristics summary
    print("\n=== Portfolio Characteristics (Neural, K=20) ===")
    print(f"{'Method':>12} {'TE':>8} {'HHI':>8} {'Eff N':>8} {'MaxW':>8} {'NActive':>8}")
    for loss in ["mse", "task"]:
        rs = [r for r in char_results if r["loss"] == loss]
        if rs:
            label = "Two-Stage" if loss == "mse" else "Task-Aware"
            print(f"{label:>12} {np.mean([r['tracking_error'] for r in rs]):8.4f} "
                  f"{np.mean([r['hhi'] for r in rs]):8.3f} "
                  f"{np.mean([r['effective_n'] for r in rs]):8.1f} "
                  f"{np.mean([r['max_position'] for r in rs]):8.3f} "
                  f"{np.mean([r['n_active'] for r in rs]):8.1f}")

    # 3. Timing comparison
    timings = run_timing_comparison(prices, returns, folds, fb)
    with open(out_dir / "timing_comparison.json", "w") as f:
        json.dump(timings, f, indent=2)

    print("\n=== Computational Time (1 fold, K=20, N=97) ===")
    print(f"{'Method':>25} {'Time (s)':>10}")
    for method, t in timings.items():
        print(f"{method:>25} {t:10.1f}")


if __name__ == "__main__":
    main()
