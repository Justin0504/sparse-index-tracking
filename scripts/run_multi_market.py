#!/usr/bin/env python3
"""Multi-market experiment: validate K/N diagnostic across 4 global stock markets.

Markets: Nikkei 225, Euro Stoxx 50, ASX 200, Hang Seng
(S&P 500 and FTSE 100 already done in prior experiments.)
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.multi_universe_loader import load_universe
from src.data.features import FeatureBuilder
from src.data.splitter import RollingWindowSplitter
from src.models.shrinkage_model import ShrinkageCovarianceModel
from src.optimization.solver import solve_tracking_qp
from src.training import Trainer
from src.evaluation.metrics import tracking_error, turnover
from src.utils import set_seed, get_logger

log = get_logger(__name__)

MARKETS = ["nikkei225", "eurostoxx50", "asx200", "hangseng"]
K_VALUES = [5, 10, 20]


def run_experiment(prices, index_prices, tickers, K, loss_type, device="cpu"):
    n_stocks = len(tickers)
    set_seed(42)

    fb = FeatureBuilder(vol_window=21, momentum_window=63,
                        use_sector=True, use_mcap_quintile=True)
    returns = fb.compute_returns(prices)

    splitter = RollingWindowSplitter(train_years=2, val_months=6,
                                     test_years=1, step_years=1)
    folds = splitter.split(returns.index)
    log.info(f"N={n_stocks}, K={K}, loss={loss_type}: {len(folds)} folds")

    min_offset = 64
    all_results = []

    for fold_info in folds:
        set_seed(42 + fold_info.fold)

        ret_train = returns.loc[(returns.index >= fold_info.train_start) &
                                (returns.index <= fold_info.train_end)]
        ret_val = returns.loc[(returns.index >= fold_info.val_start) &
                              (returns.index <= fold_info.val_end)]
        ret_test = returns.loc[(returns.index >= fold_info.test_start) &
                               (returns.index <= fold_info.test_end)]

        if len(ret_train) < 100 or len(ret_test) < 20:
            continue

        train_prices = prices.loc[prices.index.isin(ret_train.index)]
        last_p = train_prices.iloc[-1].values
        w_index = (last_p / last_p.sum()).astype(np.float32)

        def _get_rebal(ret_df):
            return [d for d in ret_df.index if d.weekday() == 4
                    and returns.index.get_loc(d) >= min_offset]

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

        model = ShrinkageCovarianceModel(n_stocks=n_stocks)

        trainer = Trainer(
            model=model, feature_builder=fb, loss_type=loss_type,
            lr=1e-3, weight_decay=1e-4, epochs=50, patience=10,
            device=device, sparsity_K=K, selection="market_cap",
        )

        trainer.train(
            returns_train=ret_train.values,
            returns_val=ret_val.values,
            rebalance_dates_train=rebal_train,
            rebalance_dates_val=rebal_val,
            returns_df_train=returns,
            returns_df_val=returns,
            w_index=w_index,
        )

        model.eval()
        portfolio_rets, index_rets, weights_history = [], [], []

        for i, date in enumerate(rebal_test):
            loc = returns.index.get_loc(date)
            next_loc = (returns.index.get_loc(rebal_test[i + 1])
                        if i + 1 < len(rebal_test) else len(returns))
            if next_loc <= loc:
                continue

            features_np = fb.build_features(returns, date)
            with torch.no_grad():
                features_t = torch.tensor(features_np, dtype=torch.float32, device=device)
                sample_cov_np = fb.compute_realized_covariance(returns, date, 63)
                model.set_sample_covariance(
                    torch.tensor(sample_cov_np, dtype=torch.float32, device=device))
                cov_pred = model(features_t).cpu().numpy()

            try:
                w_opt = solve_tracking_qp(cov_pred, w_index, K=K, selection="market_cap")
            except Exception:
                w_opt = np.ones(n_stocks, dtype=np.float64) / n_stocks

            weights_history.append(w_opt)
            holding_rets = returns.iloc[loc:next_loc].values
            portfolio_rets.extend((holding_rets @ w_opt).tolist())
            index_rets.extend((holding_rets @ w_index).tolist())

        if not portfolio_rets:
            continue

        te = tracking_error(np.array(portfolio_rets), np.array(index_rets), 252)
        avg_to = float(np.mean([turnover(weights_history[j], weights_history[j + 1])
                                for j in range(len(weights_history) - 1)])) if len(weights_history) > 1 else 0.0

        result = {
            "fold": fold_info.fold, "K": K, "N": n_stocks,
            "model": "shrinkage", "loss": loss_type,
            "tracking_error": te, "avg_turnover": avg_to,
            "test_start": str(fold_info.test_start.date()),
            "test_end": str(fold_info.test_end.date()),
        }
        all_results.append(result)
        log.info(f"  Fold {fold_info.fold} | K={K} | {loss_type} | TE={te:.4f}")

    return all_results


def run_poet_baseline(prices, index_prices, tickers, K, device="cpu"):
    n_stocks = len(tickers)
    set_seed(42)

    fb = FeatureBuilder(vol_window=21, momentum_window=63,
                        use_sector=True, use_mcap_quintile=True)
    returns = fb.compute_returns(prices)

    splitter = RollingWindowSplitter(train_years=2, val_months=6,
                                     test_years=1, step_years=1)
    folds = splitter.split(returns.index)

    min_offset = 64
    all_results = []

    for fold_info in folds:
        ret_test = returns.loc[(returns.index >= fold_info.test_start) &
                               (returns.index <= fold_info.test_end)]
        if len(ret_test) < 20:
            continue

        train_prices = prices.loc[prices.index <= fold_info.train_end]
        last_p = train_prices.iloc[-1].values
        w_index = (last_p / last_p.sum()).astype(np.float32)

        s = pd.Series(range(len(ret_test.index)), index=ret_test.index)
        monthly_idx = s.resample("ME").last().dropna()
        rebal_test = []
        for d in monthly_idx.index:
            closest = ret_test.index[ret_test.index <= d]
            if len(closest) > 0 and returns.index.get_loc(closest[-1]) >= min_offset:
                rebal_test.append(closest[-1])

        if not rebal_test:
            continue

        portfolio_rets, index_rets, weights_history = [], [], []

        for i, date in enumerate(rebal_test):
            loc = returns.index.get_loc(date)
            next_loc = (returns.index.get_loc(rebal_test[i + 1])
                        if i + 1 < len(rebal_test) else len(returns))
            if next_loc <= loc:
                continue

            cov_poet = fb.compute_poet_covariance(returns, date, lookback=63)

            try:
                w_opt = solve_tracking_qp(cov_poet, w_index, K=K, selection="market_cap")
            except Exception:
                w_opt = np.ones(n_stocks, dtype=np.float64) / n_stocks

            weights_history.append(w_opt)
            holding_rets = returns.iloc[loc:next_loc].values
            portfolio_rets.extend((holding_rets @ w_opt).tolist())
            index_rets.extend((holding_rets @ w_index).tolist())

        if not portfolio_rets:
            continue

        te = tracking_error(np.array(portfolio_rets), np.array(index_rets), 252)
        avg_to = float(np.mean([turnover(weights_history[j], weights_history[j + 1])
                                for j in range(len(weights_history) - 1)])) if len(weights_history) > 1 else 0.0

        result = {
            "fold": fold_info.fold, "K": K, "N": n_stocks,
            "model": "POET", "loss": "N/A",
            "tracking_error": te, "avg_turnover": avg_to,
            "test_start": str(fold_info.test_start.date()),
            "test_end": str(fold_info.test_end.date()),
        }
        all_results.append(result)
        log.info(f"  POET Fold {fold_info.fold} | K={K} | TE={te:.4f}")

    return all_results


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    log.info(f"Device: {device}")

    grand_results = {}

    for market in MARKETS:
        log.info(f"\n{'='*60}\n=== {market.upper()} ===")
        try:
            prices, index_prices, tickers = load_universe(
                market, start_date="2015-01-01", end_date="2025-12-31", min_coverage=0.90
            )
        except Exception as e:
            log.warning(f"Failed to load {market}: {e}")
            continue

        n_stocks = len(tickers)
        log.info(f"{market}: N={n_stocks}, T={len(prices)}")
        market_results = []

        for K in K_VALUES:
            if K > n_stocks // 2:
                log.info(f"  Skipping K={K} (> N/2={n_stocks//2})")
                continue

            kn = K / n_stocks
            alignment = np.sqrt(2 * kn)
            log.info(f"\n  K={K}, K/N={kn:.3f}, alignment={alignment:.3f}")

            log.info(f"  POET baseline...")
            results = run_poet_baseline(prices, index_prices, tickers, K, device)
            for r in results:
                r["market"] = market
                r["alignment"] = round(alignment, 4)
            market_results.extend(results)

            for loss in ["mse", "task"]:
                log.info(f"  Shrinkage {loss}...")
                results = run_experiment(prices, index_prices, tickers, K, loss, device)
                for r in results:
                    r["market"] = market
                    r["alignment"] = round(alignment, 4)
                market_results.extend(results)

        out_dir = Path(f"results/multi_market/{market}")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "results.json"
        with open(out_path, "w") as f:
            json.dump({"results": market_results, "N": n_stocks, "market": market}, f, indent=2)
        log.info(f"Saved {len(market_results)} results to {out_path}")
        grand_results[market] = market_results

    # Summary
    log.info(f"\n{'='*60}\nSUMMARY")
    log.info(f"{'Market':<15} {'K':>3} {'N':>5} {'K/N':>6} {'MSE':>8} {'DFL':>8} {'POET':>8} {'Δ%':>6}")
    log.info("-" * 70)

    for market, results in grand_results.items():
        ks = sorted(set(r["K"] for r in results))
        n = results[0]["N"] if results else 0
        for K in ks:
            mse = [r["tracking_error"] for r in results if r["K"] == K and r["loss"] == "mse"]
            task = [r["tracking_error"] for r in results if r["K"] == K and r["loss"] == "task"]
            poet = [r["tracking_error"] for r in results if r["K"] == K and r["model"] == "POET"]
            mse_mean = np.mean(mse) if mse else float("nan")
            task_mean = np.mean(task) if task else float("nan")
            poet_mean = np.mean(poet) if poet else float("nan")
            delta = (mse_mean - task_mean) / mse_mean * 100 if mse and task and mse_mean > 0 else float("nan")
            log.info(f"{market:<15} {K:>3} {n:>5} {K/n:>6.3f} {mse_mean:>8.4f} {task_mean:>8.4f} {poet_mean:>8.4f} {delta:>+6.1f}")


if __name__ == "__main__":
    main()
