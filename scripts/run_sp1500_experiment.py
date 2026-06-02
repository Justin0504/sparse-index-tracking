#!/usr/bin/env python3
"""S&P 1500 large-scale experiment: shrinkage + structured, MSE vs DFL + POET baseline.

Validates the K/N diagnostic at extreme ratios (K/N ≈ 0.007-0.038).
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
from src.models.shrinkage_model import (
    ShrinkageCovarianceModel,
    StructuredShrinkageModel,
)
from src.optimization.solver import solve_tracking_qp
from src.training import Trainer
from src.evaluation.metrics import tracking_error, turnover
from src.utils import set_seed, get_logger

log = get_logger(__name__)


def run_experiment(prices, index_prices, tickers, K, loss_type, model_type="shrinkage",
                   device="cpu"):
    n_stocks = len(tickers)
    set_seed(42)

    fb = FeatureBuilder(vol_window=21, momentum_window=63,
                        use_sector=True, use_mcap_quintile=True)
    returns = fb.compute_returns(prices)

    splitter = RollingWindowSplitter(train_years=2, val_months=6,
                                     test_years=1, step_years=1)
    folds = splitter.split(returns.index)
    log.info(f"N={n_stocks}, K={K}, loss={loss_type}, model={model_type}: {len(folds)} folds")

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

        if model_type == "structured":
            model = StructuredShrinkageModel(n_stocks=n_stocks)
        else:
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
            "model": model_type, "loss": loss_type,
            "tracking_error": te, "avg_turnover": avg_to,
            "test_start": str(fold_info.test_start.date()),
            "test_end": str(fold_info.test_end.date()),
        }
        all_results.append(result)
        log.info(f"  Fold {fold_info.fold} | K={K} | {model_type} | {loss_type} | TE={te:.4f}")

    return all_results


def run_poet_baseline(prices, index_prices, tickers, K, device="cpu"):
    """Run POET baseline (no training, just estimate and solve)."""
    n_stocks = len(tickers)
    set_seed(42)

    fb = FeatureBuilder(vol_window=21, momentum_window=63,
                        use_sector=True, use_mcap_quintile=True)
    returns = fb.compute_returns(prices)

    splitter = RollingWindowSplitter(train_years=2, val_months=6,
                                     test_years=1, step_years=1)
    folds = splitter.split(returns.index)
    log.info(f"POET baseline: N={n_stocks}, K={K}: {len(folds)} folds")

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

    log.info("Loading S&P 1500 universe...")
    prices, index_prices, tickers = load_universe(
        "sp1500", start_date="2015-01-01", end_date="2025-12-31", min_coverage=0.90
    )
    n_stocks = len(tickers)
    log.info(f"S&P 1500 universe: N={n_stocks}")

    all_results = []

    for K in [10, 20, 50]:
        log.info(f"\n{'='*60}\nPOET baseline K={K}")
        results = run_poet_baseline(prices, index_prices, tickers, K, device)
        all_results.extend(results)

        for model_type in ["shrinkage", "structured"]:
            for loss in ["mse", "task"]:
                log.info(f"\n{'='*60}\nK={K}, model={model_type}, loss={loss}")
                results = run_experiment(prices, index_prices, tickers, K, loss,
                                         model_type=model_type, device=device)
                all_results.extend(results)

    out_dir = Path("results/sp1500")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "sp1500_results.json"
    with open(out_path, "w") as f:
        json.dump({"results": all_results, "N": n_stocks}, f, indent=2)
    log.info(f"\nSaved {len(all_results)} results to {out_path}")

    # Summary
    log.info("\n" + "=" * 60)
    log.info("SUMMARY")
    for K in [10, 20, 50]:
        poet = [r["tracking_error"] for r in all_results if r["K"] == K and r["model"] == "POET"]
        for mt in ["shrinkage", "structured"]:
            mse = [r["tracking_error"] for r in all_results if r["K"] == K and r["model"] == mt and r["loss"] == "mse"]
            task = [r["tracking_error"] for r in all_results if r["K"] == K and r["model"] == mt and r["loss"] == "task"]
            if mse and task:
                from scipy import stats
                _, p = stats.wilcoxon(task, mse, alternative='less')
                wins = sum(t < m for t, m in zip(task, mse))
                import numpy as np_
                log.info(f"  K={K} {mt}: MSE={np_.mean(mse):.4f}, DFL={np_.mean(task):.4f}, "
                         f"p={p:.4f}, wins={wins}/{len(mse)}")
        if poet:
            log.info(f"  K={K} POET: mean={np.mean(poet):.4f}")


if __name__ == "__main__":
    main()
