#!/usr/bin/env python3
"""Covariance window sensitivity ablation.

Tests DFL vs MSE across lookback windows {21, 42, 63, 126} at K=20 (neural model).
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.features import FeatureBuilder
from src.data.splitter import RollingWindowSplitter
from src.models import NeuralCovarianceModel
from src.optimization.solver import solve_tracking_qp
from src.training import Trainer
from src.evaluation.metrics import tracking_error, turnover
from src.utils import set_seed, get_logger

log = get_logger(__name__)


def run_window_experiment(lookback, K=20, loss_type="task", device="cpu"):
    """Run neural model experiment with a specific lookback window."""
    set_seed(42)
    n_stocks = 100

    fb = FeatureBuilder(vol_window=21, momentum_window=63,
                        use_sector=True, use_mcap_quintile=True)

    from src.data.loader import SP500DataLoader
    loader = SP500DataLoader(start_date="2006-01-01", end_date="2025-12-31", top_n=100)
    prices, index_prices, tickers = loader.load()

    returns = fb.compute_returns(prices)
    splitter = RollingWindowSplitter(train_years=2, val_months=6,
                                     test_years=1, step_years=1)
    folds = splitter.split(returns.index)
    log.info(f"lookback={lookback}, K={K}, loss={loss_type}: {len(folds)} folds")

    min_offset = max(64, lookback + 1)
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
        last_p = train_prices.iloc[-1].values
        w_index = (last_p / last_p.sum()).astype(np.float32)

        def _get_rebal(ret_df):
            dates = [d for d in ret_df.index if d.weekday() == 4]
            return [d for d in dates if returns.index.get_loc(d) >= min_offset]

        rebal_train = _get_rebal(ret_train)
        rebal_val = _get_rebal(ret_val)

        import pandas as pd
        s = pd.Series(range(len(ret_test.index)), index=ret_test.index)
        monthly_idx = s.resample("ME").last().dropna()
        rebal_test = []
        for d in monthly_idx.index:
            closest = ret_test.index[ret_test.index <= d]
            if len(closest) > 0 and returns.index.get_loc(closest[-1]) >= min_offset:
                rebal_test.append(closest[-1])

        if not rebal_train or not rebal_test:
            continue

        model = NeuralCovarianceModel(n_stocks=n_stocks, n_features=n_features,
                                       hidden_dims=[64, 64])

        trainer = Trainer(
            model=model, feature_builder=fb, loss_type=loss_type,
            lr=1e-3, weight_decay=1e-4, epochs=50, patience=10,
            device=device, sparsity_K=K, selection="market_cap",
        )

        history = trainer.train(
            returns_train=ret_train.values,
            returns_val=ret_val.values,
            rebalance_dates_train=rebal_train,
            rebalance_dates_val=rebal_val,
            returns_df_train=returns,
            returns_df_val=returns,
            w_index=w_index,
            lookback=lookback,
        )

        model.eval()
        portfolio_rets, index_rets = [], []
        weights_history = []

        for i, date in enumerate(rebal_test):
            loc = returns.index.get_loc(date)
            next_loc = (returns.index.get_loc(rebal_test[i + 1])
                        if i + 1 < len(rebal_test) else len(returns))
            if next_loc <= loc:
                continue

            features_np = fb.build_features(returns, date)
            with torch.no_grad():
                features_t = torch.tensor(features_np, dtype=torch.float32, device=device)
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

        pr = np.array(portfolio_rets)
        ir = np.array(index_rets)
        te = tracking_error(pr, ir, 252)

        result = {
            "fold": fold_info.fold, "K": K, "lookback": lookback,
            "loss": loss_type, "tracking_error": te,
        }
        all_results.append(result)
        log.info(f"  Fold {fold_info.fold} | lb={lookback} | {loss_type} | TE={te:.4f}")

    return all_results


def main():
    all_results = []
    windows = [21, 42, 63, 126]

    for lb in windows:
        for loss in ["mse", "task"]:
            log.info(f"\n{'='*50}")
            log.info(f"Window={lb}, loss={loss}")
            log.info(f"{'='*50}")
            results = run_window_experiment(lb, K=20, loss_type=loss)
            all_results.extend(results)

    out_path = Path("results/cov_window_ablation.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"results": all_results}, f, indent=2)
    log.info(f"\nResults saved to {out_path}")

    print("\n=== Covariance Window Ablation (K=20, Neural) ===")
    print(f"{'Window':>8} {'MSE TE':>8} {'DFL TE':>8} {'Gain':>8}")
    for lb in windows:
        mse = [r["tracking_error"] for r in all_results if r["lookback"] == lb and r["loss"] == "mse"]
        dfl = [r["tracking_error"] for r in all_results if r["lookback"] == lb and r["loss"] == "task"]
        if mse and dfl:
            m, d = np.mean(mse), np.mean(dfl)
            gain = (1 - d / m) * 100
            print(f"{lb:8d} {m:8.4f} {d:8.4f} {gain:+7.1f}%")


if __name__ == "__main__":
    main()
