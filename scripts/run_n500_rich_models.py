#!/usr/bin/env python3
"""Run conditional/structured shrinkage DFL experiments at N=478.

Extends run_n500_experiment.py to richer model classes.
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.features import FeatureBuilder
from src.data.splitter import RollingWindowSplitter
from src.models.shrinkage_model import ConditionalShrinkageModel, StructuredShrinkageModel
from src.optimization.solver import solve_tracking_qp
from src.training import Trainer
from src.evaluation.metrics import tracking_error, turnover
from src.utils import set_seed, get_logger

log = get_logger(__name__)


def load_large_universe(cache_path="data/cache/sp500_n500.pkl"):
    """Load cached N=478 data."""
    cache = Path(cache_path)
    if not cache.exists():
        raise RuntimeError(f"Cache not found at {cache}. Run run_n500_experiment.py first.")
    return pickle.load(open(cache, "rb"))


def run_rich_model_n500(prices, index_prices, tickers, K, loss_type,
                         model_type="conditional", device="cpu"):
    """Run a single model/K/loss experiment at N=478."""
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

        # Index weights
        train_prices = prices.loc[prices.index.isin(ret_train.index)]
        last_p = train_prices.iloc[-1].values
        w_index = (last_p / last_p.sum()).astype(np.float32)

        # Rebalancing dates (weekly Fridays)
        def _get_rebal(ret_df):
            dates = [d for d in ret_df.index if d.weekday() == 4]
            return [d for d in dates if returns.index.get_loc(d) >= min_offset]

        rebal_train = _get_rebal(ret_train)
        rebal_val = _get_rebal(ret_val)

        # Monthly rebalancing for test
        s = pd.Series(range(len(ret_test.index)), index=ret_test.index)
        monthly_idx = s.resample("ME").last().dropna()
        rebal_test = []
        for d in monthly_idx.index:
            closest = ret_test.index[ret_test.index <= d]
            if len(closest) > 0 and returns.index.get_loc(closest[-1]) >= min_offset:
                rebal_test.append(closest[-1])

        if not rebal_train or not rebal_test:
            continue

        # Create model
        if model_type == "conditional":
            model = ConditionalShrinkageModel(
                n_stocks=n_stocks, n_market_features=5,
                hidden_dim=16, init_alpha=0.5
            )
        elif model_type == "structured":
            model = StructuredShrinkageModel(
                n_stocks=n_stocks, n_sectors=11, init_alpha=0.5
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

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
        )

        # Test evaluation
        model.eval()
        portfolio_rets, index_rets = [], []
        weights_history = []
        w_idx_t = torch.tensor(w_index, dtype=torch.float32, device=device)

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
                sample_cov_t = torch.tensor(sample_cov_np, dtype=torch.float32, device=device)
                model.set_sample_covariance(sample_cov_t)

                if model_type == "conditional":
                    mkt_feat_np = fb.build_market_features(returns, date, 63)
                    mkt_feat_t = torch.tensor(mkt_feat_np, dtype=torch.float32, device=device)
                    model.set_market_features(mkt_feat_t)

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
        log.info(f"  Fold {fold_info.fold} | K={K} | {model_type}/{loss_type} | TE={te:.4f} | TO={avg_to:.4f}")

    return all_results


def main():
    prices, index_prices, tickers = load_large_universe()
    n_stocks = len(tickers)
    device = "cpu"

    K_values = [20, 50]
    model_types = ["conditional", "structured"]
    all_results = []

    for model_type in model_types:
        for K in K_values:
            for loss_type in ["mse", "task"]:
                log.info(f"\n{'='*60}")
                log.info(f"N={n_stocks}, K={K}, model={model_type}, loss={loss_type}")
                log.info(f"{'='*60}")
                results = run_rich_model_n500(
                    prices, index_prices, tickers, K, loss_type,
                    model_type=model_type, device=device
                )
                all_results.extend(results)

    # Save
    out_path = Path("results/n500_rich_models.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"results": all_results, "N": n_stocks}, f, indent=2)
    log.info(f"\nResults saved to {out_path}")

    # Summary
    print("\n=== N=478 Rich Model Results ===")
    print(f"{'Model':>15} {'K':>4} {'MSE TE':>8} {'DFL TE':>8} {'Gain':>8}")
    print("-" * 50)
    for model_type in model_types:
        for K in K_values:
            mse_tes = [r["tracking_error"] for r in all_results
                       if r["model"] == model_type and r["K"] == K and r["loss"] == "mse"]
            dfl_tes = [r["tracking_error"] for r in all_results
                       if r["model"] == model_type and r["K"] == K and r["loss"] == "task"]
            if mse_tes and dfl_tes:
                mse_avg = np.mean(mse_tes)
                dfl_avg = np.mean(dfl_tes)
                gain = (1 - dfl_avg / mse_avg) * 100
                print(f"{model_type:>15} {K:4d} {mse_avg:8.4f} {dfl_avg:8.4f} {gain:+7.1f}%")


if __name__ == "__main__":
    main()
