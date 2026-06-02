#!/usr/bin/env python3
"""20-year S&P 500 backtest (2005-2025): validates diagnostic across regimes.

Covers 2008 crisis, 2020 COVID, 2022 rate hikes.
N=100 point-in-time, 17 folds (3y train + 0.5y val + 1y test, step 1y).
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
from src.models.shrinkage_model import (
    ShrinkageCovarianceModel,
    StructuredShrinkageModel,
)
from src.optimization.solver import solve_tracking_qp
from src.training import Trainer
from src.evaluation.metrics import tracking_error, turnover
from src.utils import set_seed, get_logger

log = get_logger(__name__)

CACHE_PATH = "data/cache/sp500_20y.pkl"


def load_20y_universe(start_date="2005-01-01", end_date="2025-12-31",
                      n_stocks=100, min_coverage=0.90):
    cache = Path(CACHE_PATH)
    if cache.exists():
        log.info(f"Loading cached 20y data from {cache}")
        return pickle.load(open(cache, "rb"))

    import yfinance as yf
    import requests
    from io import StringIO

    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
    resp = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                        headers=headers)
    resp.raise_for_status()
    table = pd.read_html(StringIO(resp.text))[0]
    all_tickers = sorted(table["Symbol"].str.replace(".", "-", regex=False).tolist())

    log.info(f"Downloading {len(all_tickers)} tickers ({start_date} → {end_date})...")
    raw = yf.download(all_tickers, start=start_date, end=end_date,
                      auto_adjust=True, threads=True)
    prices = raw["Close"].dropna(axis=1, how="all")

    coverage = prices.notna().mean()
    good = coverage[coverage >= min_coverage].index.tolist()
    log.info(f"Stocks with >= {min_coverage:.0%} coverage: {len(good)}")
    prices = prices[good].ffill().bfill().dropna()

    if len(prices.columns) > n_stocks:
        mcap_proxy = prices.iloc[-1]
        top = mcap_proxy.nlargest(n_stocks).index.tolist()
        prices = prices[top]

    idx = yf.download("^GSPC", start=start_date, end=end_date, auto_adjust=True)
    index_prices = idx["Close"].squeeze()

    common = prices.index.intersection(index_prices.index)
    prices = prices.loc[common]
    index_prices = index_prices.loc[common]
    tickers = list(prices.columns)

    log.info(f"Final: {len(tickers)} stocks, {len(prices)} trading days ({prices.index[0].date()} → {prices.index[-1].date()})")

    result = (prices, index_prices, tickers)
    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "wb") as f:
        pickle.dump(result, f)
    return result


def run_experiment(prices, index_prices, tickers, K, loss_type, model_type="shrinkage",
                   device="cpu"):
    n_stocks = len(tickers)
    set_seed(42)

    fb = FeatureBuilder(vol_window=21, momentum_window=63,
                        use_sector=True, use_mcap_quintile=True)
    returns = fb.compute_returns(prices)

    splitter = RollingWindowSplitter(train_years=3, val_months=6,
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

        regime = "normal"
        y = fold_info.test_start.year
        if 2007 <= y <= 2009:
            regime = "crisis_2008"
        elif 2019 <= y <= 2020:
            regime = "covid_2020"
        elif 2022 <= y <= 2023:
            regime = "rate_hike_2022"

        result = {
            "fold": fold_info.fold, "K": K, "N": n_stocks,
            "model": model_type, "loss": loss_type,
            "tracking_error": te, "avg_turnover": avg_to,
            "test_start": str(fold_info.test_start.date()),
            "test_end": str(fold_info.test_end.date()),
            "regime": regime,
        }
        all_results.append(result)
        log.info(f"  Fold {fold_info.fold} | {regime} | K={K} | {model_type} | {loss_type} | TE={te:.4f}")

    return all_results


def run_poet_baseline(prices, index_prices, tickers, K, device="cpu"):
    n_stocks = len(tickers)
    set_seed(42)

    fb = FeatureBuilder(vol_window=21, momentum_window=63,
                        use_sector=True, use_mcap_quintile=True)
    returns = fb.compute_returns(prices)

    splitter = RollingWindowSplitter(train_years=3, val_months=6,
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

        regime = "normal"
        y = fold_info.test_start.year
        if 2007 <= y <= 2009:
            regime = "crisis_2008"
        elif 2019 <= y <= 2020:
            regime = "covid_2020"
        elif 2022 <= y <= 2023:
            regime = "rate_hike_2022"

        result = {
            "fold": fold_info.fold, "K": K, "N": n_stocks,
            "model": "POET", "loss": "N/A",
            "tracking_error": te, "avg_turnover": avg_to,
            "test_start": str(fold_info.test_start.date()),
            "test_end": str(fold_info.test_end.date()),
            "regime": regime,
        }
        all_results.append(result)
        log.info(f"  POET Fold {fold_info.fold} | {regime} | K={K} | TE={te:.4f}")

    return all_results


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    log.info(f"Device: {device}")

    log.info("Loading 20-year S&P 500 universe (N=100)...")
    prices, index_prices, tickers = load_20y_universe()
    n_stocks = len(tickers)
    log.info(f"20y universe: N={n_stocks}, {len(prices)} trading days")

    all_results = []

    for K in [20]:
        log.info(f"\n{'='*60}\nPOET baseline K={K}")
        results = run_poet_baseline(prices, index_prices, tickers, K, device)
        all_results.extend(results)

        for model_type in ["shrinkage", "structured"]:
            for loss in ["mse", "task"]:
                log.info(f"\n{'='*60}\nK={K}, model={model_type}, loss={loss}")
                results = run_experiment(prices, index_prices, tickers, K, loss,
                                         model_type=model_type, device=device)
                all_results.extend(results)

    out_dir = Path("results/20y_backtest")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "20y_results.json"
    with open(out_path, "w") as f:
        json.dump({"results": all_results, "N": n_stocks}, f, indent=2)
    log.info(f"\nSaved {len(all_results)} results to {out_path}")

    log.info("\n" + "=" * 60)
    log.info("SUMMARY BY REGIME")
    regimes = sorted(set(r["regime"] for r in all_results))
    for regime in regimes:
        regime_results = [r for r in all_results if r["regime"] == regime]
        for mt in ["shrinkage", "structured"]:
            mse = [r["tracking_error"] for r in regime_results if r["model"] == mt and r["loss"] == "mse"]
            task = [r["tracking_error"] for r in regime_results if r["model"] == mt and r["loss"] == "task"]
            if mse and task and len(mse) >= 2:
                from scipy import stats
                try:
                    _, p = stats.wilcoxon(task, mse, alternative='less')
                except ValueError:
                    p = 1.0
                wins = sum(t < m for t, m in zip(task, mse))
                log.info(f"  {regime} | {mt}: MSE={np.mean(mse):.4f}, DFL={np.mean(task):.4f}, "
                         f"p={p:.4f}, wins={wins}/{len(mse)}")
        poet = [r["tracking_error"] for r in regime_results if r["model"] == "POET"]
        if poet:
            log.info(f"  {regime} | POET: mean={np.mean(poet):.4f}")


if __name__ == "__main__":
    main()
