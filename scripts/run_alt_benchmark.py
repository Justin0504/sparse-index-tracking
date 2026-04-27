#!/usr/bin/env python3
"""Run DFL experiments on alternative benchmarks.

Tests: Russell 2000 (IWM) small-cap and S&P 500 Sector (XLK tech).
Uses the same pipeline as main experiments but with different stock universes.
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
from src.models import ShrinkageCovarianceModel
from src.models.shrinkage_model import ConditionalShrinkageModel
from src.optimization.solver import solve_tracking_qp
from src.training import Trainer
from src.evaluation.metrics import tracking_error, turnover
from src.utils import set_seed, get_logger

log = get_logger(__name__)


def download_russell2000_universe(start="2012-01-01", end="2025-12-31",
                                   n_stocks=100, cache_path="data/cache/russell2000.pkl"):
    """Download top-N Russell 2000 constituents by volume."""
    cache = Path(cache_path)
    if cache.exists():
        log.info(f"Loading cached Russell 2000 data from {cache}")
        return pickle.load(open(cache, "rb"))

    import yfinance as yf

    # Use IWM (Russell 2000 ETF) as benchmark
    # For constituents, use a curated list of liquid small-caps
    # We'll use the iShares Russell 2000 top holdings + screen
    try:
        import requests
        from io import StringIO
        # Use a list of well-known small-cap tickers
        # These are representative Russell 2000 components
        headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
        # Fetch S&P 600 (small-cap) as proxy for Russell 2000 universe
        resp = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_600_companies",
                           headers=headers)
        resp.raise_for_status()
        table = pd.read_html(StringIO(resp.text))[0]
        all_tickers = sorted(table["Symbol"].str.replace(".", "-", regex=False).tolist())
        log.info(f"Got {len(all_tickers)} S&P 600 tickers")
    except Exception as e:
        log.error(f"Failed to get tickers: {e}")
        raise

    log.info(f"Downloading {len(all_tickers)} tickers ({start} → {end})...")
    raw = yf.download(all_tickers, start=start, end=end, auto_adjust=True, threads=True)
    prices = raw["Close"].dropna(axis=1, how="all")

    # Keep stocks with good coverage
    coverage = prices.notna().mean()
    good_tickers = coverage[coverage >= 0.90].index.tolist()
    prices = prices[good_tickers].ffill().bfill().dropna()

    # Select top N by average dollar volume
    avg_vol = prices.mean()
    top_tickers = avg_vol.nlargest(min(n_stocks, len(good_tickers))).index.tolist()
    prices = prices[top_tickers]

    # Download IWM as benchmark
    idx = yf.download("IWM", start=start, end=end, auto_adjust=True)
    index_prices = idx["Close"].squeeze()

    common_dates = prices.index.intersection(index_prices.index)
    prices = prices.loc[common_dates]
    index_prices = index_prices.loc[common_dates]

    tickers = list(prices.columns)
    log.info(f"Russell 2000 proxy: {len(tickers)} stocks, {len(prices)} trading days")

    result = (prices, index_prices, tickers, "Russell 2000 (IWM)")
    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "wb") as f:
        pickle.dump(result, f)
    return result


def download_tech_sector(start="2012-01-01", end="2025-12-31",
                          n_stocks=50, cache_path="data/cache/xlk_tech.pkl"):
    """Download tech sector stocks with XLK as benchmark."""
    cache = Path(cache_path)
    if cache.exists():
        log.info(f"Loading cached XLK data from {cache}")
        return pickle.load(open(cache, "rb"))

    import yfinance as yf

    # Major tech/IT tickers from S&P 500 IT sector
    tech_tickers = [
        "AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CRM", "AMD", "CSCO", "ACN", "ADBE",
        "IBM", "INTU", "TXN", "QCOM", "NOW", "AMAT", "PANW", "ADI", "LRCX", "MU",
        "KLAC", "SNPS", "CDNS", "MCHP", "FTNT", "MSI", "NXPI", "TEL", "HPQ", "DELL",
        "GLW", "ON", "ANSS", "KEYS", "CDW", "ZBRA", "TDY", "TRMB", "SWKS", "TER",
        "EPAM", "FFIV", "JNPR", "NTAP", "WDC", "STX", "HPE", "AKAM", "CTSH", "IT",
        "FSLR", "ENPH", "MPWR", "SMCI", "ANET", "CRWD", "ZS", "DDOG", "TEAM", "SNOW",
    ]

    log.info(f"Downloading {len(tech_tickers)} tech tickers ({start} → {end})...")
    raw = yf.download(tech_tickers, start=start, end=end, auto_adjust=True, threads=True)
    prices = raw["Close"].dropna(axis=1, how="all")

    coverage = prices.notna().mean()
    good_tickers = coverage[coverage >= 0.90].index.tolist()
    prices = prices[good_tickers].ffill().bfill().dropna()

    # Select top N by market cap proxy (latest price)
    top = prices.iloc[-1].nlargest(min(n_stocks, len(good_tickers))).index.tolist()
    prices = prices[top]

    # Download XLK as benchmark
    idx = yf.download("XLK", start=start, end=end, auto_adjust=True)
    index_prices = idx["Close"].squeeze()

    common_dates = prices.index.intersection(index_prices.index)
    prices = prices.loc[common_dates]
    index_prices = index_prices.loc[common_dates]

    tickers = list(prices.columns)
    log.info(f"Tech sector: {len(tickers)} stocks, {len(prices)} trading days")

    result = (prices, index_prices, tickers, "Tech Sector (XLK)")
    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "wb") as f:
        pickle.dump(result, f)
    return result


def run_benchmark_experiment(prices, index_prices, tickers, benchmark_name,
                              K, loss_type, model_type="shrinkage", device="cpu"):
    """Run experiment on alternative benchmark."""
    n_stocks = len(tickers)
    set_seed(42)

    fb = FeatureBuilder(vol_window=21, momentum_window=63,
                        use_sector=True, use_mcap_quintile=True)
    returns = fb.compute_returns(prices)

    splitter = RollingWindowSplitter(train_years=2, val_months=6,
                                     test_years=1, step_years=1)
    folds = splitter.split(returns.index)
    log.info(f"Benchmark={benchmark_name}, N={n_stocks}, K={K}, "
             f"model={model_type}, loss={loss_type}: {len(folds)} folds")

    min_offset = 64
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

        s = pd.Series(range(len(ret_test.index)), index=ret_test.index)
        monthly_idx = s.resample("ME").last().dropna()
        rebal_test = []
        for d in monthly_idx.index:
            closest = ret_test.index[ret_test.index <= d]
            if len(closest) > 0 and returns.index.get_loc(closest[-1]) >= min_offset:
                rebal_test.append(closest[-1])

        if not rebal_train or not rebal_test:
            continue

        if model_type == "shrinkage":
            model = ShrinkageCovarianceModel(n_stocks=n_stocks, init_alpha=0.5)
        elif model_type == "conditional":
            model = ConditionalShrinkageModel(n_stocks=n_stocks, n_market_features=5,
                                               hidden_dim=16, init_alpha=0.5)
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
            "benchmark": benchmark_name, "model": model_type,
            "loss": loss_type, "tracking_error": te, "avg_turnover": avg_to,
            "test_start": str(fold_info.test_start.date()),
            "test_end": str(fold_info.test_end.date()),
        }
        all_results.append(result)
        log.info(f"  Fold {fold_info.fold} | {benchmark_name} | K={K} | "
                 f"{model_type}/{loss_type} | TE={te:.4f}")

    return all_results


def main():
    all_results = []

    # Benchmark 1: Small-cap (Russell 2000 proxy via S&P 600)
    try:
        prices_r, idx_r, tickers_r, name_r = download_russell2000_universe()
        for K in [10, 20]:
            for model_type in ["shrinkage"]:
                for loss_type in ["mse", "task"]:
                    results = run_benchmark_experiment(
                        prices_r, idx_r, tickers_r, name_r,
                        K, loss_type, model_type
                    )
                    all_results.extend(results)
    except Exception as e:
        log.error(f"Russell 2000 failed: {e}")

    # Benchmark 2: Tech sector (XLK)
    try:
        prices_t, idx_t, tickers_t, name_t = download_tech_sector()
        for K in [5, 10, 20]:
            for model_type in ["shrinkage"]:
                for loss_type in ["mse", "task"]:
                    results = run_benchmark_experiment(
                        prices_t, idx_t, tickers_t, name_t,
                        K, loss_type, model_type
                    )
                    all_results.extend(results)
    except Exception as e:
        log.error(f"Tech sector failed: {e}")

    # Save
    out_path = Path("results/alt_benchmark_results.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"results": all_results}, f, indent=2)
    log.info(f"\nResults saved to {out_path}")

    # Summary
    print("\n=== Alternative Benchmark Results ===")
    benchmarks = set(r["benchmark"] for r in all_results)
    for bench in sorted(benchmarks):
        print(f"\n{bench}:")
        bench_results = [r for r in all_results if r["benchmark"] == bench]
        N = bench_results[0]["N"] if bench_results else "?"
        K_vals = sorted(set(r["K"] for r in bench_results))
        print(f"  N={N}")
        print(f"  {'K':>4} {'K/N':>6} {'MSE TE':>8} {'DFL TE':>8} {'Gain':>8}")
        for K in K_vals:
            mse = [r["tracking_error"] for r in bench_results if r["K"] == K and r["loss"] == "mse"]
            dfl = [r["tracking_error"] for r in bench_results if r["K"] == K and r["loss"] == "task"]
            if mse and dfl:
                mse_avg = np.mean(mse)
                dfl_avg = np.mean(dfl)
                gain = (1 - dfl_avg / mse_avg) * 100
                print(f"  {K:4d} {K/N:6.3f} {mse_avg:8.4f} {dfl_avg:8.4f} {gain:+7.1f}%")


if __name__ == "__main__":
    main()
