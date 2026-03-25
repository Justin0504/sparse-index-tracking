"""Synthetic data generator for testing without real market data.

Generates realistic multi-factor stock returns with known covariance structure,
including regime switches (bull/bear), sector clustering, and fat tails.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_synthetic_data(
    n_stocks: int = 50,
    n_days: int = 2520,
    n_factors: int = 5,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Generate synthetic stock prices and index with realistic properties.

    Multi-factor model: r_t = B f_t + ε_t
    where B (N × K) are factor loadings, f_t ~ N(μ_f, Σ_f), ε_t ~ N(0, D).

    Parameters
    ----------
    n_stocks : Number of stocks.
    n_days : Number of trading days.
    n_factors : Number of latent factors.
    seed : Random seed.

    Returns
    -------
    prices : DataFrame (n_days × n_stocks) — synthetic adjusted close prices.
    index_prices : Series (n_days,) — market-cap weighted index prices.
    tickers : list[str] — synthetic ticker names.
    """
    rng = np.random.default_rng(seed)

    tickers = [f"SYN{i:03d}" for i in range(n_stocks)]
    dates = pd.bdate_range("2006-01-02", periods=n_days)

    # Factor loadings: assign stocks to sectors (clusters)
    n_sectors = min(11, n_stocks)
    sector_ids = np.arange(n_stocks) % n_sectors
    B = rng.standard_normal((n_stocks, n_factors)) * 0.3
    # Strengthen within-sector correlation
    for s in range(n_sectors):
        mask = sector_ids == s
        B[mask, s % n_factors] += 0.5

    # Factor returns with mild autocorrelation and regime switching
    factor_vol = rng.uniform(0.005, 0.015, size=n_factors)
    factor_mu = rng.uniform(-0.0001, 0.0003, size=n_factors)

    # Regime: low-vol (60%) and high-vol (40%) periods
    regime = np.ones(n_days)
    regime_switch = rng.random(n_days) < 0.005  # ~0.5% chance of switch per day
    current_regime = 1.0
    for t in range(n_days):
        if regime_switch[t]:
            current_regime = 2.5 if current_regime == 1.0 else 1.0
        regime[t] = current_regime

    factor_returns = np.zeros((n_days, n_factors))
    for k in range(n_factors):
        innovations = rng.standard_normal(n_days)
        # Fat tails via t-distribution mixing
        if rng.random() < 0.3:
            innovations = rng.standard_t(df=5, size=n_days)
        factor_returns[:, k] = factor_mu[k] + factor_vol[k] * regime * innovations

    # Idiosyncratic returns
    idio_vol = rng.uniform(0.005, 0.025, size=n_stocks)
    idio_returns = rng.standard_normal((n_days, n_stocks)) * idio_vol[np.newaxis, :]

    # Total returns
    stock_returns = factor_returns @ B.T + idio_returns  # (n_days, N)

    # Convert returns to prices (start at randomized initial prices)
    initial_prices = rng.uniform(20, 500, size=n_stocks)
    prices_np = np.zeros((n_days + 1, n_stocks))
    prices_np[0] = initial_prices
    for t in range(n_days):
        prices_np[t + 1] = prices_np[t] * (1 + stock_returns[t])
    prices_np = np.maximum(prices_np, 0.01)  # floor at 1 cent

    # Index: market-cap weighted (use initial prices as proxy)
    mcap_weights = initial_prices / initial_prices.sum()
    index_returns = stock_returns @ mcap_weights
    index_prices_np = np.zeros(n_days + 1)
    index_prices_np[0] = 1000.0  # base index level
    for t in range(n_days):
        index_prices_np[t + 1] = index_prices_np[t] * (1 + index_returns[t])

    # Build DataFrames (drop first row to align with returns starting point)
    all_dates = pd.bdate_range("2006-01-01", periods=n_days + 1)
    prices = pd.DataFrame(prices_np, index=all_dates, columns=tickers)
    index_prices = pd.Series(index_prices_np, index=all_dates, name="INDEX")

    return prices, index_prices, tickers
