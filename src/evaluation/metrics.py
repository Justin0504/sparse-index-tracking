"""Portfolio evaluation metrics for index tracking."""

from __future__ import annotations

import numpy as np


def tracking_error(
    portfolio_returns: np.ndarray,
    index_returns: np.ndarray,
    annualization_factor: int = 252,
) -> float:
    """Annualized tracking error (std of return differences).

    TE = σ(rₚ - r_idx) × √252
    """
    diff = portfolio_returns - index_returns
    return float(np.std(diff, ddof=1) * np.sqrt(annualization_factor))


def max_tracking_deviation(
    portfolio_returns: np.ndarray,
    index_returns: np.ndarray,
) -> float:
    """Maximum absolute cumulative tracking deviation.

    MTD = max_t |Σ_{s≤t} (rₚ,ₛ - r_idx,ₛ)|
    """
    diff = portfolio_returns - index_returns
    cum_diff = np.cumsum(diff)
    return float(np.max(np.abs(cum_diff)))


def turnover(
    weights_before: np.ndarray,
    weights_after: np.ndarray,
) -> float:
    """One-way turnover: Σᵢ |w_after,ᵢ - w_before,ᵢ| / 2."""
    return float(np.sum(np.abs(weights_after - weights_before)) / 2)


def information_ratio(
    portfolio_returns: np.ndarray,
    index_returns: np.ndarray,
    annualization_factor: int = 252,
) -> float:
    """Annualized information ratio: mean excess return / tracking error.

    IR = (mean(rₚ - r_idx) × 252) / TE
    """
    diff = portfolio_returns - index_returns
    te = tracking_error(portfolio_returns, index_returns, annualization_factor)
    if te < 1e-10:
        return 0.0
    ann_mean = float(np.mean(diff)) * annualization_factor
    return ann_mean / te


def sparsity(weights: np.ndarray, threshold: float = 1e-4) -> int:
    """Number of assets with non-negligible weight."""
    return int(np.sum(np.abs(weights) > threshold))


def effective_sparsity(weights: np.ndarray) -> float:
    """Herfindahl-Hirschman index (inverse = effective number of stocks)."""
    w = np.abs(weights)
    w = w / (w.sum() + 1e-10)
    hhi = np.sum(w ** 2)
    if hhi < 1e-10:
        return 0.0
    return 1.0 / hhi
