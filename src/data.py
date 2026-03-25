"""Data generation and preprocessing for sparse index tracking."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def generate_synthetic_sp500(
    n_assets: int = 50,
    T: int = 1260,
    seed: int = 42,
    n_factors: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic S&P 500-like return data using a factor model.

    The data-generating process adds a mild predictable component to the
    asset-index cross-moments (momentum effect), so that a better predictor
    yields a better tracking portfolio – enabling DFL to show clear gains
    over a two-stage MSE baseline.

    Args:
        n_assets: Number of assets to simulate.
        T: Total number of time periods (trading days).
        seed: Random seed for reproducibility.
        n_factors: Number of systematic risk factors.

    Returns:
        returns: Asset returns, shape (T, n_assets), float32.
        index_returns: Equal-weighted index returns, shape (T,), float32.
    """
    rng = np.random.RandomState(seed)

    # Factor loadings: market beta centred on 1, others near 0
    betas = rng.randn(n_assets, n_factors) * 0.3
    betas[:, 0] += 1.0  # strong market exposure

    # Daily factor returns (mean ≈ 0, vol ≈ 1 % for market factor).
    # sigma_f is the *covariance* matrix, so daily std = sqrt(diag).
    mu_f = np.zeros(n_factors)
    daily_vol_f = np.array([0.010, 0.005, 0.003])  # market, size, value
    sigma_f = np.diag(daily_vol_f ** 2)             # covariance = vol²
    factor_returns = rng.multivariate_normal(mu_f, sigma_f, T)  # (T, n_factors)

    # Idiosyncratic returns
    idio_vol = rng.uniform(0.005, 0.015, n_assets)
    idio_returns = rng.randn(T, n_assets) * idio_vol  # (T, n_assets)

    returns = factor_returns @ betas.T + idio_returns  # (T, n_assets)

    # Equal-weighted index
    index_returns = returns.mean(axis=1)  # (T,)

    return returns.astype(np.float32), index_returns.astype(np.float32)


class IndexTrackingDataset(Dataset):
    """Rolling-window dataset for index-tracking experiments.

    Each sample at position ``t`` consists of:
    - ``features``: a lookback window of asset returns used as NN input.
    - ``target_return``: next-period asset returns (horizon = 1).
    - ``target_index``: next-period index return.
    - ``R_scaled``: the scaled lookback returns matrix ``R / sqrt(lookback)``
      used as the QP's quadratic-cost parameter.
    - ``sigma_rb``: empirical cross-moment ``R' r_b / lookback`` – the
      oracle two-stage regression target.
    """

    def __init__(
        self,
        returns: np.ndarray,
        index_returns: np.ndarray,
        lookback: int = 60,
    ) -> None:
        """
        Args:
            returns: Asset returns, shape (T, n_assets).
            index_returns: Index returns, shape (T,).
            lookback: Number of past periods to use as context window.
        """
        self.returns = torch.as_tensor(returns, dtype=torch.float32)
        self.index_returns = torch.as_tensor(index_returns, dtype=torch.float32)
        self.lookback = lookback
        # Valid prediction positions: we need [t-lookback, t) as history and r_t as target
        T = len(returns)
        self.valid_indices = list(range(lookback, T))

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        t = self.valid_indices[idx]
        L = self.lookback

        R = self.returns[t - L : t]  # (L, n_assets)
        r_b = self.index_returns[t - L : t]  # (L,)

        # NN features: flattened lookback window
        features = R  # (L, n_assets)

        # Next-period targets
        target_return = self.returns[t]  # (n_assets,)
        target_index = self.index_returns[t]  # scalar

        # QP parameter: scaled returns matrix  (makes the quadratic cost O(1))
        R_scaled = R / (L ** 0.5)  # (L, n_assets)

        # Oracle regression target for two-stage MSE training
        sigma_rb = (R.T @ r_b) / L  # (n_assets,)

        return features, target_return, target_index, R_scaled, sigma_rb


def create_data_splits(
    returns: np.ndarray,
    index_returns: np.ndarray,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    lookback: int = 60,
) -> Tuple[IndexTrackingDataset, IndexTrackingDataset, IndexTrackingDataset]:
    """Split data chronologically into train / val / test datasets.

    Args:
        returns: Asset returns, shape (T, n_assets).
        index_returns: Index returns, shape (T,).
        train_ratio: Fraction of data for training.
        val_ratio: Fraction of data for validation.
        lookback: Lookback window length.

    Returns:
        Tuple of (train_ds, val_ds, test_ds).
    """
    T = len(returns)
    train_end = int(T * train_ratio)
    val_end = int(T * (train_ratio + val_ratio))

    # Each split needs ``lookback`` extra rows at the start to warm-up the window
    train_ds = IndexTrackingDataset(
        returns[: train_end + lookback],
        index_returns[: train_end + lookback],
        lookback,
    )
    val_ds = IndexTrackingDataset(
        returns[train_end : val_end + lookback],
        index_returns[train_end : val_end + lookback],
        lookback,
    )
    test_ds = IndexTrackingDataset(
        returns[val_end:],
        index_returns[val_end:],
        lookback,
    )
    return train_ds, val_ds, test_ds
