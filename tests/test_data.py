"""Tests for data generation and the IndexTrackingDataset."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from src.data import (
    IndexTrackingDataset,
    create_data_splits,
    generate_synthetic_sp500,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_ASSETS = 10
T = 200
LOOKBACK = 20


@pytest.fixture
def raw_data():
    returns, index_returns = generate_synthetic_sp500(
        n_assets=N_ASSETS, T=T, seed=0
    )
    return returns, index_returns


# ---------------------------------------------------------------------------
# generate_synthetic_sp500
# ---------------------------------------------------------------------------

class TestGenerateSyntheticSP500:
    def test_shape(self, raw_data):
        returns, index_returns = raw_data
        assert returns.shape == (T, N_ASSETS)
        assert index_returns.shape == (T,)

    def test_dtype(self, raw_data):
        returns, index_returns = raw_data
        assert returns.dtype == np.float32
        assert index_returns.dtype == np.float32

    def test_reproducibility(self):
        r1, i1 = generate_synthetic_sp500(n_assets=5, T=50, seed=7)
        r2, i2 = generate_synthetic_sp500(n_assets=5, T=50, seed=7)
        np.testing.assert_array_equal(r1, r2)
        np.testing.assert_array_equal(i1, i2)

    def test_different_seeds(self):
        r1, _ = generate_synthetic_sp500(n_assets=5, T=50, seed=1)
        r2, _ = generate_synthetic_sp500(n_assets=5, T=50, seed=2)
        assert not np.allclose(r1, r2)

    def test_reasonable_volatility(self, raw_data):
        returns, _ = raw_data
        daily_vol = returns.std(axis=0)
        # Daily vol should be in a reasonable range (0.1 % – 5 %)
        assert (daily_vol > 0.001).all()
        assert (daily_vol < 0.05).all()


# ---------------------------------------------------------------------------
# IndexTrackingDataset
# ---------------------------------------------------------------------------

class TestIndexTrackingDataset:
    def test_length(self, raw_data):
        returns, index_returns = raw_data
        ds = IndexTrackingDataset(returns, index_returns, lookback=LOOKBACK)
        # Valid indices: LOOKBACK … T-1 inclusive
        assert len(ds) == T - LOOKBACK

    def test_item_shapes(self, raw_data):
        returns, index_returns = raw_data
        ds = IndexTrackingDataset(returns, index_returns, lookback=LOOKBACK)
        features, target_return, target_index, R_scaled, sigma_rb = ds[0]

        assert features.shape == (LOOKBACK, N_ASSETS)
        assert target_return.shape == (N_ASSETS,)
        assert target_index.ndim == 0
        assert R_scaled.shape == (LOOKBACK, N_ASSETS)
        assert sigma_rb.shape == (N_ASSETS,)

    def test_R_scaled_normalisation(self, raw_data):
        returns, index_returns = raw_data
        ds = IndexTrackingDataset(returns, index_returns, lookback=LOOKBACK)
        features, _, _, R_scaled, _ = ds[0]

        # R_scaled should equal features / sqrt(LOOKBACK)
        expected = features / (LOOKBACK ** 0.5)
        torch.testing.assert_close(R_scaled, expected)

    def test_sigma_rb_formula(self, raw_data):
        returns, index_returns = raw_data
        ds = IndexTrackingDataset(returns, index_returns, lookback=LOOKBACK)
        features, _, _, _, sigma_rb = ds[0]

        r_b = ds.index_returns[LOOKBACK - LOOKBACK : LOOKBACK]
        R = features  # lookback window
        expected = (R.T @ r_b) / LOOKBACK
        torch.testing.assert_close(sigma_rb, expected)

    def test_dataloader_batch(self, raw_data):
        returns, index_returns = raw_data
        ds = IndexTrackingDataset(returns, index_returns, lookback=LOOKBACK)
        loader = DataLoader(ds, batch_size=8, shuffle=False)
        batch = next(iter(loader))
        assert len(batch) == 5
        features, target_return, target_index, R_scaled, sigma_rb = batch
        assert features.shape == (8, LOOKBACK, N_ASSETS)
        assert target_return.shape == (8, N_ASSETS)
        assert target_index.shape == (8,)
        assert R_scaled.shape == (8, LOOKBACK, N_ASSETS)
        assert sigma_rb.shape == (8, N_ASSETS)


# ---------------------------------------------------------------------------
# create_data_splits
# ---------------------------------------------------------------------------

class TestCreateDataSplits:
    def test_splits_non_empty(self, raw_data):
        returns, index_returns = raw_data
        train_ds, val_ds, test_ds = create_data_splits(
            returns, index_returns, lookback=LOOKBACK
        )
        assert len(train_ds) > 0
        assert len(val_ds) > 0
        assert len(test_ds) > 0

    def test_total_samples(self, raw_data):
        returns, index_returns = raw_data
        train_ds, val_ds, test_ds = create_data_splits(
            returns, index_returns, lookback=LOOKBACK
        )
        # Total samples = T - LOOKBACK (the lookback pads each split)
        # (approximate since splits share boundary rows)
        total = len(train_ds) + len(val_ds) + len(test_ds)
        assert total > 0
        assert total <= T
