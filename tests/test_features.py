"""Tests for feature engineering."""

import numpy as np
import pandas as pd
import pytest

from src.data.features import FeatureBuilder


@pytest.fixture
def sample_returns():
    """Create synthetic daily returns (100 days, 20 stocks)."""
    np.random.seed(42)
    dates = pd.bdate_range("2020-01-01", periods=100)
    data = np.random.randn(100, 20) * 0.02
    return pd.DataFrame(data, index=dates, columns=[f"S{i}" for i in range(20)])


@pytest.fixture
def fb():
    return FeatureBuilder(vol_window=21, momentum_window=63, use_sector=True, use_mcap_quintile=True)


def test_compute_returns():
    prices = pd.DataFrame({"A": [100, 102, 101], "B": [50, 51, 52]})
    fb = FeatureBuilder()
    rets = fb.compute_returns(prices)
    assert rets.shape == (2, 2)
    assert abs(rets.iloc[0, 0] - 0.02) < 1e-10


def test_build_features_shape(fb, sample_returns):
    date = sample_returns.index[70]
    features = fb.build_features(sample_returns, date)
    # vol(1) + momentum(1) + sector(11) + mcap(5) = 18
    assert features.shape == (20, 18)
    assert features.dtype == np.float32


def test_build_features_early_date(fb, sample_returns):
    """Features at early dates (before lookback) should still work (zero-filled)."""
    date = sample_returns.index[5]
    features = fb.build_features(sample_returns, date)
    assert features.shape[0] == 20
    # Volatility should be zero since loc < vol_window
    assert np.allclose(features[:, 0], 0)


def test_realized_covariance_shape(fb, sample_returns):
    date = sample_returns.index[80]
    cov = fb.compute_realized_covariance(sample_returns, date, lookback=63)
    assert cov.shape == (20, 20)
    # Should be symmetric
    assert np.allclose(cov, cov.T)
    # Should be PSD (all eigenvalues >= 0)
    eigvals = np.linalg.eigvalsh(cov)
    assert np.all(eigvals >= -1e-6)


def test_stock_index_covariance(fb, sample_returns):
    idx = sample_returns.mean(axis=1)  # equal-weight index proxy
    date = sample_returns.index[80]
    cov_idx = fb.compute_stock_index_covariance(sample_returns, idx, date, lookback=63)
    assert cov_idx.shape == (20,)
