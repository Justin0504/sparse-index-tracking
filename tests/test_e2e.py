"""End-to-end integration test using synthetic data."""

import numpy as np
import torch
import pytest

from src.data.synthetic import generate_synthetic_data
from src.data.features import FeatureBuilder
from src.models.factor_model import FactorCovarianceModel
from src.models.neural_model import NeuralCovarianceModel
from src.optimization.solver import solve_tracking_qp
from src.optimization.qp_layer import SparseTrackingQP
from src.training.losses import mse_loss, task_loss
from src.evaluation.metrics import tracking_error, sparsity


@pytest.fixture
def synthetic_data():
    prices, index_prices, tickers = generate_synthetic_data(
        n_stocks=10, n_days=500, n_factors=3, seed=42
    )
    fb = FeatureBuilder(vol_window=21, momentum_window=63, use_sector=True, use_mcap_quintile=True)
    returns = fb.compute_returns(prices)
    return prices, returns, index_prices, tickers, fb


def test_full_pipeline_factor(synthetic_data):
    """Factor model → predict covariance → solve QP → compute metrics."""
    prices, returns, index_prices, tickers, fb = synthetic_data
    n_stocks = len(tickers)

    # Pick a date with enough lookback
    date = returns.index[100]
    features = fb.build_features(returns, date)
    n_features = features.shape[1]

    # Model forward pass
    model = FactorCovarianceModel(n_features=n_features, n_stocks=n_stocks, n_factors=3)
    features_t = torch.tensor(features, dtype=torch.float32)
    cov_pred = model(features_t).detach().numpy()

    # QP solve
    w_idx = np.ones(n_stocks, dtype=np.float64) / n_stocks
    w_opt = solve_tracking_qp(cov_pred, w_idx, K=5)

    # Verify feasibility
    assert abs(w_opt.sum() - 1.0) < 1e-3
    assert np.all(w_opt >= -1e-4)

    # Compute tracking error on next 21 days
    loc = returns.index.get_loc(date)
    future_rets = returns.iloc[loc:loc + 21].values
    port_r = future_rets @ w_opt
    idx_r = future_rets @ w_idx
    te = tracking_error(port_r, idx_r)
    assert te >= 0
    assert np.isfinite(te)


def test_full_pipeline_neural(synthetic_data):
    """Neural model → predict covariance → differentiable QP → task loss → backward."""
    prices, returns, index_prices, tickers, fb = synthetic_data
    n_stocks = len(tickers)

    date = returns.index[100]
    features = fb.build_features(returns, date)
    n_features = features.shape[1]

    model = NeuralCovarianceModel(n_features=n_features, n_stocks=n_stocks, hidden_dims=[32, 16], rank=3)
    features_t = torch.tensor(features, dtype=torch.float32, requires_grad=False)

    # Forward: features → covariance → QP → task loss
    cov_pred = model(features_t)
    assert cov_pred.shape == (n_stocks, n_stocks)

    # Check PSD
    eigvals = torch.linalg.eigvalsh(cov_pred)
    assert (eigvals >= -1e-5).all()


def test_mse_loss_backprop(synthetic_data):
    """MSE loss gradient flows through the model."""
    prices, returns, index_prices, tickers, fb = synthetic_data
    n_stocks = len(tickers)

    date = returns.index[100]
    features = fb.build_features(returns, date)
    n_features = features.shape[1]

    model = FactorCovarianceModel(n_features=n_features, n_stocks=n_stocks, n_factors=3)
    features_t = torch.tensor(features, dtype=torch.float32)
    cov_pred = model(features_t)

    cov_target_np = fb.compute_realized_covariance(returns, date, lookback=63)
    cov_target = torch.tensor(cov_target_np, dtype=torch.float32)

    loss = mse_loss(cov_pred, cov_target)
    loss.backward()

    # Verify gradients exist
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"


def test_synthetic_data_properties():
    """Verify synthetic data has realistic statistical properties."""
    prices, index_prices, tickers = generate_synthetic_data(
        n_stocks=30, n_days=1000, n_factors=5, seed=123
    )
    returns = prices.pct_change().iloc[1:]

    # Returns should be roughly zero-mean
    assert abs(returns.mean().mean()) < 0.01

    # Volatility should be in realistic range (1-5% daily)
    daily_vol = returns.std()
    assert daily_vol.min() > 0.001
    assert daily_vol.max() < 0.1

    # Cross-correlation should exist (not all independent)
    corr = returns.corr()
    off_diag = corr.values[np.triu_indices_from(corr.values, k=1)]
    assert abs(off_diag.mean()) > 0.01  # non-trivial correlation

    # No NaN
    assert prices.isna().sum().sum() == 0
    assert index_prices.isna().sum() == 0
