"""Tests for covariance prediction models."""

import numpy as np
import torch
import pytest

from src.models.factor_model import FactorCovarianceModel
from src.models.neural_model import NeuralCovarianceModel


@pytest.fixture
def features():
    """Synthetic features: 20 stocks, 18 features."""
    torch.manual_seed(42)
    return torch.randn(20, 18)


def test_factor_model_output_shape(features):
    model = FactorCovarianceModel(n_features=18, n_stocks=20, n_factors=5)
    cov = model(features)
    assert cov.shape == (20, 20)


def test_factor_model_psd(features):
    model = FactorCovarianceModel(n_features=18, n_stocks=20, n_factors=5)
    cov = model(features)
    eigvals = torch.linalg.eigvalsh(cov)
    assert (eigvals >= -1e-5).all(), f"Negative eigenvalue: {eigvals.min()}"


def test_factor_model_symmetric(features):
    model = FactorCovarianceModel(n_features=18, n_stocks=20, n_factors=5)
    cov = model(features)
    assert torch.allclose(cov, cov.T, atol=1e-6)


def test_neural_model_output_shape(features):
    model = NeuralCovarianceModel(n_features=18, n_stocks=20, hidden_dims=[64, 32])
    cov = model(features)
    assert cov.shape == (20, 20)


def test_neural_model_psd(features):
    model = NeuralCovarianceModel(n_features=18, n_stocks=20, hidden_dims=[64, 32])
    cov = model(features)
    eigvals = torch.linalg.eigvalsh(cov)
    assert (eigvals >= -1e-5).all(), f"Negative eigenvalue: {eigvals.min()}"


def test_neural_model_symmetric(features):
    model = NeuralCovarianceModel(n_features=18, n_stocks=20, hidden_dims=[64, 32])
    cov = model(features)
    assert torch.allclose(cov, cov.T, atol=1e-6)


def test_neural_model_low_rank_params(features):
    """Verify low-rank model has far fewer params than O(N^2)."""
    model = NeuralCovarianceModel(n_features=18, n_stocks=20, hidden_dims=[64, 32], rank=5)
    total_params = sum(p.numel() for p in model.parameters())
    # Old full-Cholesky model would have ~64M params for N=100
    # Low-rank model should be < 50K params for N=20
    assert total_params < 50_000, f"Too many parameters: {total_params}"


def test_factor_model_gradient_flows(features):
    model = FactorCovarianceModel(n_features=18, n_stocks=20, n_factors=5)
    features_grad = features.clone().requires_grad_(True)
    cov = model(features_grad)
    loss = cov.sum()
    loss.backward()
    assert features_grad.grad is not None
    assert not torch.all(features_grad.grad == 0)


def test_neural_model_gradient_flows(features):
    model = NeuralCovarianceModel(n_features=18, n_stocks=20, hidden_dims=[64, 32])
    features_grad = features.clone().requires_grad_(True)
    cov = model(features_grad)
    loss = cov.sum()
    loss.backward()
    assert features_grad.grad is not None
    assert not torch.all(features_grad.grad == 0)
