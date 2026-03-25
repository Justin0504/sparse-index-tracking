"""Tests for the differentiable QP layer."""

from __future__ import annotations

import pytest
import torch

from src.qp_layer import SparseIndexQPLayer

N_ASSETS = 8
LOOKBACK = 15
BATCH = 4


@pytest.fixture
def qp_layer():
    return SparseIndexQPLayer(n_assets=N_ASSETS, lookback=LOOKBACK)


@pytest.fixture
def sample_inputs():
    torch.manual_seed(0)
    c_hat = torch.randn(BATCH, N_ASSETS, requires_grad=True)
    R_scaled = torch.randn(BATCH, LOOKBACK, N_ASSETS)
    return c_hat, R_scaled


class TestSparseIndexQPLayer:
    def test_output_shape(self, qp_layer, sample_inputs):
        c_hat, R_scaled = sample_inputs
        weights = qp_layer(c_hat, R_scaled)
        assert weights.shape == (BATCH, N_ASSETS)

    def test_weights_nonnegative(self, qp_layer, sample_inputs):
        c_hat, R_scaled = sample_inputs
        weights = qp_layer(c_hat, R_scaled)
        assert (weights >= -1e-5).all(), "Weights must be non-negative (up to solver tol)"

    def test_weights_sum_to_one(self, qp_layer, sample_inputs):
        c_hat, R_scaled = sample_inputs
        weights = qp_layer(c_hat, R_scaled)
        sums = weights.sum(dim=-1)
        torch.testing.assert_close(sums, torch.ones(BATCH), atol=1e-4, rtol=1e-4)

    def test_gradient_flows_through_qp(self, qp_layer, sample_inputs):
        """Gradient must flow from a scalar loss back to c_hat."""
        c_hat, R_scaled = sample_inputs
        weights = qp_layer(c_hat, R_scaled)
        loss = weights.sum()
        loss.backward()
        assert c_hat.grad is not None
        assert not torch.isnan(c_hat.grad).any(), "Gradient must not contain NaN"

    def test_gradient_nonzero(self, qp_layer, sample_inputs):
        """Gradient w.r.t. c_hat should be non-trivial."""
        c_hat, R_scaled = sample_inputs
        weights = qp_layer(c_hat, R_scaled)
        loss = (weights ** 2).sum()
        loss.backward()
        assert c_hat.grad.abs().max() > 1e-8

    def test_deterministic(self, qp_layer, sample_inputs):
        """Same inputs should yield identical outputs."""
        c_hat, R_scaled = sample_inputs
        w1 = qp_layer(c_hat.detach(), R_scaled)
        w2 = qp_layer(c_hat.detach(), R_scaled)
        torch.testing.assert_close(w1, w2)

    def test_batch_size_one(self, qp_layer):
        torch.manual_seed(1)
        c_hat = torch.randn(1, N_ASSETS, requires_grad=True)
        R_scaled = torch.randn(1, LOOKBACK, N_ASSETS)
        weights = qp_layer(c_hat, R_scaled)
        assert weights.shape == (1, N_ASSETS)
        assert (weights.sum(dim=-1) - 1.0).abs().item() < 1e-4
