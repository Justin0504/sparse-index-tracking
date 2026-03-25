"""Tests for ReturnPredictor, TwoStageModel, and DFLModel."""

from __future__ import annotations

import pytest
import torch

from src.models import DFLModel, ReturnPredictor, TwoStageModel

N_ASSETS = 8
LOOKBACK = 15
BATCH = 4


@pytest.fixture
def batch_inputs():
    torch.manual_seed(0)
    features = torch.randn(BATCH, LOOKBACK, N_ASSETS)
    R_scaled = torch.randn(BATCH, LOOKBACK, N_ASSETS)
    target_return = torch.randn(BATCH, N_ASSETS)
    target_index = torch.randn(BATCH)
    sigma_rb = torch.randn(BATCH, N_ASSETS)
    return features, R_scaled, target_return, target_index, sigma_rb


# ---------------------------------------------------------------------------
# ReturnPredictor
# ---------------------------------------------------------------------------

class TestReturnPredictor:
    def test_output_shape(self, batch_inputs):
        features, *_ = batch_inputs
        model = ReturnPredictor(N_ASSETS, LOOKBACK, hidden_dims=[32, 16])
        c_hat = model(features)
        assert c_hat.shape == (BATCH, N_ASSETS)

    def test_gradient_flows(self, batch_inputs):
        features, _, target_return, *_ = batch_inputs
        model = ReturnPredictor(N_ASSETS, LOOKBACK, hidden_dims=[32, 16])
        c_hat = model(features)
        loss = c_hat.sum()
        loss.backward()
        # At least one parameter must have a gradient
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0


# ---------------------------------------------------------------------------
# TwoStageModel
# ---------------------------------------------------------------------------

class TestTwoStageModel:
    def test_predict_shape(self, batch_inputs):
        features, *_ = batch_inputs
        model = TwoStageModel(N_ASSETS, LOOKBACK, hidden_dims=[32, 16])
        c_hat = model.predict(features)
        assert c_hat.shape == (BATCH, N_ASSETS)

    def test_forward_weight_shapes(self, batch_inputs):
        features, R_scaled, *_ = batch_inputs
        model = TwoStageModel(N_ASSETS, LOOKBACK, hidden_dims=[32, 16])
        weights = model(features, R_scaled)
        assert weights.shape == (BATCH, N_ASSETS)

    def test_forward_valid_weights(self, batch_inputs):
        features, R_scaled, *_ = batch_inputs
        model = TwoStageModel(N_ASSETS, LOOKBACK, hidden_dims=[32, 16])
        weights = model(features, R_scaled)
        assert (weights >= -1e-5).all()
        sums = weights.sum(dim=-1)
        torch.testing.assert_close(sums, torch.ones(BATCH), atol=1e-4, rtol=1e-4)

    def test_no_gradient_through_qp(self, batch_inputs):
        """In TwoStageModel.forward(), the output has no grad_fn
        because the predictor runs inside torch.no_grad()."""
        features, R_scaled, *_ = batch_inputs
        model = TwoStageModel(N_ASSETS, LOOKBACK, hidden_dims=[32, 16])
        weights = model(features, R_scaled)
        # The weights tensor should not require grad (QP is detached from predictor)
        assert not weights.requires_grad, (
            "TwoStageModel weights must not carry a computation graph"
        )


# ---------------------------------------------------------------------------
# DFLModel
# ---------------------------------------------------------------------------

class TestDFLModel:
    def test_forward_weight_shapes(self, batch_inputs):
        features, R_scaled, *_ = batch_inputs
        model = DFLModel(N_ASSETS, LOOKBACK, hidden_dims=[32, 16])
        weights = model(features, R_scaled)
        assert weights.shape == (BATCH, N_ASSETS)

    def test_forward_valid_weights(self, batch_inputs):
        features, R_scaled, *_ = batch_inputs
        model = DFLModel(N_ASSETS, LOOKBACK, hidden_dims=[32, 16])
        weights = model(features, R_scaled)
        assert (weights >= -1e-5).all()
        sums = weights.sum(dim=-1)
        torch.testing.assert_close(sums, torch.ones(BATCH), atol=1e-4, rtol=1e-4)

    def test_gradient_flows_to_predictor(self, batch_inputs):
        """In DFLModel, gradients must flow from loss on weights back to predictor."""
        features, R_scaled, *_ = batch_inputs
        model = DFLModel(N_ASSETS, LOOKBACK, hidden_dims=[32, 16])
        model.zero_grad()
        weights = model(features, R_scaled)
        loss = weights.sum()
        loss.backward()
        predictor_grads = [p.grad for p in model.predictor.parameters() if p.grad is not None]
        assert len(predictor_grads) > 0, (
            "DFLModel must propagate gradients through QP to predictor"
        )
        assert all(not torch.isnan(g).any() for g in predictor_grads)
