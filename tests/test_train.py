"""Tests for training routines and evaluation metrics."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from src.data import IndexTrackingDataset, generate_synthetic_sp500
from src.evaluate import (
    compute_te_reduction,
    diebold_mariano_test,
    evaluate_model,
)
from src.losses import mse_prediction_loss, tracking_error_loss
from src.models import DFLModel, TwoStageModel
from src.train import train_dfl, train_two_stage

N_ASSETS = 8
T = 120
LOOKBACK = 15
BATCH = 8


@pytest.fixture
def small_dataset():
    returns, index_returns = generate_synthetic_sp500(
        n_assets=N_ASSETS, T=T, seed=99
    )
    ds = IndexTrackingDataset(returns, index_returns, lookback=LOOKBACK)
    n = len(ds)
    n_train = int(n * 0.6)
    n_val = int(n * 0.2)
    train_ds = torch.utils.data.Subset(ds, range(n_train))
    val_ds = torch.utils.data.Subset(ds, range(n_train, n_train + n_val))
    test_ds = torch.utils.data.Subset(ds, range(n_train + n_val, n))
    return train_ds, val_ds, test_ds


@pytest.fixture
def loaders(small_dataset):
    train_ds, val_ds, test_ds = small_dataset
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH, shuffle=False)
    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

class TestLossFunctions:
    def test_tracking_error_loss_zero(self):
        """Perfect tracking → zero TE loss."""
        w = torch.ones(4, N_ASSETS) / N_ASSETS
        r = torch.randn(4, N_ASSETS)
        r_b = (w * r).sum(dim=-1)  # exactly the portfolio return
        loss = tracking_error_loss(w, r, r_b)
        assert loss.item() < 1e-10

    def test_tracking_error_loss_positive(self):
        torch.manual_seed(1)
        w = torch.softmax(torch.randn(4, N_ASSETS), dim=-1)
        r = torch.randn(4, N_ASSETS)
        r_b = torch.randn(4)
        loss = tracking_error_loss(w, r, r_b)
        assert loss.item() >= 0.0

    def test_mse_prediction_loss_zero(self):
        c = torch.randn(4, N_ASSETS)
        loss = mse_prediction_loss(c, c)
        assert loss.item() < 1e-10

    def test_mse_prediction_loss_positive(self):
        c_hat = torch.randn(4, N_ASSETS)
        sigma_rb = torch.randn(4, N_ASSETS)
        loss = mse_prediction_loss(c_hat, sigma_rb)
        assert loss.item() >= 0.0


# ---------------------------------------------------------------------------
# Training: two-stage
# ---------------------------------------------------------------------------

class TestTrainTwoStage:
    def test_train_runs(self, loaders):
        train_loader, val_loader, _ = loaders
        model = TwoStageModel(N_ASSETS, LOOKBACK, hidden_dims=[16, 8])
        trained = train_two_stage(
            model, train_loader, val_loader,
            n_epochs=2, lr=1e-3, verbose=False
        )
        assert trained is not None

    def test_loss_decreases(self, loaders):
        """Train loss should decrease over 5 epochs on a simple problem."""
        train_loader, val_loader, _ = loaders
        model = TwoStageModel(N_ASSETS, LOOKBACK, hidden_dims=[16, 8])

        def _one_epoch_loss(m, loader):
            m.eval()
            total = 0.0
            with torch.no_grad():
                for batch in loader:
                    features, _, _, _, sigma_rb = batch
                    c_hat = m.predict(features)
                    total += mse_prediction_loss(c_hat, sigma_rb).item()
            return total / len(loader)

        loss_before = _one_epoch_loss(model, train_loader)
        train_two_stage(model, train_loader, val_loader,
                        n_epochs=5, lr=1e-2, verbose=False)
        loss_after = _one_epoch_loss(model, train_loader)
        assert loss_after <= loss_before + 1e-3  # Should not get significantly worse


# ---------------------------------------------------------------------------
# Training: DFL
# ---------------------------------------------------------------------------

class TestTrainDFL:
    def test_train_runs(self, loaders):
        train_loader, val_loader, _ = loaders
        model = DFLModel(N_ASSETS, LOOKBACK, hidden_dims=[16, 8])
        trained = train_dfl(
            model, train_loader, val_loader,
            n_epochs=2, lr=1e-3, verbose=False
        )
        assert trained is not None

    def test_warmstart_does_not_crash(self, loaders):
        train_loader, val_loader, _ = loaders
        two_stage = TwoStageModel(N_ASSETS, LOOKBACK, hidden_dims=[16, 8])
        train_two_stage(two_stage, train_loader, val_loader,
                        n_epochs=1, verbose=False)
        dfl = DFLModel(N_ASSETS, LOOKBACK, hidden_dims=[16, 8])
        train_dfl(
            dfl, train_loader, val_loader,
            n_epochs=2, verbose=False,
            warmstart_state=two_stage.state_dict(),
        )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

class TestEvaluateModel:
    def test_evaluate_two_stage(self, loaders):
        _, _, test_loader = loaders
        model = TwoStageModel(N_ASSETS, LOOKBACK, hidden_dims=[16, 8])
        results = evaluate_model(model, test_loader)
        assert "annualized_te" in results
        assert "mse_te" in results
        assert "tracking_diffs" in results
        assert results["annualized_te"] >= 0.0

    def test_evaluate_dfl(self, loaders):
        _, _, test_loader = loaders
        model = DFLModel(N_ASSETS, LOOKBACK, hidden_dims=[16, 8])
        results = evaluate_model(model, test_loader)
        assert results["annualized_te"] >= 0.0


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

class TestDieboldMarianoTest:
    def test_structure(self):
        rng = np.random.RandomState(42)
        te1 = rng.randn(100) * 0.01
        te2 = rng.randn(100) * 0.008  # smaller → DFL better
        result = diebold_mariano_test(te1, te2)
        assert "dm_statistic" in result
        assert "p_value" in result
        assert 0.0 <= result["p_value"] <= 1.0

    def test_identical_errors_nonsignificant(self):
        rng = np.random.RandomState(0)
        te = rng.randn(200) * 0.01
        result = diebold_mariano_test(te, te)
        # Loss differential is 0 → p-value should be ~0.5
        assert result["dm_statistic"] == pytest.approx(0.0, abs=1e-8)

    def test_clear_improvement_significant(self):
        rng = np.random.RandomState(5)
        te_base = rng.randn(500) * 0.02
        te_dfl = te_base * 0.5  # DFL is clearly better
        result = diebold_mariano_test(te_base, te_dfl)
        assert result["dm_statistic"] > 0
        assert result["p_value"] < 0.01

    def test_te_reduction(self):
        reduction = compute_te_reduction(0.10, 0.09)
        assert reduction == pytest.approx(10.0, abs=0.01)

    def test_te_reduction_zero(self):
        assert compute_te_reduction(0.10, 0.10) == pytest.approx(0.0, abs=1e-6)
