"""Neural network models for sparse index tracking.

Two model variants are provided:

``ReturnPredictor``
    A feed-forward MLP that maps a flattened lookback window of asset
    returns to a predicted asset–index cross-moment vector  c ∈ ℝⁿ.

``TwoStageModel``
    The two-stage MSE baseline.  The predictor is trained to minimise
    MSE(ĉ, σ_rb) where σ_rb is the empirical cross-moment.  At
    inference time the QP layer converts ĉ into portfolio weights.
    Gradients do **not** flow through the QP layer during stage-1 training.

``DFLModel``
    End-to-end Decision-Focused Learning model.  Gradients flow from the
    tracking-error loss through the differentiable QP layer back into the
    predictor, so the NN learns predictions that directly minimise the
    downstream portfolio tracking error.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .qp_layer import SparseIndexQPLayer


class ReturnPredictor(nn.Module):
    """MLP that predicts the asset-index cross-moment vector c ∈ ℝⁿ.

    Args:
        n_assets: Number of assets.
        lookback: Lookback window length (determines input dimension).
        hidden_dims: List of hidden-layer widths.
        dropout: Dropout probability applied after each hidden layer.
    """

    def __init__(
        self,
        n_assets: int,
        lookback: int,
        hidden_dims: List[int] | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128]

        input_dim = n_assets * lookback
        layers: list[nn.Module] = []
        in_dim = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, n_assets))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Lookback window of returns, shape ``(batch, lookback, n_assets)``.

        Returns:
            c_hat: Predicted cross-moments, shape ``(batch, n_assets)``.
        """
        batch = x.shape[0]
        return self.network(x.reshape(batch, -1))


class TwoStageModel(nn.Module):
    """Two-stage MSE baseline model.

    Stage 1 (training): minimise MSE(predictor(x), σ_rb).
    Stage 2 (inference): feed predicted ĉ through the QP layer.

    The QP layer is **not** part of the stage-1 computation graph.

    Args:
        n_assets: Number of assets.
        lookback: Lookback window length.
        hidden_dims: Hidden-layer widths for the predictor MLP.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        n_assets: int,
        lookback: int,
        hidden_dims: List[int] | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.predictor = ReturnPredictor(n_assets, lookback, hidden_dims, dropout)
        self.qp_layer = SparseIndexQPLayer(n_assets, lookback)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the predictor only (for stage-1 MSE training).

        Args:
            x: Lookback window, shape ``(batch, lookback, n_assets)``.

        Returns:
            c_hat: Predicted cross-moments, shape ``(batch, n_assets)``.
        """
        return self.predictor(x)

    def forward(
        self,
        x: torch.Tensor,
        R_scaled: torch.Tensor,
    ) -> torch.Tensor:
        """Predict weights without gradient flowing through the QP.

        Args:
            x: Lookback window, shape ``(batch, lookback, n_assets)``.
            R_scaled: Scaled returns, shape ``(batch, lookback, n_assets)``.

        Returns:
            weights: Portfolio weights, shape ``(batch, n_assets)``.
        """
        with torch.no_grad():
            c_hat = self.predictor(x)
        return self.qp_layer(c_hat, R_scaled)


class DFLModel(nn.Module):
    """End-to-end Decision-Focused Learning model.

    The differentiable QP layer is part of the computation graph, so
    gradients flow from the tracking-error loss back into the predictor.

    Args:
        n_assets: Number of assets.
        lookback: Lookback window length.
        hidden_dims: Hidden-layer widths for the predictor MLP.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        n_assets: int,
        lookback: int,
        hidden_dims: List[int] | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.predictor = ReturnPredictor(n_assets, lookback, hidden_dims, dropout)
        self.qp_layer = SparseIndexQPLayer(n_assets, lookback)

    def forward(
        self,
        x: torch.Tensor,
        R_scaled: torch.Tensor,
    ) -> torch.Tensor:
        """End-to-end forward pass with gradient through the QP layer.

        Args:
            x: Lookback window, shape ``(batch, lookback, n_assets)``.
            R_scaled: Scaled returns, shape ``(batch, lookback, n_assets)``.

        Returns:
            weights: Portfolio weights, shape ``(batch, n_assets)``.
        """
        c_hat = self.predictor(x)
        return self.qp_layer(c_hat, R_scaled)
