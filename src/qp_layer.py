"""Differentiable QP layer for sparse index tracking.

The tracking-error minimisation problem (over the lookback window) is:

    min_w  (1 / (2 L)) * ||R w||²  -  c' w
    s.t.   w >= 0,  sum(w) == 1

where
- R ∈ ℝ^{L×n}  is the matrix of asset returns over the lookback window,
- c ∈ ℝ^n       is the predicted asset–index cross-moment (NN output),
- L             is the lookback length,
- n             is the number of assets.

Passing  R_scaled = R / sqrt(L)  yields the equivalent DPP problem:

    min_w  (1/2) * ||R_scaled w||²  -  c' w

which is DPP-compatible with cvxpylayers and admits gradient propagation
back through the QP solve (via the KKT implicit-function theorem).
"""

from __future__ import annotations

import cvxpy as cp
import torch
import torch.nn as nn
from cvxpylayers.torch import CvxpyLayer


class SparseIndexQPLayer(nn.Module):
    """Differentiable QP layer for sparse index tracking.

    Solves a batch of portfolio-optimisation problems of the form::

        min_w  0.5 * ||R_scaled @ w||²  -  c' w
        s.t.   w >= 0,  sum(w) == 1

    and returns the optimal weights *w** as a differentiable function of
    the parameter *c* (the NN output).

    Args:
        n_assets: Number of assets (optimisation variable dimension).
        lookback: Length of the lookback window (rows in R_scaled).
    """

    def __init__(self, n_assets: int, lookback: int) -> None:
        super().__init__()
        self.n_assets = n_assets
        self.lookback = lookback

        # ------------------------------------------------------------------
        # Build the CVXPY problem once; reuse it for every forward pass.
        # ------------------------------------------------------------------
        w = cp.Variable(n_assets)
        c_param = cp.Parameter(n_assets)
        R_param = cp.Parameter((lookback, n_assets))

        # DPP-compatible objective: sum_squares(affine(w)) - linear(w)
        objective = cp.Minimize(
            0.5 * cp.sum_squares(R_param @ w) - c_param @ w
        )
        constraints = [w >= 0, cp.sum(w) == 1]
        problem = cp.Problem(objective, constraints)

        assert problem.is_dpp(), "QP problem must satisfy DPP for gradient support."

        self._layer = CvxpyLayer(
            problem,
            parameters=[c_param, R_param],
            variables=[w],
        )

    def forward(
        self,
        c_hat: torch.Tensor,
        R_scaled: torch.Tensor,
    ) -> torch.Tensor:
        """Solve a batch of QPs and return portfolio weights.

        Inputs are automatically promoted to float64 (required by the
        underlying SCS solver) and the output is cast back to the dtype of
        *c_hat*.

        Args:
            c_hat: Predicted cross-moments, shape ``(batch, n_assets)``.
            R_scaled: Scaled lookback returns ``R / sqrt(L)``,
                shape ``(batch, lookback, n_assets)``.

        Returns:
            weights: Non-negative portfolio weights that sum to 1,
                shape ``(batch, n_assets)``, same dtype as *c_hat*.
        """
        in_dtype = c_hat.dtype
        weights, = self._layer(
            c_hat.double(),
            R_scaled.double(),
            solver_args={"eps": 1e-6, "max_iters": 10_000},
        )
        # Clamp tiny negative values introduced by solver tolerance
        weights = weights.clamp(min=0.0)
        # Re-normalise to ensure the simplex constraint holds exactly
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        return weights.to(in_dtype)
