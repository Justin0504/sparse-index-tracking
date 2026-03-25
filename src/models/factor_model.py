"""Linear factor covariance model: Σ̂ = B F Bᵀ + D.

The factor loadings B are estimated via PCA on trailing returns.
F is the factor covariance (diagonal), D is the residual (idiosyncratic) variance.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class FactorCovarianceModel(nn.Module):
    """Parametric factor model for covariance estimation.

    Given per-stock features X (N × d), predict the covariance matrix
    Σ̂ = B F Bᵀ + D  where:
      - B (N × K) are factor loadings (learned linear map from features)
      - F (K × K) is a diagonal factor covariance (learnable)
      - D (N,) is idiosyncratic variance (learnable per-stock)

    Making B, F, D learnable allows end-to-end training through the QP layer.
    """

    def __init__(self, n_features: int, n_stocks: int, n_factors: int = 10):
        super().__init__()
        self.n_stocks = n_stocks
        self.n_factors = n_factors

        # Feature → factor loadings: (d,) → (K,) per stock, applied identically
        self.loading_net = nn.Linear(n_features, n_factors, bias=False)

        # Log-scale factor variances (ensure positivity via exp)
        self.log_factor_var = nn.Parameter(torch.zeros(n_factors))

        # Log-scale idiosyncratic variances
        self.log_idio_var = nn.Parameter(torch.zeros(n_stocks))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Predict covariance matrix from features.

        Parameters
        ----------
        features : Tensor of shape (N, d)

        Returns
        -------
        cov : Tensor of shape (N, N), positive semi-definite
        """
        B = self.loading_net(features)  # (N, K)
        F_diag = torch.exp(self.log_factor_var)  # (K,)
        D_diag = torch.exp(self.log_idio_var)  # (N,)

        # Σ = B diag(F) Bᵀ + diag(D)
        # Efficient: B @ diag(F) @ Bᵀ = (B * sqrt(F)) @ (B * sqrt(F))ᵀ
        BF = B * F_diag.sqrt().unsqueeze(0)  # (N, K)
        cov = BF @ BF.T + torch.diag(D_diag)  # (N, N)

        return cov

    def init_from_pca(self, returns: np.ndarray, n_factors: int | None = None) -> None:
        """Initialize factor loadings from PCA on historical returns.

        Parameters
        ----------
        returns : ndarray of shape (T, N)
        n_factors : override number of factors (defaults to self.n_factors)
        """
        K = n_factors or self.n_factors
        # Center returns
        mu = returns.mean(axis=0)
        centered = returns - mu
        # SVD
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        # Factor loadings: top-K right singular vectors scaled by singular values
        B_init = Vt[:K].T * (S[:K] / np.sqrt(returns.shape[0]))  # (N, K)
        # Residual variance
        residual = centered - centered @ Vt[:K].T @ Vt[:K]
        D_init = np.var(residual, axis=0)  # (N,)

        with torch.no_grad():
            # We can't directly set loading_net weights since it maps features→loadings.
            # Instead, set factor and idiosyncratic variances from PCA.
            factor_var = (S[:K] ** 2) / returns.shape[0]
            self.log_factor_var.copy_(torch.tensor(np.log(factor_var + 1e-8), dtype=torch.float32))
            self.log_idio_var.copy_(torch.tensor(np.log(D_init + 1e-8), dtype=torch.float32))
