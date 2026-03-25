"""Neural covariance estimator with low-rank Cholesky parameterization.

Architecture: MLP encoder → low-rank factors + diagonal → Σ̂ = FFᵀ + D
Guarantees PSD output by construction. O(NK) parameters instead of O(N²).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class NeuralCovarianceModel(nn.Module):
    """MLP that maps per-stock features to a PSD covariance matrix.

    Uses a low-rank-plus-diagonal parameterization instead of full Cholesky
    to avoid the O(N²) parameter explosion:

        Σ̂ = F Fᵀ + diag(d)

    where F (N × rank) captures cross-stock correlations and d (N,) captures
    idiosyncratic variance. Total parameters: O(N × rank) instead of O(N²).

    Parameters
    ----------
    n_features : int
        Dimension of per-stock feature vector.
    n_stocks : int
        Number of stocks (N).
    hidden_dims : list[int]
        Hidden layer sizes for the shared embedding MLP.
    dropout : float
        Dropout rate.
    rank : int
        Rank of the low-rank factor (default: min(n_stocks, 20)).
    """

    def __init__(
        self,
        n_features: int,
        n_stocks: int,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.1,
        rank: int | None = None,
    ):
        super().__init__()
        self.n_stocks = n_stocks
        self.rank = rank or min(n_stocks, 20)
        hidden_dims = hidden_dims or [256, 128]

        # Per-stock feature encoder (shared weights across stocks)
        layers: list[nn.Module] = []
        in_dim = n_features
        for h in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = h
        self.encoder = nn.Sequential(*layers)
        self.embed_dim = in_dim

        # Per-stock heads: embedding → factor row (rank,) + idio variance (1,)
        self.factor_head = nn.Linear(self.embed_dim, self.rank)
        self.log_diag_head = nn.Linear(self.embed_dim, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Predict covariance matrix.

        Parameters
        ----------
        features : Tensor of shape (N, d)

        Returns
        -------
        cov : Tensor of shape (N, N), positive semi-definite
        """
        # Encode each stock (shared MLP)
        embeddings = self.encoder(features)  # (N, embed_dim)

        # Low-rank factor: each stock contributes one row
        F = self.factor_head(embeddings)  # (N, rank)

        # Idiosyncratic variance (strictly positive via exp)
        log_d = self.log_diag_head(embeddings).squeeze(-1)  # (N,)
        d = torch.exp(log_d.clamp(max=10)) + 1e-6  # clamp to prevent overflow

        # Σ = F Fᵀ + diag(d)  — PSD by construction
        cov = F @ F.T + torch.diag(d)

        return cov
