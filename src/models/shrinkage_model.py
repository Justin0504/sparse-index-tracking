"""Learnable Ledoit-Wolf shrinkage models.

Three variants of increasing complexity:

1. ShrinkageCovarianceModel: Single global alpha (1 param).
   Grid search can match DFL — serves as the baseline.

2. ConditionalShrinkageModel: alpha_t = MLP(market_features_t).
   Alpha adapts to current market regime (volatility, correlation, etc.).
   Grid search CANNOT optimize over function space → DFL advantage.

3. StructuredShrinkageModel: Per-sector shrinkage targets (~12 params).
   Shrinkage target is block-structured instead of mu*I.
   Grid search over 12-dim space is intractable → DFL advantage.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class ShrinkageCovarianceModel(nn.Module):
    """Covariance model with a single learnable shrinkage intensity.

    This model does not use features — it takes a pre-computed sample
    covariance and applies learned shrinkage. The shrinkage parameter
    is the only trainable parameter, making this a minimal DFL baseline.
    """

    def __init__(self, n_stocks: int, init_alpha: float = 0.5):
        super().__init__()
        self.n_stocks = n_stocks
        # Learnable shrinkage in logit space for unconstrained optimization
        self._logit_alpha = nn.Parameter(
            torch.tensor(np.log(init_alpha / (1 - init_alpha)), dtype=torch.float32)
        )
        # Buffer for sample covariance (set before forward pass)
        self.register_buffer(
            "_sample_cov", torch.eye(n_stocks, dtype=torch.float32)
        )

    @property
    def alpha(self) -> torch.Tensor:
        return torch.sigmoid(self._logit_alpha)

    def set_sample_covariance(self, cov: torch.Tensor) -> None:
        """Set the sample covariance for the current rebalancing date."""
        self._sample_cov = cov

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Apply learned shrinkage to the stored sample covariance.

        Parameters
        ----------
        features : Tensor (N, d) — ignored, kept for API compatibility

        Returns
        -------
        cov : Tensor (N, N)
        """
        S = self._sample_cov
        mu = torch.trace(S) / self.n_stocks
        alpha = self.alpha
        cov = (1 - alpha) * S + alpha * mu * torch.eye(
            self.n_stocks, device=S.device, dtype=S.dtype
        )
        return cov


class ConditionalShrinkageModel(nn.Module):
    """Conditional shrinkage: alpha_t = MLP(market_features_t).

    The shrinkage intensity adapts to the current market regime
    (volatility, correlation structure, momentum). This creates a
    function-valued parameter that grid search cannot optimize.

    Parameters
    ----------
    n_stocks : int — number of stocks
    n_market_features : int — dimension of market-level features
    hidden_dim : int — MLP hidden layer size
    init_alpha : float — initial shrinkage intensity
    """

    def __init__(
        self,
        n_stocks: int,
        n_market_features: int = 5,
        hidden_dim: int = 16,
        init_alpha: float = 0.5,
    ):
        super().__init__()
        self.n_stocks = n_stocks
        self.n_market_features = n_market_features

        # MLP: market features → alpha (via sigmoid)
        self.alpha_net = nn.Sequential(
            nn.Linear(n_market_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        # Initialize bias of last layer so sigmoid output ≈ init_alpha
        with torch.no_grad():
            self.alpha_net[-1].bias.fill_(
                np.log(init_alpha / (1 - init_alpha))
            )
            # Small initial weights so alpha starts near init_alpha
            self.alpha_net[-1].weight.mul_(0.1)

        # Buffer for sample covariance
        self.register_buffer(
            "_sample_cov", torch.eye(n_stocks, dtype=torch.float32)
        )
        # Buffer for market features
        self.register_buffer(
            "_market_features", torch.zeros(n_market_features, dtype=torch.float32)
        )

    def set_sample_covariance(self, cov: torch.Tensor) -> None:
        self._sample_cov = cov

    def set_market_features(self, features: torch.Tensor) -> None:
        """Set market-level features for current rebalancing date."""
        self._market_features = features

    @property
    def alpha(self) -> torch.Tensor:
        """Current conditional alpha (for logging)."""
        return torch.sigmoid(self.alpha_net(self._market_features)).squeeze()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Apply conditional shrinkage.

        Parameters
        ----------
        features : Tensor (N, d) — per-stock features (ignored)

        Returns
        -------
        cov : Tensor (N, N)
        """
        S = self._sample_cov
        mu = torch.trace(S) / self.n_stocks

        # Conditional alpha from market features
        alpha = torch.sigmoid(self.alpha_net(self._market_features)).squeeze()

        cov = (1 - alpha) * S + alpha * mu * torch.eye(
            self.n_stocks, device=S.device, dtype=S.dtype
        )
        return cov


class StructuredShrinkageModel(nn.Module):
    """Structured shrinkage with learnable per-sector target variances.

    Sigma = (1 - alpha) * S_sample + alpha * T

    where T is a diagonal matrix with per-sector variance levels:
        T_ii = sigma^2_{sector(i)}

    This gives ~12 learnable parameters (1 alpha + 11 sector variances),
    creating a parameter space too large for grid search but natural for
    gradient-based optimization.

    Parameters
    ----------
    n_stocks : int — number of stocks
    n_sectors : int — number of sectors (default 11 for GICS)
    init_alpha : float — initial shrinkage intensity
    """

    def __init__(
        self,
        n_stocks: int,
        n_sectors: int = 11,
        init_alpha: float = 0.5,
    ):
        super().__init__()
        self.n_stocks = n_stocks
        self.n_sectors = n_sectors

        # Global shrinkage intensity
        self._logit_alpha = nn.Parameter(
            torch.tensor(np.log(init_alpha / (1 - init_alpha)), dtype=torch.float32)
        )

        # Per-sector log-variance (learnable, initialized to 0 → will be scaled by mu)
        self._log_sector_var = nn.Parameter(torch.zeros(n_sectors, dtype=torch.float32))

        # Sector assignments (deterministic: stock_i → sector i % n_sectors)
        sector_ids = torch.arange(n_stocks) % n_sectors
        self.register_buffer("_sector_ids", sector_ids)

        self.register_buffer(
            "_sample_cov", torch.eye(n_stocks, dtype=torch.float32)
        )

    @property
    def alpha(self) -> torch.Tensor:
        return torch.sigmoid(self._logit_alpha)

    def set_sample_covariance(self, cov: torch.Tensor) -> None:
        self._sample_cov = cov

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Apply structured shrinkage.

        Parameters
        ----------
        features : Tensor (N, d) — per-stock features (ignored)

        Returns
        -------
        cov : Tensor (N, N)
        """
        S = self._sample_cov
        mu = torch.trace(S) / self.n_stocks
        alpha = self.alpha

        # Build structured target: diagonal with per-sector variances
        # sector_var_i = mu * softplus(log_sector_var_i) so they stay positive
        # and are scaled relative to average variance
        sector_vars = mu * torch.nn.functional.softplus(self._log_sector_var)
        # Map each stock to its sector variance
        diag_values = sector_vars[self._sector_ids]  # (N,)
        T = torch.diag(diag_values)

        cov = (1 - alpha) * S + alpha * T
        return cov
