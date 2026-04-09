"""Learnable Ledoit-Wolf shrinkage model.

Single learnable parameter alpha:
    Sigma = (1 - alpha) * S_sample + alpha * mu * I

where S_sample is the sample covariance computed from a lookback window
and mu = trace(S_sample) / N. Training with task loss tunes alpha to
minimize realized tracking error (light-touch DFL).
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
