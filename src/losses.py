"""Loss functions for sparse index tracking."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def tracking_error_loss(
    weights: torch.Tensor,
    target_return: torch.Tensor,
    target_index: torch.Tensor,
) -> torch.Tensor:
    """Mean squared portfolio tracking error over a batch.

    Computes ``E[(w' r - r_b)²]`` where *r* and *r_b* are the next-period
    asset and index returns, respectively.

    Args:
        weights: Portfolio weights, shape ``(batch, n_assets)``.
        target_return: Next-period asset returns, shape ``(batch, n_assets)``.
        target_index: Next-period index returns, shape ``(batch,)``.

    Returns:
        Scalar MSE tracking error.
    """
    portfolio_ret = (weights * target_return).sum(dim=-1)  # (batch,)
    diff = portfolio_ret - target_index  # (batch,)
    return (diff ** 2).mean()


def mse_prediction_loss(
    c_hat: torch.Tensor,
    sigma_rb: torch.Tensor,
) -> torch.Tensor:
    """Mean squared error between predicted and true cross-moments.

    This is the stage-1 loss for the two-stage MSE baseline.

    Args:
        c_hat: Predicted cross-moments, shape ``(batch, n_assets)``.
        sigma_rb: True cross-moments, shape ``(batch, n_assets)``.

    Returns:
        Scalar MSE loss.
    """
    return F.mse_loss(c_hat, sigma_rb)
