"""Loss functions for covariance prediction.

Two training paradigms:
  1. MSE loss (two-stage): minimize ‖Σ̂ - Σ_realized‖²_F
  2. Task loss (end-to-end / DFL): minimize realized tracking error of the
     portfolio produced by solving the QP with Σ̂.
"""

from __future__ import annotations

import torch


def mse_loss(cov_pred: torch.Tensor, cov_target: torch.Tensor) -> torch.Tensor:
    """Frobenius-norm MSE between predicted and realized covariance.

    Parameters
    ----------
    cov_pred : Tensor (N, N)
    cov_target : Tensor (N, N)

    Returns
    -------
    loss : scalar Tensor
    """
    diff = cov_pred - cov_target
    return (diff ** 2).mean()


def task_loss(
    w_portfolio: torch.Tensor,
    w_index: torch.Tensor,
    future_returns: torch.Tensor,
    annualization_factor: int = 252,
    w_prev: torch.Tensor | None = None,
    turnover_penalty: float = 0.0,
) -> torch.Tensor:
    """Annualized realized tracking error with optional turnover regularization.

    L = ann_factor * (1/T) Σ_t (rₚ,ₜ - r_idx,ₜ)²  +  λ * ½ ‖w - w_prev‖₁

    Parameters
    ----------
    w_portfolio : Tensor (N,) — optimized portfolio weights
    w_index : Tensor (N,) — index weights
    future_returns : Tensor (T, N) — realized returns over evaluation period
    annualization_factor : int — trading days per year
    w_prev : Tensor (N,) | None — previous period's portfolio weights
    turnover_penalty : float — λ for turnover regularization

    Returns
    -------
    loss : scalar Tensor
    """
    dtype = future_returns.dtype
    w_portfolio = w_portfolio.to(dtype)
    w_index = w_index.to(dtype)

    port_ret = future_returns @ w_portfolio
    idx_ret = future_returns @ w_index
    tracking_diff = port_ret - idx_ret
    loss = annualization_factor * (tracking_diff ** 2).mean()

    if w_prev is not None and turnover_penalty > 0:
        w_prev = w_prev.to(dtype)
        loss = loss + turnover_penalty * torch.sum(torch.abs(w_portfolio - w_prev)) / 2

    return loss
