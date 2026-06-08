"""Loss functions for covariance prediction.

Training paradigms:
  1. MSE loss (two-stage): minimize ‖Σ̂ - Σ_realized‖²_F
  2. Task loss (end-to-end / DFL): minimize realized tracking error of the
     portfolio produced by solving the QP with Σ̂.
  3. LODL (Shah et al. 2022): learn a local surrogate loss that approximates
     the task loss without differentiable optimization.
"""

from __future__ import annotations

import torch
import numpy as np


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


def bd_dfl_loss(
    w_portfolio: torch.Tensor,
    w_index: torch.Tensor,
    future_returns: torch.Tensor,
    cov_pred: torch.Tensor,
    cov_target: torch.Tensor,
    beta: float = 0.1,
    annualization_factor: int = 252,
    w_prev: torch.Tensor | None = None,
    turnover_penalty: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Block-Diagonal DFL loss.

    Combines task loss (tracking error) on the decision-relevant block with
    MSE loss on the full covariance matrix. This prevents catastrophic
    forgetting of off-block covariance structure when the selection set
    changes between rebalancing dates.

    L_BD = task_loss(w, w_idx, r) + β · RelMSE(Σ̂, Σ_target)

    where RelMSE = ‖Σ̂ - Σ_target‖²_F / ‖Σ_target‖²_F normalizes the MSE
    to be scale-invariant (daily covariance entries are O(1e-4), making raw
    MSE ~1e-10 vs task loss ~1e-3).

    Parameters
    ----------
    w_portfolio : Tensor (N,) — optimized portfolio weights
    w_index : Tensor (N,) — index weights
    future_returns : Tensor (T, N) — realized returns over evaluation period
    cov_pred : Tensor (N, N) — predicted covariance matrix
    cov_target : Tensor (N, N) — realized covariance (target for MSE)
    beta : float — weight of MSE regularization (0 = pure DFL)
    annualization_factor : int — trading days per year
    w_prev : Tensor (N,) | None — previous weights for turnover penalty
    turnover_penalty : float — λ for turnover regularization

    Returns
    -------
    total_loss : scalar Tensor — blended loss
    t_loss : scalar Tensor — task loss component (for logging)
    m_loss : scalar Tensor — relative MSE loss component (for logging)
    """
    t_loss = task_loss(
        w_portfolio, w_index, future_returns,
        annualization_factor=annualization_factor,
        w_prev=w_prev, turnover_penalty=turnover_penalty,
    )
    # Relative MSE: normalize by target Frobenius norm for scale invariance
    diff = cov_pred - cov_target
    target_norm_sq = (cov_target ** 2).sum() + 1e-12
    m_loss = (diff ** 2).sum() / target_norm_sq
    total_loss = t_loss + beta * m_loss
    return total_loss, t_loss, m_loss


def lodl_surrogate_loss(
    cov_pred: torch.Tensor,
    w_index: torch.Tensor,
    future_returns: torch.Tensor,
    solve_qp_fn,
    K: int,
    selection: str = "market_cap",
    n_samples: int = 10,
    perturbation_scale: float = 0.01,
    annualization_factor: int = 252,
) -> torch.Tensor:
    """LODL: locally optimized decision loss (Shah et al. 2022).

    Approximates the task loss gradient without differentiating through the QP.
    Samples perturbations around cov_pred, evaluates the non-differentiable
    task loss at each perturbation, then fits a linear surrogate whose
    gradient can be backpropagated through.

    Parameters
    ----------
    cov_pred : Tensor (N, N) — predicted covariance (requires_grad=True)
    w_index : Tensor (N,) — index weights
    future_returns : Tensor (T, N) — realized returns for task loss evaluation
    solve_qp_fn : callable — non-differentiable QP solver: (cov_np, w_idx_np, K, selection) -> w_np
    K : int — sparsity budget
    selection : str — stock selection method
    n_samples : int — number of perturbation samples
    perturbation_scale : float — scale of Gaussian perturbations (relative to cov_pred norm)
    annualization_factor : int — trading days per year

    Returns
    -------
    surrogate_loss : scalar Tensor — differentiable surrogate of task loss
    """
    N = cov_pred.shape[0]
    cov_detached = cov_pred.detach()
    cov_np_base = cov_detached.cpu().numpy()
    w_idx_np = w_index.detach().cpu().numpy()
    future_ret_np = future_returns.detach().cpu().numpy()

    scale = perturbation_scale * torch.norm(cov_detached).item() / N

    deltas = []
    losses = []

    for _ in range(n_samples):
        eps = torch.randn(N, N, device=cov_pred.device) * scale
        eps = (eps + eps.T) / 2  # symmetric perturbation

        cov_perturbed_np = cov_np_base + eps.cpu().numpy()
        cov_perturbed_np = (cov_perturbed_np + cov_perturbed_np.T) / 2
        np.fill_diagonal(cov_perturbed_np, np.maximum(np.diag(cov_perturbed_np), 1e-6))

        try:
            w_opt_np = solve_qp_fn(cov_perturbed_np, w_idx_np, K=K, selection=selection)
        except Exception:
            continue

        port_ret = future_ret_np @ w_opt_np
        idx_ret = future_ret_np @ w_idx_np
        te = annualization_factor * np.mean((port_ret - idx_ret) ** 2)

        deltas.append(eps.view(-1))
        losses.append(te)

    if len(deltas) < 3:
        return mse_loss(cov_pred, torch.tensor(cov_np_base, device=cov_pred.device))

    delta_matrix = torch.stack(deltas)  # (n_samples, N^2)
    loss_vector = torch.tensor(losses, dtype=torch.float32, device=cov_pred.device)  # (n_samples,)

    loss_centered = loss_vector - loss_vector.mean()
    coeffs = torch.linalg.lstsq(delta_matrix, loss_centered).solution  # (N^2,)

    diff_from_base = (cov_pred - cov_detached).view(-1)
    surrogate = (coeffs.detach() * diff_from_base).sum() + loss_vector.mean()

    return surrogate
