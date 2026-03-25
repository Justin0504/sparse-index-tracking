"""Differentiable QP layer for sparse index tracking.

Two-phase approach:
  1. Select top-K stocks using a covariance-aware tracking score
  2. Solve a REDUCED K-dim QP over only those stocks

Stock selection score: score_i = (Σ̂_i · w_idx)² / Σ̂_ii
This measures each stock's tracking R² — how much of the index's variance
it can explain. Covariance-aware selection lets the model influence WHICH
stocks are chosen (not just their weights), amplifying the DFL advantage.

Training uses unrolled projected gradient descent for strong gradient flow.
Evaluation uses cvxpylayers for exact solutions.
"""

from __future__ import annotations

import cvxpy as cp
import torch
from cvxpylayers.torch import CvxpyLayer

from src.utils.logging import get_logger

log = get_logger(__name__)


def _project_simplex(w: torch.Tensor) -> torch.Tensor:
    """Project onto the probability simplex {w : w >= 0, sum(w) = 1}.

    Algorithm from Duchi et al. (2008).
    """
    n = w.shape[0]
    sorted_w, _ = torch.sort(w, descending=True)
    cumsum = torch.cumsum(sorted_w, dim=0)
    rho_candidates = sorted_w - (cumsum - 1.0) / torch.arange(1, n + 1, device=w.device, dtype=w.dtype)
    rho = (rho_candidates > 0).sum() - 1
    theta = (cumsum[rho] - 1.0) / (rho.float() + 1.0)
    return torch.clamp(w - theta, min=0.0)


def compute_tracking_scores(
    cov: torch.Tensor,
    w_idx: torch.Tensor,
) -> torch.Tensor:
    """Covariance-aware stock selection score for index tracking.

    score_i = w_idx_i * (Σ̂_i · w_idx)² / Σ̂_ii

    Combines market cap (w_idx_i) with tracking R² to select stocks
    that are both large AND good trackers of the index.

    Parameters
    ----------
    cov : Tensor (N, N) — predicted covariance
    w_idx : Tensor (N,) — index weights

    Returns
    -------
    scores : Tensor (N,) — selection scores (higher = better)
    """
    cov_with_idx = cov @ w_idx                  # (N,) cov of each stock with index
    var_stocks = torch.diag(cov).clamp(min=1e-10)  # (N,) variance of each stock
    tracking_r2 = cov_with_idx ** 2 / var_stocks    # (N,) tracking R²
    scores = w_idx * tracking_r2                     # weight by market cap
    return scores


class SparseTrackingQP:
    """Differentiable sparse index tracking QP.

    Phase 1: Select top-K stocks by covariance-aware tracking score.
    Phase 2: Solve reduced QP to minimize tracking error against full index.
    """

    def __init__(self, n_stocks: int):
        self.n_stocks = n_stocks
        self._cvxpy_cache: dict[int, CvxpyLayer] = {}

    def _get_cvxpy_layer(self, k: int) -> CvxpyLayer:
        """Build or retrieve a cached K-dim QP layer."""
        if k not in self._cvxpy_cache:
            w = cp.Variable(k)
            Q_param = cp.Parameter((k, k))
            q_param = cp.Parameter(k)
            objective = cp.Minimize(cp.sum_squares(Q_param @ w - q_param))
            constraints = [cp.sum(w) == 1, w >= 0]
            problem = cp.Problem(objective, constraints)
            assert problem.is_dcp(dpp=True)
            self._cvxpy_cache[k] = CvxpyLayer(problem, parameters=[Q_param, q_param], variables=[w])
        return self._cvxpy_cache[k]

    def solve(
        self,
        cov: torch.Tensor,
        w_idx: torch.Tensor,
        K: int,
        temperature: float = 0.01,
        hard: bool = False,
        solver_args: dict | None = None,
        selection: str = "market_cap",
    ) -> torch.Tensor:
        """Select top-K stocks then solve reduced tracking QP.

        Parameters
        ----------
        cov : Tensor (N, N) — predicted covariance matrix (PSD)
        w_idx : Tensor (N,) — index weights (sum to 1)
        K : int — number of stocks to hold
        hard : bool — if True, use exact QP (evaluation); else unrolled (training)
        selection : str — "market_cap" (stable) or "tracking_score" (covariance-aware)
        """
        N = cov.shape[0]
        K = min(K, N)

        # Phase 1: Select top-K stocks
        if selection == "tracking_score":
            scores = compute_tracking_scores(cov.detach(), w_idx.detach())
        else:
            scores = w_idx.detach()
        _, selected = torch.topk(scores, K)
        selected_sorted, _ = torch.sort(selected)

        # Extract submatrices (gradients flow through cov)
        cov_SS = cov[selected_sorted][:, selected_sorted]  # (K, K)
        cov_SN = cov[selected_sorted]                       # (K, N)
        b = cov_SN @ w_idx  # (K,) — cov of selected stocks with index portfolio

        if hard:
            w_S = self._solve_reduced_exact(cov_SS, b, K, solver_args)
        else:
            w_S = self._solve_reduced_unrolled(cov_SS, b, K)

        # Embed back to N-dimensional vector
        w_full = torch.zeros(N, device=cov.device, dtype=w_S.dtype)
        w_full[selected_sorted] = w_S
        return w_full

    def _solve_reduced_exact(
        self,
        cov_SS: torch.Tensor,
        b: torch.Tensor,
        K: int,
        solver_args: dict | None = None,
    ) -> torch.Tensor:
        """Exact reduced QP via cvxpylayers."""
        solver_args = solver_args or {"max_iters": 10_000}

        Q = 0.5 * (cov_SS + cov_SS.T)
        diag_max = torch.max(torch.diag(Q)).clamp(min=1e-8)
        Q = Q + 1e-4 * diag_max * torch.eye(K, device=Q.device, dtype=Q.dtype)

        try:
            L = torch.linalg.cholesky(Q)
        except torch.linalg.LinAlgError:
            Q = Q + 1e-2 * diag_max * torch.eye(K, device=Q.device, dtype=Q.dtype)
            L = torch.linalg.cholesky(Q)

        P = L.T
        c = torch.linalg.solve_triangular(L, b.unsqueeze(-1), upper=False).squeeze(-1)

        layer = self._get_cvxpy_layer(K)
        (w_S,) = layer(P, c, solver_args=solver_args)
        return w_S

    def _solve_reduced_unrolled(
        self,
        cov_SS: torch.Tensor,
        b: torch.Tensor,
        K: int,
        n_iters: int = 100,
        step_size: float = 0.5,
    ) -> torch.Tensor:
        """Unrolled projected gradient descent on the reduced QP."""
        Q = 0.5 * (cov_SS + cov_SS.T)
        diag_max = torch.max(torch.diag(Q)).clamp(min=1e-8)
        Q = Q + 1e-4 * diag_max * torch.eye(K, device=Q.device, dtype=Q.dtype)

        effective_step = step_size / (diag_max + 1e-8)
        w = torch.ones(K, device=Q.device, dtype=Q.dtype) / K

        for _ in range(n_iters):
            grad = 2.0 * (Q @ w - b)
            w = w - effective_step * grad
            w = _project_simplex(w)

        return w
