"""Standalone (non-differentiable) QP solver for two-stage baseline.

Selects top-K stocks by covariance-aware tracking score, then solves
a reduced QP to minimize tracking error against the full index.
"""

from __future__ import annotations

import cvxpy as cp
import numpy as np

from src.utils.logging import get_logger

log = get_logger(__name__)


def compute_tracking_scores_np(
    cov: np.ndarray,
    w_idx: np.ndarray,
) -> np.ndarray:
    """Covariance-aware stock selection score (numpy version).

    score_i = w_idx_i * (Σ_i · w_idx)² / Σ_ii
    """
    cov_with_idx = cov @ w_idx
    var_stocks = np.maximum(np.diag(cov), 1e-10)
    tracking_r2 = cov_with_idx ** 2 / var_stocks
    return w_idx * tracking_r2


def naive_index_weights(w_idx: np.ndarray, K: int) -> np.ndarray:
    """Naive baseline: top-K stocks by index weight, renormalized.

    No covariance estimation — just pick the largest index constituents
    and scale their weights to sum to 1.
    """
    n = len(w_idx)
    K = min(K, n)
    selected = np.argsort(w_idx)[-K:]
    w_full = np.zeros(n, dtype=np.float64)
    w_full[selected] = w_idx[selected]
    w_full = w_full / (w_full.sum() + 1e-10)
    return w_full


def solve_tracking_qp_l1(
    cov: np.ndarray,
    w_idx: np.ndarray,
    lam: float = 0.01,
    solver: str = "SCS",
    verbose: bool = False,
) -> np.ndarray:
    """Solve the ℓ₁-penalized index tracking QP (Formulation B).

    min_w  w^T Σ w - 2 w^T σ_idx + λ ||w||_1
    s.t.   1^T w = 1, w >= 0

    Since w >= 0, ||w||_1 = 1^T w = 1, so the ℓ₁ penalty is constant
    under the simplex constraint. To get actual sparsity, we use an
    elastic-net style formulation with an additional ℓ₁ penalty on
    individual weights (encouraging concentration):

    min_w  w^T Σ w - 2 w^T σ_idx + λ Σ_i max(w_i, 0)
    s.t.   1^T w = 1, w >= 0

    In practice, we add a squared ℓ₂ penalty λ||w||₂² which shrinks
    small weights toward zero more aggressively than large ones.
    """
    n = cov.shape[0]
    w = cp.Variable(n)
    cov_sym = 0.5 * (cov + cov.T)
    sigma_idx = cov @ w_idx

    objective = cp.Minimize(
        cp.quad_form(w, cov_sym, assume_PSD=True)
        - 2 * sigma_idx @ w
        + lam * cp.sum_squares(w)
    )
    constraints = [cp.sum(w) == 1, w >= 0]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=solver, verbose=verbose)

    if prob.status in ("optimal", "optimal_inaccurate"):
        w_val = np.clip(w.value, 0, None).astype(np.float64)
        w_val = w_val / (w_val.sum() + 1e-10)
    else:
        log.warning(f"L1 QP status: {prob.status} — falling back to equal weight")
        w_val = np.ones(n, dtype=np.float64) / n

    return w_val


def solve_tracking_qp(
    cov: np.ndarray,
    w_idx: np.ndarray,
    K: int,
    solver: str = "SCS",
    verbose: bool = False,
    selection: str = "market_cap",
) -> np.ndarray:
    """Solve the sparse index tracking QP (non-differentiable).

    Phase 1: Select top-K stocks by covariance-aware tracking score.
    Phase 2: Solve reduced QP to minimize tracking error against full index.

    Parameters
    ----------
    cov : ndarray (N, N) — covariance matrix
    w_idx : ndarray (N,) — index weights
    K : int — target number of stocks

    Returns
    -------
    w_opt : ndarray (N,) — sparse optimal weights
    """
    n = cov.shape[0]
    K = min(K, n)

    # Phase 1: Select stocks
    if selection == "tracking_score":
        scores = compute_tracking_scores_np(cov, w_idx)
    else:
        scores = w_idx.copy()
    selected = np.argsort(scores)[-K:]
    selected_sorted = np.sort(selected)

    # Phase 2: Reduced QP
    cov_SS = cov[np.ix_(selected_sorted, selected_sorted)]
    b = cov[selected_sorted] @ w_idx

    w_S = cp.Variable(K)
    objective = cp.Minimize(cp.quad_form(w_S, cov_SS, assume_PSD=True) - 2 * b @ w_S)
    constraints = [cp.sum(w_S) == 1, w_S >= 0]

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=solver, verbose=verbose)

    if prob.status in ("optimal", "optimal_inaccurate"):
        w_S_val = np.clip(w_S.value, 0, None).astype(np.float64)
    else:
        log.warning(f"QP solver status: {prob.status} — falling back to equal weight")
        w_S_val = np.ones(K, dtype=np.float64) / K

    w_full = np.zeros(n, dtype=np.float64)
    w_full[selected_sorted] = w_S_val
    w_full = w_full / (w_full.sum() + 1e-10)

    return w_full
