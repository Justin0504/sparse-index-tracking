"""Tests for QP optimization layer and standalone solver."""

import numpy as np
import torch
import pytest

from src.optimization.solver import solve_tracking_qp
from src.optimization.qp_layer import SparseTrackingQP


@pytest.fixture
def small_problem():
    """5-stock problem with known covariance."""
    np.random.seed(42)
    n = 5
    A = np.random.randn(20, n).astype(np.float64)
    cov = (A.T @ A) / 20 + 0.01 * np.eye(n)
    w_idx = np.array([0.3, 0.25, 0.2, 0.15, 0.1], dtype=np.float64)
    return cov, w_idx


def test_solver_feasibility(small_problem):
    cov, w_idx = small_problem
    w = solve_tracking_qp(cov, w_idx, K=3)
    assert abs(w.sum() - 1.0) < 1e-4
    assert np.all(w >= -1e-6)


def test_solver_sparsity(small_problem):
    """Lower K should produce sparser portfolios."""
    cov, w_idx = small_problem
    w_dense = solve_tracking_qp(cov, w_idx, K=5)
    w_sparse = solve_tracking_qp(cov, w_idx, K=2)
    nnz_dense = np.sum(np.abs(w_dense) > 1e-4)
    nnz_sparse = np.sum(np.abs(w_sparse) > 1e-4)
    assert nnz_sparse <= nnz_dense


def test_solver_k_controls_cardinality(small_problem):
    """K=2 should produce exactly 2 nonzero weights."""
    cov, w_idx = small_problem
    w = solve_tracking_qp(cov, w_idx, K=2)
    nnz = np.sum(np.abs(w) > 1e-6)
    assert nnz == 2


def test_solver_full_k_matches_dense(small_problem):
    """K=N should give similar result to unconstrained."""
    cov, w_idx = small_problem
    w = solve_tracking_qp(cov, w_idx, K=5)
    assert np.allclose(w, w_idx, atol=0.1)


def test_qp_layer_output(small_problem):
    cov_np, w_idx_np = small_problem
    n = cov_np.shape[0]
    qp = SparseTrackingQP(n)

    cov_t = torch.tensor(cov_np, dtype=torch.float32)
    w_idx_t = torch.tensor(w_idx_np, dtype=torch.float32)

    w_opt = qp.solve(cov_t, w_idx_t, K=3, hard=True)
    w_np = w_opt.detach().numpy()

    assert abs(w_np.sum() - 1.0) < 1e-2
    assert np.all(w_np >= -1e-3)
    # Hard top-K should give exactly 3 nonzero weights
    assert np.sum(np.abs(w_np) > 1e-4) == 3


def test_qp_layer_soft_topk(small_problem):
    """Soft top-K should be differentiable."""
    cov_np, w_idx_np = small_problem
    n = cov_np.shape[0]
    qp = SparseTrackingQP(n)

    cov_t = torch.tensor(cov_np, dtype=torch.float32, requires_grad=True)
    w_idx_t = torch.tensor(w_idx_np, dtype=torch.float32)

    w_opt = qp.solve(cov_t, w_idx_t, K=3, hard=False, temperature=0.01)
    loss = w_opt.sum()
    loss.backward()
    # Gradients should flow through soft top-K
    assert cov_t.grad is not None
