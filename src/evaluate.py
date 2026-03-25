"""Evaluation metrics and statistical tests for index-tracking models."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import scipy.stats as stats
import torch
from torch.utils.data import DataLoader

from .models import DFLModel, TwoStageModel


# ---------------------------------------------------------------------------
# Portfolio evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model: TwoStageModel | DFLModel,
    test_loader: DataLoader,
    device: torch.device | None = None,
) -> Dict[str, np.ndarray | float]:
    """Evaluate a trained model on a test set.

    Computes the per-period tracking difference ``w' r - r_b`` for every
    sample in the test loader, then derives summary statistics.

    Args:
        model: Trained ``TwoStageModel`` or ``DFLModel``.
        test_loader: DataLoader for test data.
        device: Compute device.

    Returns:
        Dictionary with keys:

        - ``tracking_diffs``: Array of per-period tracking differences.
        - ``mse_te``: Mean squared tracking error (daily).
        - ``annualized_te``: Annualized tracking error (std × √252).
    """
    if device is None:
        device = torch.device("cpu")
    model = model.to(device)
    model.eval()

    diffs: list[np.ndarray] = []

    with torch.no_grad():
        for batch in test_loader:
            features, target_return, target_index, R_scaled, _ = [
                t.to(device) for t in batch
            ]

            if isinstance(model, TwoStageModel):
                c_hat = model.predictor(features)
                weights = model.qp_layer(c_hat, R_scaled)
            else:
                c_hat = model.predictor(features)
                weights = model.qp_layer(c_hat, R_scaled)

            portfolio_ret = (weights * target_return).sum(dim=-1)
            diff = (portfolio_ret - target_index).cpu().numpy()
            diffs.append(diff)

    all_diffs = np.concatenate(diffs, axis=0)

    return {
        "tracking_diffs": all_diffs,
        "mse_te": float(np.mean(all_diffs ** 2)),
        "annualized_te": float(np.std(all_diffs, ddof=1) * np.sqrt(252)),
    }


# ---------------------------------------------------------------------------
# Statistical testing
# ---------------------------------------------------------------------------

def diebold_mariano_test(
    te_baseline: np.ndarray,
    te_dfl: np.ndarray,
) -> Dict[str, float]:
    """Diebold–Mariano test for equal predictive accuracy.

    Tests whether the DFL model achieves significantly lower squared
    tracking error than the two-stage baseline.

    H₀: E[d_t] = 0  where  d_t = te_baseline_t² − te_dfl_t²
    H₁: E[d_t] > 0  (DFL is better)

    Args:
        te_baseline: Per-period tracking differences for the baseline model.
        te_dfl: Per-period tracking differences for the DFL model.

    Returns:
        Dictionary with ``dm_statistic``, ``p_value`` (one-sided), and
        ``loss_differential_mean``.
    """
    d = te_baseline ** 2 - te_dfl ** 2  # loss differential
    n = len(d)
    d_bar = float(np.mean(d))
    d_std = float(np.std(d, ddof=1)) / np.sqrt(n)
    dm_stat = d_bar / (d_std + 1e-12)
    # One-sided: H₁: DFL has lower squared TE → d_bar > 0
    p_value = float(1.0 - stats.t.cdf(dm_stat, df=n - 1))
    return {
        "dm_statistic": dm_stat,
        "p_value": p_value,
        "loss_differential_mean": d_bar,
    }


def compute_te_reduction(te_baseline: float, te_dfl: float) -> float:
    """Compute the percentage tracking-error reduction of DFL over baseline.

    Args:
        te_baseline: Annualized tracking error of the baseline.
        te_dfl: Annualized tracking error of the DFL model.

    Returns:
        Percentage reduction (positive = DFL is better).
    """
    return 100.0 * (te_baseline - te_dfl) / (te_baseline + 1e-12)


# ---------------------------------------------------------------------------
# Reporting helper
# ---------------------------------------------------------------------------

def print_comparison(
    results_baseline: Dict,
    results_dfl: Dict,
    dm_result: Dict,
) -> None:
    """Print a formatted comparison table to stdout."""
    ate_base = results_baseline["annualized_te"]
    ate_dfl = results_dfl["annualized_te"]
    reduction = compute_te_reduction(ate_base, ate_dfl)

    print("\n" + "=" * 60)
    print("  Sparse Index Tracking – Model Comparison")
    print("=" * 60)
    print(f"  Two-stage MSE baseline  ATE: {ate_base:.4f}  ({ate_base*100:.2f}% p.a.)")
    print(f"  Neural + DFL            ATE: {ate_dfl:.4f}  ({ate_dfl*100:.2f}% p.a.)")
    print(f"  TE reduction            : {reduction:+.1f}%")
    print(f"  DM statistic            : {dm_result['dm_statistic']:.3f}")
    print(f"  p-value (one-sided)     : {dm_result['p_value']:.4f}")
    sig = "***" if dm_result["p_value"] < 0.01 else ("**" if dm_result["p_value"] < 0.05 else "")
    print(f"  Significance            : {sig or '(not significant)'}")
    print("=" * 60)
