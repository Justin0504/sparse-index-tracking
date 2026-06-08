"""Run additional baselines requested by professor:

1. Naive baseline: top-K by market cap, weights = index weights renormalized
2. Ledoit-Wolf shrinkage: sample cov + optimal shrinkage -> QP
3. Shrinkage alpha DFL: learn shrinkage intensity with task loss

Usage:
    python scripts/run_baselines.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import SP500DataLoader, FeatureBuilder, RollingWindowSplitter
from src.models import ShrinkageCovarianceModel
from src.optimization.solver import naive_index_weights, solve_tracking_qp
from src.optimization.qp_layer import SparseTrackingQP
from src.training.losses import task_loss
from src.evaluation.metrics import (
    tracking_error,
    max_tracking_deviation,
    turnover,
    information_ratio,
    sparsity,
)
from src.utils import set_seed, get_logger

log = get_logger(__name__)

# ── Config ──
SEED = 42
SPARSITY_LEVELS = [5, 10, 20, 30, 50]
LOOKBACK = 63
DEVICE = "cpu"


def _compute_index_weights(prices: pd.DataFrame) -> np.ndarray:
    last_prices = prices.iloc[-1].values
    w = last_prices / last_prices.sum()
    return w.astype(np.float32)


def _get_monthly_rebal(dates: pd.DatetimeIndex) -> list[pd.Timestamp]:
    s = pd.Series(range(len(dates)), index=dates)
    return list(s.resample("ME").last().dropna().index)


def load_data():
    """Load S&P 500 data and set up rolling folds."""
    set_seed(SEED)
    loader = SP500DataLoader(
        start_date="2006-01-01",
        end_date="2025-12-31",
        top_n=100,
        cache_dir="data/cache",
    )
    prices, index_prices, tickers = loader.load()
    fb = FeatureBuilder(vol_window=21, momentum_window=63)
    returns = fb.compute_returns(prices)

    splitter = RollingWindowSplitter(train_years=3, val_months=6, test_years=1, step_years=1)
    folds = splitter.split(returns.index)

    return prices, returns, tickers, folds, fb


def evaluate_portfolio(
    weights_history: list[np.ndarray],
    rebal_test: list,
    returns: pd.DataFrame,
    w_index: np.ndarray,
) -> dict | None:
    """Compute metrics for a series of portfolio weights."""
    portfolio_rets = []
    index_rets = []

    for i, date in enumerate(rebal_test):
        loc = returns.index.get_loc(date)
        next_loc = (
            returns.index.get_loc(rebal_test[i + 1])
            if i + 1 < len(rebal_test)
            else len(returns)
        )
        if next_loc <= loc:
            continue

        w_opt = weights_history[i]
        holding_rets = returns.iloc[loc:next_loc].values
        port_r = holding_rets @ w_opt
        idx_r = holding_rets @ w_index
        portfolio_rets.extend(port_r.tolist())
        index_rets.extend(idx_r.tolist())

    if not portfolio_rets:
        return None

    portfolio_rets = np.array(portfolio_rets)
    index_rets = np.array(index_rets)

    te = tracking_error(portfolio_rets, index_rets, 252)
    mtd = max_tracking_deviation(portfolio_rets, index_rets)
    ir = information_ratio(portfolio_rets, index_rets, 252)
    avg_sp = float(np.mean([sparsity(w) for w in weights_history]))
    avg_to = (
        float(np.mean([turnover(weights_history[j], weights_history[j + 1])
                        for j in range(len(weights_history) - 1)]))
        if len(weights_history) > 1 else 0.0
    )

    return {
        "tracking_error": te,
        "max_tracking_deviation": mtd,
        "information_ratio": ir,
        "avg_sparsity": avg_sp,
        "avg_turnover": avg_to,
        "portfolio_returns": portfolio_rets.tolist(),
        "index_returns": index_rets.tolist(),
    }


# ═══════════════════════════════════════════════════════
# Baseline 1: Naive (index weights renormalized)
# ═══════════════════════════════════════════════════════

def run_naive_baseline(prices, returns, folds, fb):
    """Top-K by market cap, weights = renormalized index weights. No model."""
    log.info("\n" + "=" * 60)
    log.info("BASELINE: Naive Index Weights")
    log.info("=" * 60)

    all_results = []
    min_offset = max(21, 63) + 1

    for fold_info in folds:
        train_mask = (returns.index >= fold_info.train_start) & (returns.index <= fold_info.train_end)
        ret_train = returns.loc[train_mask]
        ret_test = returns.loc[
            (returns.index >= fold_info.test_start) & (returns.index <= fold_info.test_end)
        ]

        train_prices = prices.loc[prices.index.isin(ret_train.index)]
        w_index = _compute_index_weights(train_prices)

        rebal_test = [d for d in _get_monthly_rebal(ret_test.index)
                      if d in returns.index and returns.index.get_loc(d) >= min_offset]

        if not rebal_test:
            continue

        for K in SPARSITY_LEVELS:
            # Same weights at every rebalancing date — no model needed
            w_naive = naive_index_weights(w_index, K)
            weights_history = [w_naive] * len(rebal_test)

            metrics = evaluate_portfolio(weights_history, rebal_test, returns, w_index)
            if metrics is None:
                continue

            result = {
                "fold": fold_info.fold,
                "K": K,
                "model": "naive",
                "loss": "none",
                "test_start": str(fold_info.test_start.date()),
                "test_end": str(fold_info.test_end.date()),
                **metrics,
            }
            all_results.append(result)
            log.info(f"Fold {fold_info.fold} K={K} | TE={metrics['tracking_error']:.4f}")

    return all_results


# ═══════════════════════════════════════════════════════
# Baseline 2: Ledoit-Wolf shrinkage + QP
# ═══════════════════════════════════════════════════════

def run_ledoit_wolf_baseline(prices, returns, folds, fb):
    """Ledoit-Wolf shrinkage covariance -> QP. No learning."""
    log.info("\n" + "=" * 60)
    log.info("BASELINE: Ledoit-Wolf Shrinkage + QP")
    log.info("=" * 60)

    all_results = []
    min_offset = max(21, 63) + 1

    for fold_info in folds:
        train_mask = (returns.index >= fold_info.train_start) & (returns.index <= fold_info.train_end)
        ret_train = returns.loc[train_mask]
        ret_test = returns.loc[
            (returns.index >= fold_info.test_start) & (returns.index <= fold_info.test_end)
        ]

        train_prices = prices.loc[prices.index.isin(ret_train.index)]
        w_index = _compute_index_weights(train_prices)

        rebal_test = [d for d in _get_monthly_rebal(ret_test.index)
                      if d in returns.index and returns.index.get_loc(d) >= min_offset]

        if not rebal_test:
            continue

        for K in SPARSITY_LEVELS:
            weights_history = []
            alphas = []

            for date in rebal_test:
                cov_lw, alpha = fb.compute_ledoit_wolf_covariance(returns, date, lookback=LOOKBACK)
                alphas.append(alpha)

                try:
                    w_opt = solve_tracking_qp(cov_lw, w_index, K=K, selection="market_cap")
                except Exception as e:
                    log.warning(f"QP failed at {date}: {e}")
                    w_opt = naive_index_weights(w_index, K)

                weights_history.append(w_opt)

            metrics = evaluate_portfolio(weights_history, rebal_test, returns, w_index)
            if metrics is None:
                continue

            result = {
                "fold": fold_info.fold,
                "K": K,
                "model": "ledoit_wolf",
                "loss": "none",
                "avg_shrinkage_alpha": float(np.mean(alphas)),
                "test_start": str(fold_info.test_start.date()),
                "test_end": str(fold_info.test_end.date()),
                **metrics,
            }
            all_results.append(result)
            log.info(
                f"Fold {fold_info.fold} K={K} | TE={metrics['tracking_error']:.4f} | "
                f"avg_alpha={np.mean(alphas):.3f}"
            )

    return all_results


# ═══════════════════════════════════════════════════════
# Baseline 3: Shrinkage alpha DFL (light-touch)
# ═══════════════════════════════════════════════════════

def run_shrinkage_dfl(prices, returns, folds, fb):
    """Learn shrinkage intensity alpha via task loss. Single learnable parameter."""
    log.info("\n" + "=" * 60)
    log.info("BASELINE: Shrinkage Alpha DFL (light-touch)")
    log.info("=" * 60)

    all_results = []
    min_offset = max(21, 63) + 1
    n_stocks = returns.shape[1]

    for fold_info in folds:
        train_mask = (returns.index >= fold_info.train_start) & (returns.index <= fold_info.train_end)
        val_mask = (returns.index >= fold_info.val_start) & (returns.index <= fold_info.val_end)
        test_mask = (returns.index >= fold_info.test_start) & (returns.index <= fold_info.test_end)

        ret_train = returns.loc[train_mask]
        ret_test = returns.loc[test_mask]

        train_prices = prices.loc[prices.index.isin(ret_train.index)]
        w_index = _compute_index_weights(train_prices)
        w_idx_t = torch.tensor(w_index, dtype=torch.float32, device=DEVICE)

        # Rebalancing dates
        rebal_train = [d for d in _get_monthly_rebal(ret_train.index)
                       if d in returns.index and returns.index.get_loc(d) >= min_offset]
        rebal_test = [d for d in _get_monthly_rebal(ret_test.index)
                      if d in returns.index and returns.index.get_loc(d) >= min_offset]

        if not rebal_train or not rebal_test:
            continue

        for K in SPARSITY_LEVELS:
            log.info(f"\nFold {fold_info.fold} K={K}: Training shrinkage alpha...")

            model = ShrinkageCovarianceModel(n_stocks, init_alpha=0.5).to(DEVICE)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
            qp_layer = SparseTrackingQP(n_stocks)

            forward_window = 21

            # Train: optimize alpha over training rebalancing dates
            best_alpha = 0.5
            best_loss = float("inf")

            for epoch in range(100):
                model.train()
                epoch_loss = 0.0
                n_valid = 0

                for date in rebal_train:
                    loc = returns.index.get_loc(date)
                    if loc + forward_window > len(returns):
                        continue

                    # Compute sample cov and set it
                    cov_np = fb.compute_realized_covariance(returns, date, lookback=LOOKBACK)
                    cov_t = torch.tensor(cov_np, dtype=torch.float32, device=DEVICE)
                    model.set_sample_covariance(cov_t)

                    # Forward: apply shrinkage
                    dummy_features = torch.zeros(n_stocks, 1, device=DEVICE)
                    cov_shrunk = model(dummy_features)

                    # Solve QP
                    try:
                        w_opt = qp_layer.solve(
                            cov_shrunk, w_idx_t, K=K, hard=False, selection="market_cap"
                        )
                    except Exception:
                        continue

                    # Task loss
                    future_ret = torch.tensor(
                        returns.iloc[loc: loc + forward_window].values,
                        dtype=torch.float32, device=DEVICE,
                    )
                    loss = task_loss(w_opt, w_idx_t, future_ret)
                    loss.backward()
                    epoch_loss += loss.item()
                    n_valid += 1

                if n_valid > 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    avg_loss = epoch_loss / n_valid

                    if avg_loss < best_loss:
                        best_loss = avg_loss
                        best_alpha = model.alpha.item()

                if (epoch + 1) % 20 == 0:
                    log.info(
                        f"  Epoch {epoch+1}/100 | loss={epoch_loss/max(n_valid,1):.6f} | "
                        f"alpha={model.alpha.item():.4f}"
                    )

            log.info(f"  Best alpha={best_alpha:.4f}")

            # Test: use learned alpha
            model.eval()
            weights_history = []

            for date in rebal_test:
                cov_np = fb.compute_realized_covariance(returns, date, lookback=LOOKBACK)
                cov_t = torch.tensor(cov_np, dtype=torch.float32, device=DEVICE)
                model.set_sample_covariance(cov_t)

                with torch.no_grad():
                    dummy_features = torch.zeros(n_stocks, 1, device=DEVICE)
                    cov_shrunk = model(dummy_features).cpu().numpy()

                try:
                    w_opt = solve_tracking_qp(cov_shrunk, w_index, K=K, selection="market_cap")
                except Exception:
                    w_opt = naive_index_weights(w_index, K)

                weights_history.append(w_opt)

            metrics = evaluate_portfolio(weights_history, rebal_test, returns, w_index)
            if metrics is None:
                continue

            result = {
                "fold": fold_info.fold,
                "K": K,
                "model": "shrinkage_dfl",
                "loss": "task",
                "learned_alpha": best_alpha,
                "test_start": str(fold_info.test_start.date()),
                "test_end": str(fold_info.test_end.date()),
                **metrics,
            }
            all_results.append(result)
            log.info(
                f"Fold {fold_info.fold} K={K} | TE={metrics['tracking_error']:.4f} | "
                f"alpha={best_alpha:.4f}"
            )

    return all_results


# ═══════════════════════════════════════════════════════
# Baseline 4: Validation-Tuned Shrinkage (grid search α)
# ═══════════════════════════════════════════════════════

def _eval_alpha_on_dates(alpha, fb, returns, rebal_dates, w_index, K, n_stocks, lookback=63):
    """Evaluate a fixed shrinkage alpha on a set of rebalancing dates.

    Returns mean realized tracking error (no gradients, no QP layer).
    """
    forward_window = 21
    squared_diffs = []

    for date in rebal_dates:
        loc = returns.index.get_loc(date)
        if loc + forward_window > len(returns):
            continue

        # Compute shrunk covariance
        cov_np = fb.compute_realized_covariance(returns, date, lookback=lookback)
        mu = np.trace(cov_np) / n_stocks
        cov_shrunk = (1 - alpha) * cov_np + alpha * mu * np.eye(n_stocks)

        # Solve QP (non-differentiable)
        try:
            w_opt = solve_tracking_qp(cov_shrunk, w_index, K=K, selection="market_cap")
        except Exception:
            continue

        # Realized tracking error over forward window
        future_ret = returns.iloc[loc: loc + forward_window].values
        port_ret = future_ret @ w_opt
        idx_ret = future_ret @ w_index
        diff = port_ret - idx_ret
        squared_diffs.extend((diff ** 2).tolist())

    if not squared_diffs:
        return float("inf")
    return 252.0 * np.mean(squared_diffs)  # annualized MSE proxy


def run_val_tuned_baseline(prices, returns, folds, fb):
    """Grid search over 50 α values on validation TE, evaluate best α on test.

    This is the critical baseline: if it matches DFL, the differentiable QP
    machinery is unnecessary.
    """
    log.info("\n" + "=" * 60)
    log.info("BASELINE: Validation-Tuned Shrinkage (Grid Search α)")
    log.info("=" * 60)

    all_results = []
    min_offset = max(21, 63) + 1
    n_stocks = returns.shape[1]
    alphas_grid = np.linspace(0.01, 0.99, 50)

    for fold_info in folds:
        train_mask = (returns.index >= fold_info.train_start) & (returns.index <= fold_info.train_end)
        val_mask = (returns.index >= fold_info.val_start) & (returns.index <= fold_info.val_end)
        test_mask = (returns.index >= fold_info.test_start) & (returns.index <= fold_info.test_end)

        ret_train = returns.loc[train_mask]
        ret_val = returns.loc[val_mask]
        ret_test = returns.loc[test_mask]

        train_prices = prices.loc[prices.index.isin(ret_train.index)]
        w_index = _compute_index_weights(train_prices)

        rebal_val = [d for d in _get_monthly_rebal(ret_val.index)
                     if d in returns.index and returns.index.get_loc(d) >= min_offset]
        rebal_test = [d for d in _get_monthly_rebal(ret_test.index)
                      if d in returns.index and returns.index.get_loc(d) >= min_offset]

        if not rebal_val or not rebal_test:
            continue

        for K in SPARSITY_LEVELS:
            log.info(f"\nFold {fold_info.fold} K={K}: Grid searching α...")

            # Grid search on validation set
            best_alpha = 0.5
            best_val_te = float("inf")

            for alpha in alphas_grid:
                val_te = _eval_alpha_on_dates(alpha, fb, returns, rebal_val, w_index, K, n_stocks)
                if val_te < best_val_te:
                    best_val_te = val_te
                    best_alpha = alpha

            log.info(f"  Best α={best_alpha:.3f} (val TE proxy={best_val_te:.6f})")

            # Evaluate on test with best alpha
            weights_history = []
            for date in rebal_test:
                cov_np = fb.compute_realized_covariance(returns, date, lookback=LOOKBACK)
                mu = np.trace(cov_np) / n_stocks
                cov_shrunk = (1 - best_alpha) * cov_np + best_alpha * mu * np.eye(n_stocks)

                try:
                    w_opt = solve_tracking_qp(cov_shrunk, w_index, K=K, selection="market_cap")
                except Exception:
                    w_opt = naive_index_weights(w_index, K)
                weights_history.append(w_opt)

            metrics = evaluate_portfolio(weights_history, rebal_test, returns, w_index)
            if metrics is None:
                continue

            result = {
                "fold": fold_info.fold,
                "K": K,
                "model": "val_tuned_shrinkage",
                "loss": "grid_search_val_te",
                "best_alpha": float(best_alpha),
                "val_te_proxy": float(best_val_te),
                "test_start": str(fold_info.test_start.date()),
                "test_end": str(fold_info.test_end.date()),
                **metrics,
            }
            all_results.append(result)
            log.info(
                f"Fold {fold_info.fold} K={K} | TE={metrics['tracking_error']:.4f} | "
                f"α={best_alpha:.3f}"
            )

    return all_results


# ═══════════════════════════════════════════════════════
# Baseline 5: SPO+ Surrogate Loss
# ═══════════════════════════════════════════════════════

def _spo_plus_loss_alpha(alpha, fb, returns, rebal_dates, w_index, K, n_stocks, lookback=63):
    """SPO+ regret for a given shrinkage alpha.

    SPO+ loss = max_{w in W} (2*c_hat - c)^T w - (c_hat)^T w_hat

    For tracking error: c = diag(Sigma) contribution to TE.
    We approximate: SPO+ regret ≈ TE(w*(Sigma_hat), true_Sigma) - TE(w*(true_Sigma), true_Sigma)
    i.e., the decision regret from using predicted vs oracle covariance.
    """
    forward_window = 21
    regrets = []

    for date in rebal_dates:
        loc = returns.index.get_loc(date)
        if loc + forward_window > len(returns):
            continue

        # Predicted covariance (with candidate alpha)
        cov_np = fb.compute_realized_covariance(returns, date, lookback=lookback)
        mu = np.trace(cov_np) / n_stocks
        cov_hat = (1 - alpha) * cov_np + alpha * mu * np.eye(n_stocks)

        # "True" covariance (realized over forward window)
        future_ret = returns.iloc[loc: loc + forward_window].values
        cov_true = np.cov(future_ret.T) + 1e-6 * np.eye(n_stocks)

        # Decision from predicted cov
        try:
            w_hat = solve_tracking_qp(cov_hat, w_index, K=K, selection="market_cap")
        except Exception:
            continue

        # Oracle decision from true cov
        try:
            w_star = solve_tracking_qp(cov_true, w_index, K=K, selection="market_cap")
        except Exception:
            continue

        # Decision regret: TE(w_hat, true) - TE(w_star, true)
        diff_hat = future_ret @ w_hat - future_ret @ w_index
        diff_star = future_ret @ w_star - future_ret @ w_index
        te_hat = np.mean(diff_hat ** 2)
        te_star = np.mean(diff_star ** 2)
        regrets.append(te_hat - te_star)

    if not regrets:
        return float("inf")
    return np.mean(regrets)


def run_spo_plus_baseline(prices, returns, folds, fb):
    """Find α that minimizes SPO+ regret on training data, evaluate on test."""
    log.info("\n" + "=" * 60)
    log.info("BASELINE: SPO+ Surrogate Loss")
    log.info("=" * 60)

    all_results = []
    min_offset = max(21, 63) + 1
    n_stocks = returns.shape[1]
    alphas_grid = np.linspace(0.01, 0.99, 50)

    for fold_info in folds:
        train_mask = (returns.index >= fold_info.train_start) & (returns.index <= fold_info.train_end)
        test_mask = (returns.index >= fold_info.test_start) & (returns.index <= fold_info.test_end)

        ret_train = returns.loc[train_mask]
        ret_test = returns.loc[test_mask]

        train_prices = prices.loc[prices.index.isin(ret_train.index)]
        w_index = _compute_index_weights(train_prices)

        rebal_train = [d for d in _get_monthly_rebal(ret_train.index)
                       if d in returns.index and returns.index.get_loc(d) >= min_offset]
        rebal_test = [d for d in _get_monthly_rebal(ret_test.index)
                      if d in returns.index and returns.index.get_loc(d) >= min_offset]

        if not rebal_train or not rebal_test:
            continue

        for K in SPARSITY_LEVELS:
            log.info(f"\nFold {fold_info.fold} K={K}: SPO+ grid search...")

            # Grid search: minimize SPO+ regret on training dates
            best_alpha = 0.5
            best_regret = float("inf")

            for alpha in alphas_grid:
                regret = _spo_plus_loss_alpha(alpha, fb, returns, rebal_train, w_index, K, n_stocks)
                if regret < best_regret:
                    best_regret = regret
                    best_alpha = alpha

            log.info(f"  Best α={best_alpha:.3f} (train regret={best_regret:.6f})")

            # Evaluate on test with best alpha
            weights_history = []
            for date in rebal_test:
                cov_np = fb.compute_realized_covariance(returns, date, lookback=LOOKBACK)
                mu = np.trace(cov_np) / n_stocks
                cov_shrunk = (1 - best_alpha) * cov_np + best_alpha * mu * np.eye(n_stocks)

                try:
                    w_opt = solve_tracking_qp(cov_shrunk, w_index, K=K, selection="market_cap")
                except Exception:
                    w_opt = naive_index_weights(w_index, K)
                weights_history.append(w_opt)

            metrics = evaluate_portfolio(weights_history, rebal_test, returns, w_index)
            if metrics is None:
                continue

            result = {
                "fold": fold_info.fold,
                "K": K,
                "model": "spo_plus",
                "loss": "spo_plus_regret",
                "best_alpha": float(best_alpha),
                "train_regret": float(best_regret),
                "test_start": str(fold_info.test_start.date()),
                "test_end": str(fold_info.test_end.date()),
                **metrics,
            }
            all_results.append(result)
            log.info(
                f"Fold {fold_info.fold} K={K} | TE={metrics['tracking_error']:.4f} | "
                f"α={best_alpha:.3f}"
            )

    return all_results


# ═══════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════
# Baseline 6: POET (Fan, Liao & Mincheva, 2013)
# ═══════════════════════════════════════════════════════

def run_poet_baseline(prices, returns, folds, fb):
    """POET covariance estimator -> QP. Sweep n_factors and threshold_constant on val TE."""
    log.info("\n" + "=" * 60)
    log.info("BASELINE: POET Covariance + QP")
    log.info("=" * 60)

    all_results = []
    min_offset = max(21, 63) + 1
    N_FACTORS_GRID = [3, 5, 10]
    THRESHOLD_GRID = [0.5, 1.0, 2.0]

    for fold_info in folds:
        ret_train = returns.loc[
            (returns.index >= fold_info.train_start) & (returns.index <= fold_info.train_end)
        ]
        ret_val = returns.loc[
            (returns.index >= fold_info.val_start) & (returns.index <= fold_info.val_end)
        ]
        ret_test = returns.loc[
            (returns.index >= fold_info.test_start) & (returns.index <= fold_info.test_end)
        ]

        train_prices = prices.loc[prices.index.isin(ret_train.index)]
        w_index = _compute_index_weights(train_prices)

        rebal_val = [d for d in _get_monthly_rebal(ret_val.index)
                     if d in returns.index and returns.index.get_loc(d) >= min_offset]
        rebal_test = [d for d in _get_monthly_rebal(ret_test.index)
                      if d in returns.index and returns.index.get_loc(d) >= min_offset]

        if not rebal_val or not rebal_test:
            continue

        for K in SPARSITY_LEVELS:
            # Tune n_factors and threshold on validation TE
            best_val_te = float("inf")
            best_nf, best_tc = 5, 1.0

            for nf in N_FACTORS_GRID:
                for tc in THRESHOLD_GRID:
                    val_weights = []
                    for date in rebal_val:
                        try:
                            cov = fb.compute_poet_covariance(
                                returns, date, lookback=LOOKBACK,
                                n_factors=nf, threshold_constant=tc,
                            )
                            w_opt = solve_tracking_qp(cov, w_index, K=K, selection="market_cap")
                        except Exception:
                            w_opt = naive_index_weights(w_index, K)
                        val_weights.append(w_opt)

                    val_metrics = evaluate_portfolio(val_weights, rebal_val, returns, w_index)
                    if val_metrics and val_metrics["tracking_error"] < best_val_te:
                        best_val_te = val_metrics["tracking_error"]
                        best_nf, best_tc = nf, tc

            # Evaluate on test with best hyperparameters
            weights_history = []
            for date in rebal_test:
                try:
                    cov = fb.compute_poet_covariance(
                        returns, date, lookback=LOOKBACK,
                        n_factors=best_nf, threshold_constant=best_tc,
                    )
                    w_opt = solve_tracking_qp(cov, w_index, K=K, selection="market_cap")
                except Exception as e:
                    log.warning(f"POET QP failed at {date}: {e}")
                    w_opt = naive_index_weights(w_index, K)
                weights_history.append(w_opt)

            metrics = evaluate_portfolio(weights_history, rebal_test, returns, w_index)
            if metrics is None:
                continue

            result = {
                "fold": fold_info.fold,
                "K": K,
                "model": "poet",
                "loss": "none",
                "best_n_factors": best_nf,
                "best_threshold_constant": best_tc,
                "test_start": str(fold_info.test_start.date()),
                "test_end": str(fold_info.test_end.date()),
                **metrics,
            }
            all_results.append(result)
            log.info(
                f"Fold {fold_info.fold} K={K} | TE={metrics['tracking_error']:.4f} | "
                f"nf={best_nf} tc={best_tc:.1f}"
            )

    return all_results


# ═══════════════════════════════════════════════════════
# Baseline 7: Graphical Lasso
# ═══════════════════════════════════════════════════════

def run_glasso_baseline(prices, returns, folds, fb):
    """Graphical Lasso covariance -> QP. Uses cross-validation for alpha."""
    log.info("\n" + "=" * 60)
    log.info("BASELINE: Graphical Lasso + QP")
    log.info("=" * 60)

    all_results = []
    min_offset = max(21, 63) + 1

    for fold_info in folds:
        ret_train = returns.loc[
            (returns.index >= fold_info.train_start) & (returns.index <= fold_info.train_end)
        ]
        ret_test = returns.loc[
            (returns.index >= fold_info.test_start) & (returns.index <= fold_info.test_end)
        ]

        train_prices = prices.loc[prices.index.isin(ret_train.index)]
        w_index = _compute_index_weights(train_prices)

        rebal_test = [d for d in _get_monthly_rebal(ret_test.index)
                      if d in returns.index and returns.index.get_loc(d) >= min_offset]

        if not rebal_test:
            continue

        for K in SPARSITY_LEVELS:
            weights_history = []

            for date in rebal_test:
                try:
                    cov = fb.compute_glasso_covariance(returns, date, lookback=LOOKBACK)
                    w_opt = solve_tracking_qp(cov, w_index, K=K, selection="market_cap")
                except Exception as e:
                    log.warning(f"GLasso QP failed at {date}: {e}")
                    w_opt = naive_index_weights(w_index, K)
                weights_history.append(w_opt)

            metrics = evaluate_portfolio(weights_history, rebal_test, returns, w_index)
            if metrics is None:
                continue

            result = {
                "fold": fold_info.fold,
                "K": K,
                "model": "glasso",
                "loss": "none",
                "test_start": str(fold_info.test_start.date()),
                "test_end": str(fold_info.test_end.date()),
                **metrics,
            }
            all_results.append(result)
            log.info(f"Fold {fold_info.fold} K={K} | TE={metrics['tracking_error']:.4f}")

    return all_results


# ═══════════════════════════════════════════════════════
# Baseline 8: K-only estimation (decision-blind on subset)
# ═══════════════════════════════════════════════════════

def run_konly_baseline(prices, returns, folds, fb):
    """Pick top-K by market cap, estimate covariance ONLY among those K stocks,
    then solve QP. No full-N covariance estimation.

    This is the natural 'decision-blind subset' baseline suggested as a
    comparison: if you know which K stocks you want, do you even need to
    estimate the full N×N matrix?
    """
    from sklearn.covariance import LedoitWolf

    log.info("\n" + "=" * 60)
    log.info("BASELINE: K-only Estimation (LW on K stocks)")
    log.info("=" * 60)

    all_results = []
    min_offset = max(21, 63) + 1

    for fold_info in folds:
        ret_test = returns.loc[
            (returns.index >= fold_info.test_start) & (returns.index <= fold_info.test_end)
        ]
        train_prices = prices.loc[
            (prices.index >= fold_info.train_start) & (prices.index <= fold_info.train_end)
        ]
        w_index = _compute_index_weights(train_prices)

        rebal_test = [d for d in _get_monthly_rebal(ret_test.index)
                      if d in returns.index and returns.index.get_loc(d) >= min_offset]
        if not rebal_test:
            continue

        for K in SPARSITY_LEVELS:
            # Select top-K by market cap (same selection as other baselines)
            selected = np.argsort(w_index)[-K:]
            w_idx_K = w_index[selected]
            w_idx_K = w_idx_K / (w_idx_K.sum() + 1e-10)

            weights_history = []
            for date in rebal_test:
                loc = returns.index.get_loc(date)
                start = max(0, loc - LOOKBACK)
                # Extract returns ONLY for the K selected stocks
                ret_window = returns.iloc[start:loc, selected].values

                # Estimate K×K covariance via Ledoit-Wolf
                if ret_window.shape[0] > 2:
                    lw = LedoitWolf().fit(ret_window)
                    cov_K = lw.covariance_.astype(np.float64)
                else:
                    cov_K = np.eye(K, dtype=np.float64)

                # Solve QP directly on K stocks (no selection step needed)
                cov_K = 0.5 * (cov_K + cov_K.T)
                cov_K += 1e-6 * np.eye(K)
                b_K = cov_K @ w_idx_K

                try:
                    import cvxpy as cp
                    w_var = cp.Variable(K)
                    objective = cp.Minimize(cp.quad_form(w_var, cp.psd_wrap(cov_K)) - 2 * b_K @ w_var)
                    constraints = [cp.sum(w_var) == 1, w_var >= 0]
                    prob = cp.Problem(objective, constraints)
                    prob.solve(solver=cp.SCS, verbose=False)
                    if w_var.value is not None:
                        w_K = np.maximum(w_var.value, 0)
                        w_K = w_K / (w_K.sum() + 1e-10)
                    else:
                        w_K = w_idx_K.copy()
                except Exception:
                    w_K = w_idx_K.copy()

                # Embed back into N-dimensional weight vector
                w_full = np.zeros(len(w_index), dtype=np.float64)
                w_full[selected] = w_K
                weights_history.append(w_full)

            metrics = evaluate_portfolio(weights_history, rebal_test, returns, w_index)
            if metrics is None:
                continue

            result = {
                "fold": fold_info.fold,
                "K": K,
                "model": "konly_lw",
                "loss": "none",
                "test_start": str(fold_info.test_start.date()),
                "test_end": str(fold_info.test_end.date()),
                **metrics,
            }
            all_results.append(result)
            log.info(f"Fold {fold_info.fold} K={K} | TE={metrics['tracking_error']:.4f}")

    return all_results


def main():
    prices, returns, tickers, folds, fb = load_data()
    log.info(f"Loaded {len(tickers)} stocks, {len(folds)} folds")

    out_dir = Path("results/baselines")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Naive baseline
    naive_results = run_naive_baseline(prices, returns, folds, fb)
    with open(out_dir / "naive_results.json", "w") as f:
        json.dump({"results": naive_results}, f, indent=2)
    log.info(f"\nNaive results saved ({len(naive_results)} entries)")

    # 2. Ledoit-Wolf baseline
    lw_results = run_ledoit_wolf_baseline(prices, returns, folds, fb)
    with open(out_dir / "ledoit_wolf_results.json", "w") as f:
        json.dump({"results": lw_results}, f, indent=2)
    log.info(f"Ledoit-Wolf results saved ({len(lw_results)} entries)")

    # 3. Shrinkage alpha DFL
    shrinkage_results = run_shrinkage_dfl(prices, returns, folds, fb)
    with open(out_dir / "shrinkage_dfl_results.json", "w") as f:
        json.dump({"results": shrinkage_results}, f, indent=2)
    log.info(f"Shrinkage DFL results saved ({len(shrinkage_results)} entries)")

    # 4. Validation-tuned shrinkage (grid search α)
    val_tuned_results = run_val_tuned_baseline(prices, returns, folds, fb)
    with open(out_dir / "val_tuned_shrinkage_results.json", "w") as f:
        json.dump({"results": val_tuned_results}, f, indent=2)
    log.info(f"Val-tuned shrinkage results saved ({len(val_tuned_results)} entries)")

    # 5. SPO+ baseline
    spo_results = run_spo_plus_baseline(prices, returns, folds, fb)
    with open(out_dir / "spo_plus_results.json", "w") as f:
        json.dump({"results": spo_results}, f, indent=2)
    log.info(f"SPO+ results saved ({len(spo_results)} entries)")

    # 6. POET baseline
    poet_results = run_poet_baseline(prices, returns, folds, fb)
    with open(out_dir / "poet_results.json", "w") as f:
        json.dump({"results": poet_results}, f, indent=2)
    log.info(f"POET results saved ({len(poet_results)} entries)")

    # 7. Graphical Lasso baseline
    glasso_results = run_glasso_baseline(prices, returns, folds, fb)
    with open(out_dir / "glasso_results.json", "w") as f:
        json.dump({"results": glasso_results}, f, indent=2)
    log.info(f"Graphical Lasso results saved ({len(glasso_results)} entries)")

    # 8. K-only estimation baseline
    konly_results = run_konly_baseline(prices, returns, folds, fb)
    with open(out_dir / "konly_results.json", "w") as f:
        json.dump({"results": konly_results}, f, indent=2)
    log.info(f"K-only results saved ({len(konly_results)} entries)")

    # ── Summary ──
    log.info("\n" + "=" * 60)
    log.info("SUMMARY: Mean Tracking Error by K")
    log.info("=" * 60)

    all_baselines = [
        ("Naive", naive_results),
        ("Ledoit-Wolf", lw_results),
        ("Shrinkage-DFL", shrinkage_results),
        ("Val-Tuned", val_tuned_results),
        ("SPO+", spo_results),
        ("POET", poet_results),
        ("GraphicalLasso", glasso_results),
        ("K-only LW", konly_results),
    ]
    for name, results in all_baselines:
        log.info(f"\n{name}:")
        for K in SPARSITY_LEVELS:
            tes = [r["tracking_error"] for r in results if r["K"] == K]
            if tes:
                log.info(f"  K={K:3d}: TE={np.mean(tes):.4f} +/- {np.std(tes):.4f}")


if __name__ == "__main__":
    main()
