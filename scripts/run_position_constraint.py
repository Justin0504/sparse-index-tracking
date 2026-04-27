#!/usr/bin/env python3
"""Position size constraint experiment.

Compares DFL vs MSE with max position constraint w_i <= w_max.
Tests whether DFL advantage holds under realistic portfolio constraints.
"""

import json
import sys
from pathlib import Path

import cvxpy as cp
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.features import FeatureBuilder
from src.data.loader import SP500DataLoader
from src.data.splitter import RollingWindowSplitter
from src.models import NeuralCovarianceModel
from src.optimization.solver import compute_tracking_scores_np
from src.training import Trainer
from src.evaluation.metrics import tracking_error, turnover
from src.utils import set_seed, get_logger

log = get_logger(__name__)


def solve_tracking_qp_constrained(cov, w_idx, K, w_max=0.10, selection="market_cap"):
    """Solve sparse tracking QP with max position constraint."""
    n = cov.shape[0]
    K = min(K, n)

    if selection == "tracking_score":
        scores = compute_tracking_scores_np(cov, w_idx)
    else:
        scores = w_idx.copy()
    selected = np.argsort(scores)[-K:]
    selected_sorted = np.sort(selected)

    cov_SS = cov[np.ix_(selected_sorted, selected_sorted)]
    b = cov[selected_sorted] @ w_idx

    w_S = cp.Variable(K)
    cov_sym = 0.5 * (cov_SS + cov_SS.T)
    objective = cp.Minimize(cp.quad_form(w_S, cov_sym, assume_PSD=True) - 2 * b @ w_S)
    constraints = [cp.sum(w_S) == 1, w_S >= 0, w_S <= w_max]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver="SCS", verbose=False)

    if prob.status in ("optimal", "optimal_inaccurate"):
        w_S_val = np.clip(w_S.value, 0, w_max).astype(np.float64)
    else:
        w_S_val = np.ones(K, dtype=np.float64) / K

    w_full = np.zeros(n, dtype=np.float64)
    w_full[selected_sorted] = w_S_val
    w_full = w_full / (w_full.sum() + 1e-10)
    return w_full


def run_constrained_experiment(K=20, w_max=0.10, loss_type="task", device="cpu"):
    set_seed(42)

    fb = FeatureBuilder(vol_window=21, momentum_window=63,
                        use_sector=True, use_mcap_quintile=True)
    loader = SP500DataLoader(start_date="2006-01-01", end_date="2025-12-31", top_n=100)
    prices, index_prices, tickers = loader.load()
    n_stocks = len(tickers)

    returns = fb.compute_returns(prices)
    splitter = RollingWindowSplitter(train_years=3, val_months=6,
                                     test_years=1, step_years=1)
    folds = splitter.split(returns.index)

    min_offset = 64
    sample_date = returns.index[min_offset + 1]
    n_features = fb.build_features(returns, sample_date).shape[1]

    all_results = []

    for fold_info in folds:
        train_mask = (returns.index >= fold_info.train_start) & (returns.index <= fold_info.train_end)
        val_mask = (returns.index >= fold_info.val_start) & (returns.index <= fold_info.val_end)
        test_mask = (returns.index >= fold_info.test_start) & (returns.index <= fold_info.test_end)

        ret_train = returns.loc[train_mask]
        ret_val = returns.loc[val_mask]
        ret_test = returns.loc[test_mask]

        train_prices = prices.loc[prices.index.isin(ret_train.index)]
        last_p = train_prices.iloc[-1].values
        w_index = (last_p / last_p.sum()).astype(np.float32)

        def _get_rebal(ret_df):
            dates = [d for d in ret_df.index if d.weekday() == 4]
            return [d for d in dates if returns.index.get_loc(d) >= min_offset]

        rebal_train = _get_rebal(ret_train)
        rebal_val = _get_rebal(ret_val)

        s = pd.Series(range(len(ret_test.index)), index=ret_test.index)
        monthly_idx = s.resample("ME").last().dropna()
        rebal_test = []
        for d in monthly_idx.index:
            closest = ret_test.index[ret_test.index <= d]
            if len(closest) > 0 and returns.index.get_loc(closest[-1]) >= min_offset:
                rebal_test.append(closest[-1])

        if not rebal_train or not rebal_test:
            continue

        model = NeuralCovarianceModel(n_stocks=n_stocks, n_features=n_features,
                                       hidden_dims=[64, 64])

        trainer = Trainer(
            model=model, feature_builder=fb, loss_type=loss_type,
            lr=1e-3, weight_decay=1e-4, epochs=50, patience=10,
            device=device, sparsity_K=K, selection="market_cap",
        )

        trainer.train(
            returns_train=ret_train.values,
            returns_val=ret_val.values,
            rebalance_dates_train=rebal_train,
            rebalance_dates_val=rebal_val,
            returns_df_train=returns,
            returns_df_val=returns,
            w_index=w_index,
            lookback=63,
        )

        model.eval()
        portfolio_rets, index_rets = [], []
        weights_history = []
        max_positions = []

        for i, date in enumerate(rebal_test):
            loc = returns.index.get_loc(date)
            next_loc = (returns.index.get_loc(rebal_test[i + 1])
                        if i + 1 < len(rebal_test) else len(returns))
            if next_loc <= loc:
                continue

            features_np = fb.build_features(returns, date)
            with torch.no_grad():
                features_t = torch.tensor(features_np, dtype=torch.float32, device=device)
                cov_pred = model(features_t).cpu().numpy()

            try:
                w_opt = solve_tracking_qp_constrained(cov_pred, w_index, K=K,
                                                       w_max=w_max, selection="market_cap")
            except Exception:
                w_opt = np.ones(n_stocks, dtype=np.float64) / n_stocks

            weights_history.append(w_opt)
            max_positions.append(float(np.max(w_opt)))
            holding_rets = returns.iloc[loc:next_loc].values
            portfolio_rets.extend((holding_rets @ w_opt).tolist())
            index_rets.extend((holding_rets @ w_index).tolist())

        if not portfolio_rets:
            continue

        pr = np.array(portfolio_rets)
        ir = np.array(index_rets)
        te = tracking_error(pr, ir, 252)
        avg_to = float(np.mean([turnover(weights_history[j], weights_history[j + 1])
                                 for j in range(len(weights_history) - 1)])) if len(weights_history) > 1 else 0.0

        result = {
            "fold": fold_info.fold, "K": K, "w_max": w_max,
            "loss": loss_type, "tracking_error": te, "avg_turnover": avg_to,
            "max_position": float(np.mean(max_positions)),
        }
        all_results.append(result)
        log.info(f"  Fold {fold_info.fold} | {loss_type} | w_max={w_max} | TE={te:.4f} | max_pos={np.mean(max_positions):.3f}")

    return all_results


def main():
    all_results = []
    for w_max in [0.10, 0.20, 1.0]:  # 1.0 = unconstrained
        for loss in ["mse", "task"]:
            log.info(f"\n{'='*50}")
            log.info(f"w_max={w_max}, loss={loss}")
            log.info(f"{'='*50}")
            results = run_constrained_experiment(K=20, w_max=w_max, loss_type=loss)
            all_results.extend(results)

    out_path = Path("results/position_constraint.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"results": all_results}, f, indent=2)
    log.info(f"\nResults saved to {out_path}")

    print("\n=== Position Constraint Experiment (K=20, Neural) ===")
    print(f"{'w_max':>6} {'MSE TE':>8} {'DFL TE':>8} {'Gain':>8} {'MSE maxW':>9} {'DFL maxW':>9}")
    for wm in [0.10, 0.20, 1.0]:
        mse = [r for r in all_results if abs(r["w_max"] - wm) < 0.01 and r["loss"] == "mse"]
        dfl = [r for r in all_results if abs(r["w_max"] - wm) < 0.01 and r["loss"] == "task"]
        if mse and dfl:
            m_te = np.mean([r["tracking_error"] for r in mse])
            d_te = np.mean([r["tracking_error"] for r in dfl])
            m_mp = np.mean([r["max_position"] for r in mse])
            d_mp = np.mean([r["max_position"] for r in dfl])
            gain = (1 - d_te / m_te) * 100
            label = f"{wm:.0%}" if wm < 1.0 else "none"
            print(f"{label:>6} {m_te:8.4f} {d_te:8.4f} {gain:+7.1f}% {m_mp:9.3f} {d_mp:9.3f}")


if __name__ == "__main__":
    main()
