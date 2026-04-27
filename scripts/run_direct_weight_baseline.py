#!/usr/bin/env python3
"""Direct weight prediction baseline.

Instead of predict-covariance → solve-QP, directly predict portfolio weights from features.
This tests whether the QP structure adds value.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.features import FeatureBuilder
from src.data.loader import SP500DataLoader
from src.data.splitter import RollingWindowSplitter
from src.evaluation.metrics import tracking_error, turnover
from src.utils import set_seed, get_logger

log = get_logger(__name__)


class DirectWeightModel(nn.Module):
    """MLP that directly predicts sparse portfolio weights."""

    def __init__(self, n_features, n_stocks, K, hidden_dims=None):
        super().__init__()
        self.n_stocks = n_stocks
        self.K = K
        hidden_dims = hidden_dims or [64, 64]

        # Per-stock feature encoder (shared weights)
        layers = []
        in_dim = n_features
        for h in hidden_dims:
            layers.extend([nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(0.1)])
            in_dim = h
        self.encoder = nn.Sequential(*layers)

        # Score head: per-stock score for selection + weight
        self.score_head = nn.Linear(in_dim, 1)

    def forward(self, features):
        """
        features: (N, n_features)
        Returns: (N,) portfolio weights summing to 1, at most K nonzero.
        """
        h = self.encoder(features)  # (N, hidden)
        scores = self.score_head(h).squeeze(-1)  # (N,)

        # Top-K selection (straight-through estimator for gradients)
        _, top_idx = scores.topk(self.K)
        mask = torch.zeros_like(scores)
        mask[top_idx] = 1.0

        # Soft weights via softmax on selected scores
        selected_scores = scores * mask + (1 - mask) * (-1e9)
        weights = torch.softmax(selected_scores, dim=0)

        return weights


def run_direct_weight(K=20, device="cpu"):
    """Run direct weight prediction baseline."""
    set_seed(42)
    n_stocks = 100

    fb = FeatureBuilder(vol_window=21, momentum_window=63,
                        use_sector=True, use_mcap_quintile=True)
    loader = SP500DataLoader(start_date="2006-01-01", end_date="2025-12-31", top_n=100)
    prices, index_prices, tickers = loader.load()

    returns = fb.compute_returns(prices)
    splitter = RollingWindowSplitter(train_years=2, val_months=6,
                                     test_years=1, step_years=1)
    folds = splitter.split(returns.index)
    log.info(f"Direct weight prediction: K={K}, {len(folds)} folds")

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
        w_idx_t = torch.tensor(w_index, dtype=torch.float32, device=device)

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

        model = DirectWeightModel(n_features, n_stocks, K).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

        # Train: minimize tracking error directly
        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(50):
            model.train()
            epoch_losses = []

            for date in rebal_train:
                loc = returns.index.get_loc(date)
                features_np = fb.build_features(returns, date)
                features_t = torch.tensor(features_np, dtype=torch.float32, device=device)

                # Forward returns (21 days)
                end_loc = min(loc + 21, len(returns))
                fwd_rets = torch.tensor(
                    returns.iloc[loc:end_loc].values,
                    dtype=torch.float32, device=device
                )

                w = model(features_t)
                port_rets = fwd_rets @ w
                idx_rets = fwd_rets @ w_idx_t
                te_sq = ((port_rets - idx_rets) ** 2).mean()

                optimizer.zero_grad()
                te_sq.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_losses.append(te_sq.item())

            # Validation
            model.eval()
            val_losses = []
            with torch.no_grad():
                for date in rebal_val:
                    loc = returns.index.get_loc(date)
                    features_np = fb.build_features(returns, date)
                    features_t = torch.tensor(features_np, dtype=torch.float32, device=device)
                    end_loc = min(loc + 21, len(returns))
                    fwd_rets = torch.tensor(
                        returns.iloc[loc:end_loc].values,
                        dtype=torch.float32, device=device
                    )
                    w = model(features_t)
                    port_rets = fwd_rets @ w
                    idx_rets = fwd_rets @ w_idx_t
                    val_losses.append(((port_rets - idx_rets) ** 2).mean().item())

            val_loss = np.mean(val_losses) if val_losses else float("inf")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 10:
                    break

        if best_state:
            model.load_state_dict(best_state)

        # Test
        model.eval()
        portfolio_rets, index_rets_list = [], []
        weights_history = []

        for i, date in enumerate(rebal_test):
            loc = returns.index.get_loc(date)
            next_loc = (returns.index.get_loc(rebal_test[i + 1])
                        if i + 1 < len(rebal_test) else len(returns))
            if next_loc <= loc:
                continue

            features_np = fb.build_features(returns, date)
            with torch.no_grad():
                features_t = torch.tensor(features_np, dtype=torch.float32, device=device)
                w = model(features_t).cpu().numpy()

            weights_history.append(w)
            holding_rets = returns.iloc[loc:next_loc].values
            portfolio_rets.extend((holding_rets @ w).tolist())
            index_rets_list.extend((holding_rets @ w_index).tolist())

        if not portfolio_rets:
            continue

        pr = np.array(portfolio_rets)
        ir = np.array(index_rets_list)
        te = tracking_error(pr, ir, 252)
        avg_to = float(np.mean([turnover(weights_history[j], weights_history[j + 1])
                                 for j in range(len(weights_history) - 1)])) if len(weights_history) > 1 else 0.0

        result = {
            "fold": fold_info.fold, "K": K,
            "method": "direct_weight", "tracking_error": te, "avg_turnover": avg_to,
        }
        all_results.append(result)
        log.info(f"  Fold {fold_info.fold} | Direct Weight K={K} | TE={te:.4f} | TO={avg_to:.4f}")

    return all_results


def main():
    all_results = []
    for K in [10, 20]:
        results = run_direct_weight(K=K)
        all_results.extend(results)

    out_path = Path("results/direct_weight_baseline.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"results": all_results}, f, indent=2)
    log.info(f"\nResults saved to {out_path}")

    print("\n=== Direct Weight Prediction Baseline ===")
    for K in [10, 20]:
        tes = [r["tracking_error"] for r in all_results if r["K"] == K]
        tos = [r["avg_turnover"] for r in all_results if r["K"] == K]
        if tes:
            print(f"K={K}: TE={np.mean(tes):.4f}, TO={np.mean(tos):.4f}")


if __name__ == "__main__":
    main()
