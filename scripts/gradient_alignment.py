#!/usr/bin/env python3
"""Empirically measure cosine similarity between MSE and task gradients.

For each K in {5, 10, 20, 30, 50}, computes both gradients at the same
model state and rebalancing date, then reports cos(g_MSE, g_task).
Validates the theoretical bound: cos <= sqrt(2K/N).
"""

import sys
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import SP500DataLoader, FeatureBuilder, RollingWindowSplitter
from src.models import FactorCovarianceModel, NeuralCovarianceModel
from src.optimization.qp_layer import SparseTrackingQP
from src.training.losses import mse_loss, task_loss
from src.utils import set_seed, get_logger

log = get_logger(__name__)


def compute_gradient(model, loss_fn, retain_graph=False):
    """Compute gradient of loss_fn w.r.t. model parameters, return as flat vector."""
    model.zero_grad()
    loss = loss_fn()
    loss.backward(retain_graph=retain_graph)
    grads = []
    for p in model.parameters():
        if p.grad is not None:
            grads.append(p.grad.detach().flatten())
        else:
            grads.append(torch.zeros(p.numel()))
    return torch.cat(grads), float(loss.detach())


def cosine_sim(a, b):
    """Cosine similarity between two vectors."""
    na = torch.norm(a)
    nb = torch.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(torch.dot(a, b) / (na * nb))


def main():
    set_seed(42)
    device = "cpu"  # Needs double backward, CPU is safer

    # Load data
    log.info("Loading S&P 500 data...")
    loader = SP500DataLoader(
        start_date="2006-01-01", end_date="2025-12-31",
        top_n=100, cache_dir="data/cache",
    )
    prices, index_prices, tickers = loader.load()
    n_stocks = len(tickers)

    fb = FeatureBuilder(vol_window=21, momentum_window=63, use_sector=True, use_mcap_quintile=True)
    returns = fb.compute_returns(prices)

    # Use a single fold for gradient analysis
    splitter = RollingWindowSplitter(train_years=3, val_months=6, test_years=1, step_years=1)
    folds = splitter.split(returns.index)
    fold = folds[4]  # Mid-range fold (2013-2016 era)
    log.info(f"Using fold: {fold}")

    train_mask = (returns.index >= fold.train_start) & (returns.index <= fold.train_end)
    ret_train = returns.loc[train_mask]

    # Index weights
    train_prices = prices.loc[prices.index.isin(ret_train.index)]
    last_p = train_prices.iloc[-1].values
    w_index = (last_p / last_p.sum()).astype(np.float32)
    w_idx_t = torch.tensor(w_index, dtype=torch.float32, device=device)

    # Get rebalancing dates (weekly Fridays within training period)
    min_offset = 64
    rebal_dates = [
        d for d in ret_train.index
        if d.weekday() == 4 and returns.index.get_loc(d) >= min_offset
    ]

    K_values = [5, 10, 20, 30, 50]
    lookback = 63
    forward_window = 21

    # Sample features to get n_features
    sample_date = rebal_dates[0]
    n_features = fb.build_features(returns, sample_date).shape[1]

    # Test with factor model (moderate params) and neural model (many params)
    results = {"factor": {}, "neural": {}}

    for model_name in ["factor", "neural"]:
        log.info(f"\n=== Model: {model_name} ===")

        for K in K_values:
            cos_sims = []
            theoretical_bound = np.sqrt(2 * K / n_stocks)

            # Sample multiple dates for robustness
            sample_dates = rebal_dates[2:22]  # 20 dates

            for date in sample_dates:
                loc = returns.index.get_loc(date)
                if loc + forward_window > len(returns):
                    continue

                # Fresh model each time (untrained - measures gradient geometry)
                if model_name == "factor":
                    model = FactorCovarianceModel(n_features=n_features, n_stocks=n_stocks, n_factors=10)
                else:
                    model = NeuralCovarianceModel(n_features=n_features, n_stocks=n_stocks,
                                                  hidden_dims=[256, 128], dropout=0.0, rank=20)

                model.to(device)
                model.train()

                features_np = fb.build_features(returns, date)
                features_t = torch.tensor(features_np, dtype=torch.float32, device=device)

                # MSE gradient
                cov_target_np = fb.compute_realized_covariance(returns, date, lookback)
                cov_target = torch.tensor(cov_target_np, dtype=torch.float32, device=device)

                def mse_fn():
                    return mse_loss(model(features_t), cov_target)

                g_mse, _ = compute_gradient(model, mse_fn)

                try:
                    qp = SparseTrackingQP(n_stocks)

                    def task_fn():
                        cp = model(features_t)
                        w_opt = qp.solve(cp, w_idx_t, K=K, hard=False, selection="market_cap")
                        future_ret_np = returns.iloc[loc:loc + forward_window].values
                        future_ret_t = torch.tensor(future_ret_np, dtype=torch.float32, device=device)
                        return task_loss(w_opt, w_idx_t, future_ret_t)

                    g_task, _ = compute_gradient(model, task_fn)
                except Exception as e:
                    log.debug(f"QP failed at {date}, K={K}: {e}")
                    continue

                cs = cosine_sim(g_mse, g_task)
                cos_sims.append(cs)

            if cos_sims:
                mean_cs = np.mean(cos_sims)
                std_cs = np.std(cos_sims)
                results[model_name][K] = {
                    "mean_cos": mean_cs,
                    "std_cos": std_cs,
                    "theoretical_bound": theoretical_bound,
                    "n_samples": len(cos_sims),
                    "all_cos": cos_sims,
                }
                log.info(
                    f"K={K:3d} | cos(g_MSE, g_task) = {mean_cs:.4f} +/- {std_cs:.4f} | "
                    f"bound = {theoretical_bound:.4f} | ratio = {mean_cs/theoretical_bound:.2f}"
                )

    # Save results
    out_path = Path("results/gradient_alignment.json")
    out_path.parent.mkdir(exist_ok=True)

    # Convert for JSON
    save_results = {}
    for m in results:
        save_results[m] = {}
        for k, v in results[m].items():
            save_results[m][str(k)] = {kk: vv for kk, vv in v.items()}

    with open(out_path, "w") as f:
        json.dump(save_results, f, indent=2)
    log.info(f"\nResults saved to {out_path}")

    # Generate plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    K_range = np.linspace(1, 100, 200)
    bound_curve = np.sqrt(2 * K_range / n_stocks)
    ax.plot(K_range, bound_curve, 'k--', linewidth=2, label=r'Bound: $\sqrt{2K/N}$', zorder=5)

    colors = {'factor': '#4682b4', 'neural': '#d4a574'}
    markers = {'factor': 'o', 'neural': 's'}
    labels = {'factor': 'Factor model ($K_f{=}10$)', 'neural': 'Neural model ($r{=}20$)'}

    for model_name in results:
        ks = sorted(results[model_name].keys())
        means = [results[model_name][k]["mean_cos"] for k in ks]
        stds = [results[model_name][k]["std_cos"] for k in ks]

        ax.errorbar(ks, means, yerr=stds, fmt=markers[model_name] + '-',
                     color=colors[model_name], linewidth=1.5, markersize=8,
                     capsize=4, label=labels[model_name], zorder=10)

    ax.set_xlabel('Portfolio cardinality $K$', fontsize=12)
    ax.set_ylabel(r'$\cos(\mathbf{g}_{\mathrm{MSE}},\, \mathbf{g}_{\mathrm{task}})$', fontsize=12)
    ax.set_title(f'Gradient Alignment vs. Sparsity ($N = {n_stocks}$)', fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xlim(0, 55)
    ax.set_ylim(0, 1.1)
    ax.grid(alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add shaded region above bound
    ax.fill_between(K_range, bound_curve, 1.1, alpha=0.08, color='red', label='_nolegend_')
    ax.text(45, 0.95, 'Bound\nviolation', fontsize=8, color='red', alpha=0.5, ha='center')

    plt.tight_layout()
    plt.savefig("paper/figures/gradient_alignment.pdf", bbox_inches="tight", dpi=150)
    plt.savefig("paper/figures/gradient_alignment.png", bbox_inches="tight", dpi=150)
    log.info("Figure saved to paper/figures/gradient_alignment.{pdf,png}")


if __name__ == "__main__":
    main()
