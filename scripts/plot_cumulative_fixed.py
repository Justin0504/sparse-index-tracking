#!/usr/bin/env python3
"""Fixed cumulative wealth figure with correct date alignment."""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 12


def load_fold_returns(path, K=20):
    with open(path) as f:
        data = json.load(f)
    folds = sorted([r for r in data['results'] if r['K'] == K], key=lambda x: x['fold'])
    return folds


def extract_non_overlapping(folds):
    """Extract each fold's unique (non-overlapping) test period returns."""
    port_all, idx_all, dates_all = [], [], []

    for i, fold in enumerate(folds):
        # Each fold's unique days = its length minus next fold's length
        if i + 1 < len(folds):
            n_unique = len(fold['portfolio_returns']) - len(folds[i+1]['portfolio_returns'])
        else:
            n_unique = len(fold['portfolio_returns'])

        port_all.extend(fold['portfolio_returns'][:n_unique])
        idx_all.extend(fold['index_returns'][:n_unique])

        # Generate approximate trading dates
        start = pd.Timestamp(fold['test_start'])
        trading_dates = pd.bdate_range(start, periods=n_unique)
        dates_all.extend(trading_dates.tolist())

    return np.array(port_all), np.array(idx_all), dates_all


task_folds = load_fold_returns('results/neural_task_results.json')
mse_folds = load_fold_returns('results/neural_mse_results.json')
naive_folds = load_fold_returns('results/baselines/naive_results.json')

task_port, task_idx, dates = extract_non_overlapping(task_folds)
mse_port, mse_idx, _ = extract_non_overlapping(mse_folds)
naive_port, naive_idx, _ = extract_non_overlapping(naive_folds)

# Convert dates to years for x-axis
years = np.array([(d - dates[0]).days / 365.25 for d in dates])
year_labels = np.array([d.year + d.month/12 for d in dates])

# Cumulative wealth
cum = lambda r: np.cumprod(1 + r)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), height_ratios=[3, 1],
                                 gridspec_kw={'hspace': 0.15})

ax1.plot(year_labels, cum(task_idx), 'k-', linewidth=2.5, label='S&P 500 Index', zorder=5)
ax1.plot(year_labels, cum(task_port), '-', color='#1E88E5', linewidth=1.8,
         label='Task-Aware DFL (K=20)', alpha=0.9, zorder=4)
ax1.plot(year_labels, cum(mse_port), '-', color='#E53935', linewidth=1.8,
         label='Two-Stage MSE (K=20)', alpha=0.7, zorder=3)
ax1.plot(year_labels, cum(naive_port), '-', color='#9E9E9E', linewidth=1.2,
         label='Naive Equal-Weight (K=20)', alpha=0.6, zorder=2)

# Shade key periods
ax1.axvspan(2020.15, 2020.45, alpha=0.08, color='red')
ax1.axvspan(2022.0, 2022.75, alpha=0.08, color='orange')
ylim = ax1.get_ylim()
ax1.text(2020.15, ylim[1]*0.92, 'COVID', fontsize=9, color='#C62828', alpha=0.8)
ax1.text(2022.0, ylim[1]*0.92, 'Rate Hikes', fontsize=9, color='#E65100', alpha=0.8)

ax1.set_ylabel('Cumulative Wealth ($1 invested)', fontsize=12)
ax1.set_title('Sparse Index Tracking: 20 Stocks Replicate the S&P 500 (2016-2025)',
              fontsize=14, fontweight='bold')
ax1.legend(fontsize=10, loc='upper left', framealpha=0.9)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(year_labels[0], year_labels[-1])

# Bottom: cumulative tracking difference
track_task = np.cumsum(task_port - task_idx)
track_mse = np.cumsum(mse_port - mse_idx)
track_naive = np.cumsum(naive_port - naive_idx)

ax2.plot(year_labels, track_task, '-', color='#1E88E5', linewidth=1.8, label='Task-Aware')
ax2.plot(year_labels, track_mse, '-', color='#E53935', linewidth=1.8, label='Two-Stage MSE')
ax2.plot(year_labels, track_naive, '-', color='#9E9E9E', linewidth=1.2, alpha=0.6, label='Naive')
ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
ax2.fill_between(year_labels, track_task, 0, alpha=0.08, color='#1E88E5')
ax2.fill_between(year_labels, track_mse, 0, alpha=0.08, color='#E53935')

ax2.set_xlabel('Year', fontsize=12)
ax2.set_ylabel('Cumulative\nTracking Diff', fontsize=10)
ax2.legend(fontsize=9, loc='lower left', ncol=3, framealpha=0.9)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(year_labels[0], year_labels[-1])

# Add fold boundaries
for fold in task_folds[1:]:
    t = pd.Timestamp(fold['test_start'])
    yr = t.year + t.month/12
    ax1.axvline(x=yr, color='gray', linestyle=':', alpha=0.2, linewidth=0.8)
    ax2.axvline(x=yr, color='gray', linestyle=':', alpha=0.2, linewidth=0.8)

fig.tight_layout()
fig.savefig("figures/fig12_cumulative_wealth.png", dpi=200, bbox_inches='tight')
fig.savefig("figures/fig12_cumulative_wealth.pdf", bbox_inches='tight')
print("Saved fig12_cumulative_wealth.png/pdf")
plt.close()
