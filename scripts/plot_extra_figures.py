#!/usr/bin/env python3
"""Generate three additional presentation figures."""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 12


# ══════════════════════════════════════════════════════════════════
# Figure A: Cumulative Wealth — Sparse Portfolio vs Index
# ══════════════════════════════════════════════════════════════════

with open("results/neural_task_results.json") as f:
    task_data = json.load(f)
with open("results/neural_mse_results.json") as f:
    mse_data = json.load(f)
with open("results/baselines/naive_results.json") as f:
    naive_data = json.load(f)

# Concatenate all K=20 folds (they cover 2016-2025)
def get_k20_returns(data, K=20):
    folds = sorted([r for r in data['results'] if r['K'] == K], key=lambda x: x['fold'])
    port_all, idx_all = [], []
    for f in folds:
        port_all.extend(f['portfolio_returns'])
        idx_all.extend(f['index_returns'])
    return np.array(port_all), np.array(idx_all)

task_port, task_idx = get_k20_returns(task_data)
mse_port, mse_idx = get_k20_returns(mse_data)
naive_port, naive_idx = get_k20_returns(naive_data)

# Cumulative wealth
def cum_wealth(r):
    return np.cumprod(1 + r)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), height_ratios=[3, 1])

n = len(task_port)
days = np.arange(n)
# Approximate dates: 9 folds × ~252 days = 2016-2025
years = 2016 + days / 252

ax1.plot(years, cum_wealth(task_idx), 'k-', linewidth=2, label='S&P 500 Index', zorder=5)
ax1.plot(years, cum_wealth(task_port), '-', color='#1E88E5', linewidth=1.5,
         label='Task-Aware (K=20)', alpha=0.9, zorder=4)
ax1.plot(years, cum_wealth(mse_port), '-', color='#E53935', linewidth=1.5,
         label='Two-Stage MSE (K=20)', alpha=0.7, zorder=3)
ax1.plot(years, cum_wealth(naive_port), '-', color='#9E9E9E', linewidth=1,
         label='Naive Equal-Weight (K=20)', alpha=0.6, zorder=2)

ax1.set_ylabel('Cumulative Wealth ($1 invested)', fontsize=12)
ax1.set_title('Sparse Index Tracking: 20 Stocks vs. S&P 500 (2016–2025)', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10, loc='upper left', framealpha=0.9)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(years[0], years[-1])

# Shade key periods
ax1.axvspan(2020.0, 2020.5, alpha=0.1, color='red', label='_COVID')
ax1.axvspan(2022.0, 2022.75, alpha=0.1, color='orange', label='_Rate hikes')
ax1.text(2020.1, ax1.get_ylim()[1]*0.95, 'COVID', fontsize=8, color='red', alpha=0.7)
ax1.text(2022.05, ax1.get_ylim()[1]*0.95, 'Rate\nHikes', fontsize=8, color='orange', alpha=0.7)

# Bottom panel: tracking difference
track_diff_task = np.cumsum(task_port - task_idx)
track_diff_mse = np.cumsum(mse_port - mse_idx)
track_diff_naive = np.cumsum(naive_port - naive_idx)

ax2.plot(years, track_diff_task, '-', color='#1E88E5', linewidth=1.5, label='Task-Aware')
ax2.plot(years, track_diff_mse, '-', color='#E53935', linewidth=1.5, label='Two-Stage MSE')
ax2.plot(years, track_diff_naive, '-', color='#9E9E9E', linewidth=1, alpha=0.6, label='Naive')
ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
ax2.fill_between(years, track_diff_task, 0, alpha=0.1, color='#1E88E5')
ax2.fill_between(years, track_diff_mse, 0, alpha=0.1, color='#E53935')

ax2.set_xlabel('Year', fontsize=12)
ax2.set_ylabel('Cumulative\nTracking Diff', fontsize=10)
ax2.legend(fontsize=9, loc='lower left', ncol=3, framealpha=0.9)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(years[0], years[-1])

fig.tight_layout()
fig.savefig("figures/fig12_cumulative_wealth.png", dpi=200, bbox_inches='tight')
fig.savefig("figures/fig12_cumulative_wealth.pdf", bbox_inches='tight')
print("Saved fig12_cumulative_wealth")
plt.close()


# ══════════════════════════════════════════════════════════════════
# Figure B: Method Comparison Bar Chart (K=20)
# ══════════════════════════════════════════════════════════════════

with open("results/baselines/ledoit_wolf_results.json") as f:
    lw_data = json.load(f)
with open("results/baselines/val_tuned_shrinkage_results.json") as f:
    vt_data = json.load(f)
with open("results/baselines/spo_plus_results.json") as f:
    spo_data = json.load(f)
with open("results/nlshrink_baseline.json") as f:
    nl_data = json.load(f)
with open("results/baselines/shrinkage_dfl_results.json") as f:
    sdfl_data = json.load(f)

def avg_te(data, K=20, method_filter=None):
    results = data.get('results', [])
    if method_filter:
        folds = [r for r in results if r.get('K', 0) == K and r.get('method', '') == method_filter]
    else:
        folds = [r for r in results if r.get('K', 0) == K]
    if not folds:
        return None
    return np.mean([r['tracking_error'] for r in folds])

def std_te(data, K=20, method_filter=None):
    results = data.get('results', [])
    if method_filter:
        folds = [r for r in results if r.get('K', 0) == K and r.get('method', '') == method_filter]
    else:
        folds = [r for r in results if r.get('K', 0) == K]
    if not folds:
        return None
    return np.std([r['tracking_error'] for r in folds])

methods = [
    ('Naive\nEqual-Wt', avg_te(naive_data), std_te(naive_data), '#BDBDBD'),
    ('LW 2004\n(MSE)', avg_te(lw_data), std_te(lw_data), '#EF9A9A'),
    ('LW 2020\n(NL Shrink)', avg_te(nl_data, method_filter='lw2020'), std_te(nl_data, method_filter='lw2020'), '#CE93D8'),
    ('Val-Tuned\n(Grid Search)', avg_te(vt_data), std_te(vt_data), '#FFB74D'),
    ('SPO+\nSurrogate', avg_te(spo_data), std_te(spo_data), '#A5D6A7'),
    ('Neural\n+MSE', avg_te(mse_data), std_te(mse_data), '#EF5350'),
    ('Neural\n+Task (DFL)', avg_te(task_data), std_te(task_data), '#1E88E5'),
]

# Filter None
methods = [(n, t, s, c) for n, t, s, c in methods if t is not None]

fig, ax = plt.subplots(figsize=(10, 5))

names = [m[0] for m in methods]
tes = [m[1] for m in methods]
stds = [m[2] for m in methods]
colors = [m[3] for m in methods]

bars = ax.bar(range(len(methods)), tes, color=colors, edgecolor='white', linewidth=1.5, width=0.7)
ax.errorbar(range(len(methods)), tes, yerr=stds, fmt='none', ecolor='black',
            capsize=4, capthick=1.5, linewidth=1.5, zorder=5)

# Value labels
for i, (bar, te) in enumerate(zip(bars, tes)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + stds[i] + 0.0005,
            f'{te:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Highlight best
best_idx = np.argmin(tes)
bars[best_idx].set_edgecolor('#0D47A1')
bars[best_idx].set_linewidth(3)

# Reference lines
ax.axhline(y=tes[1], color='#E53935', linestyle=':', alpha=0.4, linewidth=1)
ax.text(len(methods)-0.5, tes[1]+0.0003, 'LW baseline', fontsize=8, color='#E53935', alpha=0.6, ha='right')

# Gain annotation
gain = (1 - tes[best_idx] / tes[1]) * 100
ax.annotate(f'{gain:.1f}% reduction',
            xy=(best_idx, tes[best_idx]),
            xytext=(best_idx-1.5, tes[best_idx]-0.005),
            fontsize=11, fontweight='bold', color='#0D47A1',
            arrowprops=dict(arrowstyle='->', color='#0D47A1', lw=2))

ax.set_xticks(range(len(methods)))
ax.set_xticklabels(names, fontsize=10)
ax.set_ylabel('Annualized Tracking Error', fontsize=12)
ax.set_title('Method Comparison at $K{=}20$ (S&P 500 Top 100, 9 Folds)', fontsize=14, fontweight='bold')
ax.set_ylim(0, max(tes) * 1.25)
ax.grid(True, alpha=0.3, axis='y')

# Category brackets
ax.text(0, -0.008, 'Baseline', fontsize=8, ha='center', color='gray', fontstyle='italic')
ax.text(2, -0.008, 'Better\nEstimation', fontsize=8, ha='center', color='gray', fontstyle='italic')
ax.text(4, -0.008, 'Alignment\n(1-param)', fontsize=8, ha='center', color='gray', fontstyle='italic')
ax.text(5.5, -0.008, 'DFL\n(385 params)', fontsize=8, ha='center', color='gray', fontstyle='italic')

fig.tight_layout()
fig.savefig("figures/fig13_method_comparison.png", dpi=200, bbox_inches='tight')
fig.savefig("figures/fig13_method_comparison.pdf", bbox_inches='tight')
print("Saved fig13_method_comparison")
plt.close()


# ══════════════════════════════════════════════════════════════════
# Figure C: Covariance Matrix "Attention" Heatmap
# ══════════════════════════════════════════════════════════════════

N = 97
K = 20

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

# Panel 1: MSE gradient weight (all N^2 entries equal)
mse_weight = np.ones((N, N))
im1 = axes[0].imshow(mse_weight, cmap='Reds', vmin=0, vmax=2, aspect='equal')
axes[0].set_title('MSE Gradient\n(equal weight on all $N^2$ entries)', fontsize=11, fontweight='bold')
axes[0].set_xlabel(f'Stock $j$ (N={N})', fontsize=10)
axes[0].set_ylabel(f'Stock $i$ (N={N})', fontsize=10)

# Panel 2: Task gradient weight (concentrated on K×K block)
task_weight = np.zeros((N, N))
# Top-K stocks (simulated as first K for illustration)
selected = np.sort(np.random.RandomState(42).choice(N, K, replace=False))
task_weight[np.ix_(selected, selected)] = 2.0
# Some spillover to cross terms
for s in selected:
    task_weight[s, :] += 0.3
    task_weight[:, s] += 0.3

im2 = axes[1].imshow(task_weight, cmap='Blues', vmin=0, vmax=2.5, aspect='equal')
axes[1].set_title('Task Gradient via QP Jacobian\n(focused on selected $K{\\times}K$ block)', fontsize=11, fontweight='bold')
axes[1].set_xlabel(f'Stock $j$ (N={N})', fontsize=10)

# Panel 3: Wasted effort = MSE - Task (normalized)
wasted = mse_weight.copy()
wasted[np.ix_(selected, selected)] = 0.0  # These entries are useful
for s in selected:
    wasted[s, :] *= 0.5
    wasted[:, s] *= 0.5

im3 = axes[2].imshow(wasted, cmap='Oranges', vmin=0, vmax=1.5, aspect='equal')
axes[2].set_title('MSE Wasted Effort\n(gradient on irrelevant entries)', fontsize=11, fontweight='bold')
axes[2].set_xlabel(f'Stock $j$ (N={N})', fontsize=10)

# Add K×K box outline to all panels
for ax in axes:
    # Draw rectangles around selected blocks (approximate)
    rect_min = min(selected)
    rect_max = max(selected)
    # Since stocks are scattered, just annotate
    ax.set_xticks([0, K, N-1])
    ax.set_xticklabels(['0', f'{K}', f'{N}'])
    ax.set_yticks([0, K, N-1])
    ax.set_yticklabels(['0', f'{K}', f'{N}'])

# Stats annotation
total = N * N
useful = K * K
wasted_pct = (1 - useful / total) * 100
fig.text(0.5, -0.02,
         f'At $K/N = {K}/{N}$: only {useful:,} of {total:,} covariance entries matter ({100-wasted_pct:.1f}%). '
         f'MSE wastes {wasted_pct:.1f}% of gradient effort.',
         ha='center', fontsize=11, fontstyle='italic',
         bbox=dict(boxstyle='round,pad=0.5', fc='#FFF3E0', ec='#FF9800', alpha=0.9))

fig.tight_layout()
fig.subplots_adjust(bottom=0.08)
fig.savefig("figures/fig14_covariance_attention.png", dpi=200, bbox_inches='tight')
fig.savefig("figures/fig14_covariance_attention.pdf", bbox_inches='tight')
print("Saved fig14_covariance_attention")
plt.close()

print("\nDone! Three new figures generated.")
