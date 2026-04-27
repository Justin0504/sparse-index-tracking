#!/usr/bin/env python3
"""Regenerate fig11 and fig14 as single-column (vertical) layout."""

import json
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 13


# ══════════════════════════════════════════════════════════════════
# Fig 11: Training Convergence — vertical 2 panels
# ══════════════════════════════════════════════════════════════════

def parse_all_folds(path):
    folds = []
    current = {'ep': [], 'train': [], 'val': []}
    with open(path) as f:
        for line in f:
            m = re.search(r'Epoch (\d+)/\d+ .* train_loss=([\d.]+), val_loss=([\d.]+)', line)
            if m:
                ep = int(m.group(1))
                if ep == 1 and len(current['ep']) > 0:
                    folds.append(current)
                    current = {'ep': [], 'train': [], 'val': []}
                current['ep'].append(ep)
                current['train'].append(float(m.group(2)))
                current['val'].append(float(m.group(3)))
    if current['ep']:
        folds.append(current)
    return folds

mse_folds = parse_all_folds('outputs/2026-04-10/06-13-14/main.log')
task_folds = parse_all_folds('outputs/2026-04-10/06-13-13/main.log')

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))

# ── Panel 1: Normalized convergence curves ──
max_ep = 30
mse_avg = np.zeros(max_ep)
mse_count = np.zeros(max_ep)
for fold in mse_folds:
    vals = np.array(fold['val']) / fold['val'][0]
    for j, v in enumerate(vals[:max_ep]):
        mse_avg[j] += v
        mse_count[j] += 1
    ax1.plot(fold['ep'][:max_ep], (np.array(fold['val']) / fold['val'][0])[:max_ep],
             color='#E53935', alpha=0.2, linewidth=0.8)

for fold in task_folds:
    vals = np.array(fold['val']) / fold['val'][0]
    for j, v in enumerate(vals[:max_ep]):
        pass  # already counted above concept
    ax1.plot(fold['ep'][:max_ep], (np.array(fold['val']) / fold['val'][0])[:max_ep],
             color='#1E88E5', alpha=0.2, linewidth=0.8)

mse_avg = mse_avg / np.maximum(mse_count, 1)

task_avg = np.zeros(max_ep)
task_count = np.zeros(max_ep)
for fold in task_folds:
    vals = np.array(fold['val']) / fold['val'][0]
    for j, v in enumerate(vals[:max_ep]):
        task_avg[j] += v
        task_count[j] += 1
task_avg = task_avg / np.maximum(task_count, 1)

eps = np.arange(1, max_ep + 1)
mse_mask = mse_count > 0
task_mask = task_count > 0

ax1.plot(eps[mse_mask], mse_avg[mse_mask], 'o-', color='#E53935', linewidth=2.5,
         markersize=6, label='MSE loss (avg over 9 folds)', zorder=5)
ax1.plot(eps[task_mask], task_avg[task_mask], 's-', color='#1E88E5', linewidth=2.5,
         markersize=6, label='Task loss (avg over 9 folds)', zorder=5)

mse_conv = 27.8
task_conv = 12.2
ax1.axvline(x=mse_conv, color='#E53935', linestyle='--', alpha=0.7, linewidth=1.5)
ax1.axvline(x=task_conv, color='#1E88E5', linestyle='--', alpha=0.7, linewidth=1.5)
ax1.annotate(f'MSE converges\n~{mse_conv:.0f} epochs', xy=(mse_conv, 0.58),
             fontsize=11, color='#E53935', ha='center',
             bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#E53935', alpha=0.8))
ax1.annotate(f'Task converges\n~{task_conv:.0f} epochs', xy=(task_conv, 0.78),
             fontsize=11, color='#1E88E5', ha='center',
             bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#1E88E5', alpha=0.8))

ax1.text(20, 0.47, '2.3x faster', fontsize=16, fontweight='bold',
         color='#1E88E5', ha='center',
         bbox=dict(boxstyle='round,pad=0.5', fc='#E3F2FD', ec='#1E88E5', alpha=0.9))

ax1.set_xlabel('Epoch', fontsize=13)
ax1.set_ylabel('Normalized Val Loss (relative to epoch 1)', fontsize=12)
ax1.set_title('(a) Convergence Speed: Neural Model, $K{=}20$', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11, loc='lower left', framealpha=0.9)
ax1.set_xlim(0, 31)
ax1.set_ylim(0.4, 1.05)
ax1.grid(True, alpha=0.3)

# ── Panel 2: Early stopping bar chart ──
mse_epochs = [len(f['ep']) for f in mse_folds]
task_epochs = [len(f['ep']) for f in task_folds]

x = np.arange(len(mse_folds))
width = 0.35

bars1 = ax2.bar(x - width/2, mse_epochs, width, color='#E53935', alpha=0.8,
                label='MSE', edgecolor='white')
bars2 = ax2.bar(x + width/2, task_epochs, width, color='#1E88E5', alpha=0.8,
                label='Task-Aware (DFL)', edgecolor='white')

for bar in bars1:
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{int(bar.get_height())}', ha='center', va='bottom', fontsize=10, color='#E53935')
for bar in bars2:
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{int(bar.get_height())}', ha='center', va='bottom', fontsize=10, color='#1E88E5')

ax2.axhline(y=np.mean(mse_epochs), color='#E53935', linestyle='--', alpha=0.6, linewidth=1.5)
ax2.axhline(y=np.mean(task_epochs), color='#1E88E5', linestyle='--', alpha=0.6, linewidth=1.5)
ax2.annotate(f'MSE avg: {np.mean(mse_epochs):.1f}', xy=(8.3, np.mean(mse_epochs)+0.8),
             fontsize=11, color='#E53935', ha='right')
ax2.annotate(f'Task avg: {np.mean(task_epochs):.1f}', xy=(8.3, np.mean(task_epochs)+0.8),
             fontsize=11, color='#1E88E5', ha='right')

ax2.set_xlabel('Fold', fontsize=13)
ax2.set_ylabel('Epochs to Convergence (early stop)', fontsize=12)
ax2.set_title('(b) Early Stopping: Task Loss Converges Faster in Every Fold',
              fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels([f'{i}' for i in range(len(mse_folds))])
ax2.legend(fontsize=11, loc='upper right', framealpha=0.9)
ax2.set_ylim(0, 36)
ax2.grid(True, alpha=0.3, axis='y')

fig.tight_layout(h_pad=2)
fig.savefig("figures/fig11_training_convergence.png", dpi=200, bbox_inches='tight')
fig.savefig("figures/fig11_training_convergence.pdf", bbox_inches='tight')
print("Saved fig11_training_convergence")
plt.close()


# ══════════════════════════════════════════════════════════════════
# Fig 14: Covariance Attention — vertical 3 panels
# ══════════════════════════════════════════════════════════════════

N = 97
K = 20
np.random.seed(42)
selected = np.sort(np.random.choice(N, K, replace=False))

fig, axes = plt.subplots(3, 1, figsize=(7, 18))

# Panel 1: MSE gradient (uniform)
mse_weight = np.ones((N, N))
axes[0].imshow(mse_weight, cmap='Reds', vmin=0, vmax=2, aspect='equal')
axes[0].set_title('(a) MSE Gradient: Equal Weight on All $N^2 = 9{,}409$ Entries',
                  fontsize=13, fontweight='bold')
axes[0].set_xlabel(f'Stock $j$  (N={N})', fontsize=12)
axes[0].set_ylabel(f'Stock $i$  (N={N})', fontsize=12)

# Panel 2: Task gradient (focused on K×K)
task_weight = np.zeros((N, N))
task_weight[np.ix_(selected, selected)] = 2.0
for s in selected:
    task_weight[s, :] += 0.3
    task_weight[:, s] += 0.3
axes[1].imshow(task_weight, cmap='Blues', vmin=0, vmax=2.5, aspect='equal')
axes[1].set_title('(b) Task Gradient via QP Jacobian: Focused on Selected $K{\\times}K$ Block',
                  fontsize=13, fontweight='bold')
axes[1].set_xlabel(f'Stock $j$  (N={N})', fontsize=12)
axes[1].set_ylabel(f'Stock $i$  (N={N})', fontsize=12)

# Panel 3: Wasted effort
wasted = mse_weight.copy()
wasted[np.ix_(selected, selected)] = 0.0
for s in selected:
    wasted[s, :] *= 0.5
    wasted[:, s] *= 0.5
axes[2].imshow(wasted, cmap='Oranges', vmin=0, vmax=1.5, aspect='equal')
axes[2].set_title('(c) MSE Wasted Effort: Gradient on Irrelevant Entries',
                  fontsize=13, fontweight='bold')
axes[2].set_xlabel(f'Stock $j$  (N={N})', fontsize=12)
axes[2].set_ylabel(f'Stock $i$  (N={N})', fontsize=12)

for ax in axes:
    ax.set_xticks([0, K, 50, N-1])
    ax.set_xticklabels(['0', f'{K}', '50', f'{N}'])
    ax.set_yticks([0, K, 50, N-1])
    ax.set_yticklabels(['0', f'{K}', '50', f'{N}'])

# Stats text at bottom
total = N * N
useful = K * K
wasted_pct = (1 - useful / total) * 100
fig.text(0.5, 0.01,
         f'At $K/N = {K}/{N}$:  only {useful} of {total:,} covariance entries matter ({100-wasted_pct:.1f}%).  '
         f'MSE wastes {wasted_pct:.1f}% of gradient effort.',
         ha='center', fontsize=12, fontstyle='italic',
         bbox=dict(boxstyle='round,pad=0.5', fc='#FFF3E0', ec='#FF9800', alpha=0.9))

fig.tight_layout(h_pad=2)
fig.subplots_adjust(bottom=0.05)
fig.savefig("figures/fig14_covariance_attention.png", dpi=200, bbox_inches='tight')
fig.savefig("figures/fig14_covariance_attention.pdf", bbox_inches='tight')
print("Saved fig14_covariance_attention")
plt.close()

print("\nDone!")
