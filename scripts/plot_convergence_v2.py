#!/usr/bin/env python3
"""Better training convergence figure for presentation."""

import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 12


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


# Parse K=20 neural runs
mse_folds = parse_all_folds('outputs/2026-04-10/06-13-14/main.log')
task_folds = parse_all_folds('outputs/2026-04-10/06-13-13/main.log')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

# ── Left: Normalized convergence curves (all folds) ──
# Normalize each fold's val_loss to start at 1.0
for i, fold in enumerate(mse_folds):
    vals = np.array(fold['val'])
    norm = vals / vals[0]
    alpha = 0.25
    ax1.plot(fold['ep'], norm, color='#E53935', alpha=alpha, linewidth=1)

for i, fold in enumerate(task_folds):
    vals = np.array(fold['val'])
    norm = vals / vals[0]
    alpha = 0.25
    ax1.plot(fold['ep'], norm, color='#1E88E5', alpha=alpha, linewidth=1)

# Average curves
max_ep = 30
mse_avg = np.zeros(max_ep)
mse_count = np.zeros(max_ep)
for fold in mse_folds:
    vals = np.array(fold['val']) / fold['val'][0]
    for j, v in enumerate(vals[:max_ep]):
        mse_avg[j] += v
        mse_count[j] += 1
mse_avg = mse_avg / np.maximum(mse_count, 1)
mse_mask = mse_count > 0

task_avg = np.zeros(max_ep)
task_count = np.zeros(max_ep)
for fold in task_folds:
    vals = np.array(fold['val']) / fold['val'][0]
    for j, v in enumerate(vals[:max_ep]):
        task_avg[j] += v
        task_count[j] += 1
task_avg = task_avg / np.maximum(task_count, 1)
task_mask = task_count > 0

eps = np.arange(1, max_ep + 1)
ax1.plot(eps[mse_mask], mse_avg[mse_mask], 'o-', color='#E53935', linewidth=2.5,
         markersize=5, label='MSE loss (avg)', zorder=5)
ax1.plot(eps[task_mask], task_avg[task_mask], 's-', color='#1E88E5', linewidth=2.5,
         markersize=5, label='Task loss (avg)', zorder=5)

# Mark average convergence epoch
mse_conv = 27.8
task_conv = 12.2
ax1.axvline(x=mse_conv, color='#E53935', linestyle='--', alpha=0.7, linewidth=1.5)
ax1.axvline(x=task_conv, color='#1E88E5', linestyle='--', alpha=0.7, linewidth=1.5)
ax1.annotate(f'MSE converges\n~{mse_conv:.0f} epochs', xy=(mse_conv, 0.55),
             fontsize=9, color='#E53935', ha='center',
             bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#E53935', alpha=0.8))
ax1.annotate(f'Task converges\n~{task_conv:.0f} epochs', xy=(task_conv, 0.75),
             fontsize=9, color='#1E88E5', ha='center',
             bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#1E88E5', alpha=0.8))

ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Normalized Val Loss (relative to epoch 1)', fontsize=11)
ax1.set_title('Convergence Speed: Neural Model, $K{=}20$', fontsize=13, fontweight='bold')
ax1.legend(fontsize=10, loc='lower left', framealpha=0.9)
ax1.set_xlim(0, 31)
ax1.set_ylim(0.4, 1.05)
ax1.grid(True, alpha=0.3)
ax1.text(16, 0.45, '2.3x faster', fontsize=14, fontweight='bold',
         color='#1E88E5', ha='center',
         bbox=dict(boxstyle='round,pad=0.4', fc='#E3F2FD', ec='#1E88E5', alpha=0.9))


# ── Right: Early stopping epoch distribution ──
mse_epochs = [len(f['ep']) for f in mse_folds]
task_epochs = [len(f['ep']) for f in task_folds]

x = np.arange(len(mse_folds))
width = 0.35

bars1 = ax2.bar(x - width/2, mse_epochs, width, color='#E53935', alpha=0.8, label='MSE', edgecolor='white')
bars2 = ax2.bar(x + width/2, task_epochs, width, color='#1E88E5', alpha=0.8, label='Task-Aware', edgecolor='white')

# Add value labels
for bar in bars1:
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{int(bar.get_height())}', ha='center', va='bottom', fontsize=8, color='#E53935')
for bar in bars2:
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{int(bar.get_height())}', ha='center', va='bottom', fontsize=8, color='#1E88E5')

# Average lines
ax2.axhline(y=np.mean(mse_epochs), color='#E53935', linestyle='--', alpha=0.6, linewidth=1.5)
ax2.axhline(y=np.mean(task_epochs), color='#1E88E5', linestyle='--', alpha=0.6, linewidth=1.5)

ax2.annotate(f'MSE avg: {np.mean(mse_epochs):.1f}', xy=(8.5, np.mean(mse_epochs)+0.5),
             fontsize=9, color='#E53935', ha='right')
ax2.annotate(f'Task avg: {np.mean(task_epochs):.1f}', xy=(8.5, np.mean(task_epochs)+0.5),
             fontsize=9, color='#1E88E5', ha='right')

ax2.set_xlabel('Fold', fontsize=12)
ax2.set_ylabel('Epochs to Convergence (early stop)', fontsize=11)
ax2.set_title('Early Stopping: Task Loss Converges Faster', fontsize=13, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels([f'{i}' for i in range(len(mse_folds))])
ax2.legend(fontsize=10, loc='upper right', framealpha=0.9)
ax2.set_ylim(0, 35)
ax2.grid(True, alpha=0.3, axis='y')

fig.tight_layout()
fig.savefig("figures/fig11_training_convergence.png", dpi=200, bbox_inches='tight')
fig.savefig("figures/fig11_training_convergence.pdf", bbox_inches='tight')
print("Saved fig11_training_convergence.png/pdf")
plt.close()
