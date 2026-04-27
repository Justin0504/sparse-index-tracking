#!/usr/bin/env python3
"""Generate two supplementary figures for the presentation."""

import json
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 12

# ── Figure 1: Gradient Alignment vs K/N ──────────────────────────

with open("results/gradient_alignment.json") as f:
    ga = json.load(f)

N = 97
Ks = [5, 10, 20, 30, 50]
KN = [k / N for k in Ks]

# Theoretical bound: cos <= sqrt(2K/N)
KN_smooth = np.linspace(0.02, 0.55, 200)
bound_smooth = np.sqrt(2 * KN_smooth)
bound_smooth = np.minimum(bound_smooth, 1.0)

# Measured values (factor model — higher signal than neural)
factor_mean = [ga["factor"][str(k)]["mean_cos"] for k in Ks]
factor_std = [ga["factor"][str(k)]["std_cos"] for k in Ks]
neural_mean = [ga["neural"][str(k)]["mean_cos"] for k in Ks]
neural_std = [ga["neural"][str(k)]["std_cos"] for k in Ks]
theo_bound = [ga["factor"][str(k)]["theoretical_bound"] for k in Ks]

# Experimental TE gain (from paper results)
te_gains = [19.3, 11.9, 11.9, 4.9, 0.0]  # % reduction at K=5,10,20,30,50

fig, ax1 = plt.subplots(1, 1, figsize=(7, 4.5))

# Theoretical bound curve
ax1.plot(KN_smooth, bound_smooth, 'k--', linewidth=2, label='Theoretical bound $\\sqrt{2K/N}$', zorder=3)

# Measured cos (factor)
ax1.errorbar(KN, factor_mean, yerr=factor_std, fmt='o-', color='#2196F3',
             markersize=8, capsize=4, linewidth=1.5, label='Factor model (measured)', zorder=4)

# Measured cos (neural)
ax1.errorbar(KN, neural_mean, yerr=neural_std, fmt='s-', color='#FF9800',
             markersize=7, capsize=4, linewidth=1.5, label='Neural model (measured)', zorder=4)

# TE gain on secondary axis
ax2 = ax1.twinx()
ax2.bar(KN, te_gains, width=0.035, alpha=0.25, color='#4CAF50', label='TE gain (%)', zorder=1)
ax2.set_ylabel('Task-Aware TE Gain (%)', color='#4CAF50', fontsize=11)
ax2.tick_params(axis='y', labelcolor='#4CAF50')
ax2.set_ylim(0, 25)

# Annotations
ax1.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
ax1.annotate('Near-orthogonal\n(large gap)', xy=(0.05, 0.05), fontsize=9,
             color='#d32f2f', fontstyle='italic')
ax1.annotate('Well-aligned\n(no gap)', xy=(0.42, 0.85), fontsize=9,
             color='#388E3C', fontstyle='italic')

ax1.set_xlabel('$K/N$ (Sparsity Ratio)', fontsize=12)
ax1.set_ylabel('$\\cos(\\nabla_{\\mathrm{MSE}}, \\nabla_{\\mathrm{task}})$', fontsize=12)
ax1.set_title('Gradient Alignment Bound vs. Empirical Measurement', fontsize=13, fontweight='bold')
ax1.set_xlim(0, 0.58)
ax1.set_ylim(-0.15, 1.1)

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9, framealpha=0.9)

ax1.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("figures/fig10_gradient_alignment.png", dpi=200, bbox_inches='tight')
fig.savefig("figures/fig10_gradient_alignment.pdf", bbox_inches='tight')
print("Saved fig10_gradient_alignment.png/pdf")
plt.close()


# ── Figure 2: Training Convergence (MSE vs Task) ────────────────

def parse_log(path):
    """Extract epoch, train_loss, val_loss from Hydra log."""
    epochs, train, val = [], [], []
    with open(path) as f:
        for line in f:
            m = re.search(r'Epoch (\d+)/\d+ — train_loss=([\d.]+), val_loss=([\d.]+)', line)
            if m:
                epochs.append(int(m.group(1)))
                train.append(float(m.group(2)))
                val.append(float(m.group(3)))
    return epochs, train, val

# Use 2026-04-12 paired runs (fold 0): MSE then Task
mse_log = "outputs/2026-04-12/08-11-57/main.log"
task_log = "outputs/2026-04-12/08-15-21/main.log"

mse_ep, mse_train, mse_val = parse_log(mse_log)
task_ep, task_train, task_val = parse_log(task_log)

# These logs contain multiple folds concatenated. Extract first fold only (first 30 epochs max)
def extract_first_fold(epochs, train, val, max_ep=50):
    out_e, out_t, out_v = [], [], []
    for e, t, v in zip(epochs, train, val):
        if e == 1 and len(out_e) > 0:
            break  # new fold starts
        out_e.append(e)
        out_t.append(t)
        out_v.append(v)
        if e >= max_ep:
            break
    return out_e, out_t, out_v

mse_ep1, mse_t1, mse_v1 = extract_first_fold(mse_ep, mse_train, mse_val)
task_ep1, task_t1, task_v1 = extract_first_fold(task_ep, task_train, task_val)

fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 4.5))

# Left: Both training losses (different scales, so normalize)
ax_left.plot(mse_ep1, mse_v1, 'o-', color='#E53935', markersize=4, linewidth=1.5, label='MSE loss (val)')
ax_left.plot(mse_ep1, mse_t1, 's--', color='#E53935', markersize=3, linewidth=1, alpha=0.5, label='MSE loss (train)')
ax_left.plot(task_ep1, task_v1, 'o-', color='#1E88E5', markersize=4, linewidth=1.5, label='Task loss (val)')
ax_left.plot(task_ep1, task_t1, 's--', color='#1E88E5', markersize=3, linewidth=1, alpha=0.5, label='Task loss (train)')

ax_left.set_xlabel('Epoch', fontsize=12)
ax_left.set_ylabel('Loss (Tracking Error)', fontsize=12)
ax_left.set_title('Training Convergence: Neural Model (Fold 0)', fontsize=13, fontweight='bold')
ax_left.legend(fontsize=9, framealpha=0.9)
ax_left.grid(True, alpha=0.3)

# Mark convergence points
mse_converge = next((i for i in range(3, len(mse_v1))
                     if abs(mse_v1[i] - mse_v1[i-1]) < 1e-6), len(mse_v1)-1)
task_converge = next((i for i in range(3, len(task_v1))
                      if abs(task_v1[i] - task_v1[i-1]) < 1e-6), len(task_v1)-1)

if mse_converge < len(mse_ep1) and task_converge < len(task_ep1):
    ax_left.axvline(x=mse_ep1[mse_converge], color='#E53935', linestyle=':', alpha=0.5)
    ax_left.axvline(x=task_ep1[task_converge], color='#1E88E5', linestyle=':', alpha=0.5)

# Right: Structured shrinkage MSE vs Task from diagnostics
with open("results/shrinkage_diagnostics.json") as f:
    diag = json.load(f)

# Find structured MSE fold 0 and structured task fold 0
struct_mse = next(d for d in diag["diagnostics"]
                  if d["model_type"] == "structured" and d["loss"] == "mse" and d["fold"] == 0)
struct_task = next(d for d in diag["diagnostics"]
                   if d["model_type"] == "structured" and d["loss"] == "task" and d["fold"] == 0)

ax_right.plot(range(1, len(struct_mse["val_losses"])+1), struct_mse["val_losses"],
              'o-', color='#E53935', markersize=4, linewidth=1.5, label='Structured + MSE (val)')
ax_right.plot(range(1, len(struct_mse["train_losses"])+1), struct_mse["train_losses"],
              's--', color='#E53935', markersize=3, linewidth=1, alpha=0.5, label='Structured + MSE (train)')
ax_right.plot(range(1, len(struct_task["val_losses"])+1), struct_task["val_losses"],
              'o-', color='#1E88E5', markersize=4, linewidth=1.5, label='Structured + Task (val)')
ax_right.plot(range(1, len(struct_task["train_losses"])+1), struct_task["train_losses"],
              's--', color='#1E88E5', markersize=3, linewidth=1, alpha=0.5, label='Structured + Task (train)')

ax_right.set_xlabel('Epoch', fontsize=12)
ax_right.set_ylabel('Loss', fontsize=12)
ax_right.set_title('Structured Shrinkage (12 params, Fold 0)', fontsize=13, fontweight='bold')
ax_right.legend(fontsize=9, framealpha=0.9)
ax_right.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig("figures/fig11_training_convergence.png", dpi=200, bbox_inches='tight')
fig.savefig("figures/fig11_training_convergence.pdf", bbox_inches='tight')
print("Saved fig11_training_convergence.png/pdf")
plt.close()

print("\nDone! Two new figures generated.")
