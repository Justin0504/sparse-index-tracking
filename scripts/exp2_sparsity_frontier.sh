#!/usr/bin/env bash
# Experiment 2: Sparsity-Accuracy Frontier
# Sweeps K ∈ {10, 20, 50, 100, 200} to show tracking error vs portfolio size.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== Exp 2: Sparsity frontier (DFL) ==="
python -m src.main model.name=factor training.loss=task \
  "optimization.sparsity_levels=[10,20,50,100,200]"

echo "=== Exp 2: Sparsity frontier (Two-Stage) ==="
python -m src.main model.name=factor training.loss=mse \
  "optimization.sparsity_levels=[10,20,50,100,200]"

echo "Done. Results in results/"
