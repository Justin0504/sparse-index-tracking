#!/usr/bin/env bash
# Quick smoke test using synthetic data (no internet needed).
# Runs a small experiment to verify the full pipeline works end-to-end.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== Quick test: Synthetic data, Factor + Task loss ==="
python -m src.main \
  data.universe=synthetic \
  data.synthetic.n_stocks=20 \
  data.synthetic.n_days=1260 \
  model.name=factor \
  model.factor.n_factors=5 \
  training.loss=task \
  training.epochs=5 \
  training.patience=3 \
  training.device=cpu \
  "optimization.sparsity_levels=[5,10,20]"

echo ""
echo "=== Quick test: Synthetic data, Neural + MSE loss ==="
python -m src.main \
  data.universe=synthetic \
  data.synthetic.n_stocks=20 \
  data.synthetic.n_days=1260 \
  model.name=neural \
  model.neural.rank=5 \
  training.loss=mse \
  training.epochs=5 \
  training.patience=3 \
  training.device=cpu \
  "optimization.sparsity_levels=[5,10,20]"

echo ""
echo "Smoke test passed. Results in results/"
