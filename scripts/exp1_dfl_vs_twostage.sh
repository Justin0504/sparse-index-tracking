#!/usr/bin/env bash
# Experiment 1: DFL (task loss) vs Two-Stage (MSE loss)
# Compares end-to-end training through the QP vs traditional predict-then-optimize.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== Exp 1a: Factor model + Task loss (DFL) ==="
python -m src.main model.name=factor training.loss=task

echo "=== Exp 1b: Factor model + MSE loss (Two-Stage) ==="
python -m src.main model.name=factor training.loss=mse

echo "Done. Results in results/"
