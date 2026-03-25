#!/usr/bin/env bash
# Experiment 3: Factor Model vs Neural Covariance Estimator
# Tests whether the flexible neural model improves over the linear factor model.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== Exp 3a: Neural + Task loss ==="
python -m src.main model.name=neural training.loss=task

echo "=== Exp 3b: Neural + MSE loss ==="
python -m src.main model.name=neural training.loss=mse

echo "=== Exp 3c: Factor + Task loss (baseline) ==="
python -m src.main model.name=factor training.loss=task

echo "Done. Results in results/"
