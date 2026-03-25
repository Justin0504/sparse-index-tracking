#!/usr/bin/env bash
# Run all four experiments from the proposal:
#   Exp 1: DFL (task loss) vs Two-Stage (MSE loss) — factor model
#   Exp 2: Sparsity–accuracy frontier (sweep K)
#   Exp 3: Factor model vs Neural model
#   Exp 4: Regime analysis (bull/bear/sideways)
#
# Usage:
#   cd /Users/justin/sparse-index-tracking
#   bash scripts/run_all.sh

set -euo pipefail
cd "$(dirname "$0")/.."

echo "========================================="
echo "Experiment 1: DFL vs Two-Stage (Factor)"
echo "========================================="
python -m src.main model.name=factor training.loss=task
python -m src.main model.name=factor training.loss=mse

echo "========================================="
echo "Experiment 2: Sparsity Frontier"
echo "========================================="
python -m src.main model.name=factor training.loss=task \
  "optimization.sparsity_levels=[10,20,50,100,200]"

echo "========================================="
echo "Experiment 3: Factor vs Neural"
echo "========================================="
python -m src.main model.name=neural training.loss=task
python -m src.main model.name=neural training.loss=mse

echo "========================================="
echo "Experiment 4: Regime Analysis"
echo "========================================="
# Regime analysis uses the same run but post-processes by VIX regime.
# Results are split by regime in the analysis notebook.
python -m src.main model.name=factor training.loss=task

echo ""
echo "All experiments complete. Results in results/"
