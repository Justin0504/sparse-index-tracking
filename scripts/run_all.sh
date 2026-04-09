#!/usr/bin/env bash
# Run all six experiments from the revised proposal.
#
# Usage:
#   cd /Users/justin/sparse-index-tracking
#   bash scripts/run_all.sh
#
# Each run saves results to results/<model>_<loss>_results.json
# Use Hydra overrides for sweeps.

set -euo pipefail
cd "$(dirname "$0")/.."

echo "========================================="
echo "Experiment 1: Cardinality Sweep (Hard-K)"
echo "DFL vs Two-Stage across K={10,20,50,100}"
echo "========================================="
python -m src.main model.name=factor training.loss=task \
  "optimization.sparsity_levels=[10,20,50,100]" \
  optimization.qp_formulation=hard_k
python -m src.main model.name=factor training.loss=mse \
  "optimization.sparsity_levels=[10,20,50,100]" \
  optimization.qp_formulation=hard_k

echo "========================================="
echo "Experiment 2: Model Misspecification"
echo "Factor count sweep K_f={3,5,10,20} at K=20"
echo "========================================="
for nf in 3 5 10 20; do
  echo "--- n_factors=$nf, task loss ---"
  python -m src.main model.name=factor training.loss=task \
    model.factor.n_factors=$nf \
    "optimization.sparsity_levels=[20]"
  echo "--- n_factors=$nf, MSE loss ---"
  python -m src.main model.name=factor training.loss=mse \
    model.factor.n_factors=$nf \
    "optimization.sparsity_levels=[20]"
done

echo "========================================="
echo "Experiment 3: Shrinkage Intensity Tuning"
echo "Learnable alpha (DFL) vs analytical (MSE)"
echo "========================================="
python -m src.main model.name=shrinkage training.loss=task \
  "optimization.sparsity_levels=[10,20,50]"
python -m src.main model.name=shrinkage training.loss=mse \
  "optimization.sparsity_levels=[10,20,50]"

echo "========================================="
echo "Experiment 4: Hard-K vs L1 Relaxation"
echo "========================================="
python -m src.main model.name=factor training.loss=task \
  optimization.qp_formulation=l1 optimization.l1_lambda=0.01 \
  "optimization.sparsity_levels=[10,20,50]"
python -m src.main model.name=factor training.loss=task \
  optimization.qp_formulation=hard_k \
  "optimization.sparsity_levels=[10,20,50]"

echo "========================================="
echo "Experiment 5: Market Regime Analysis"
echo "(Post-process results by VIX regime)"
echo "========================================="
# Uses same factor-task run from Exp 1; regime split done in analysis.
echo "Skipping — uses Exp 1 results, split by VIX in analysis notebook."

echo "========================================="
echo "Experiment 6: Fixed vs Dynamic Selection"
echo "========================================="
python -m src.main model.name=factor training.loss=task \
  optimization.selection=market_cap \
  "optimization.sparsity_levels=[10,20,50]"
python -m src.main model.name=factor training.loss=task \
  optimization.selection=tracking_score \
  "optimization.sparsity_levels=[10,20,50]"

echo "========================================="
echo "Baselines: Neural Model"
echo "========================================="
python -m src.main model.name=neural training.loss=task \
  "optimization.sparsity_levels=[10,20,50]"
python -m src.main model.name=neural training.loss=mse \
  "optimization.sparsity_levels=[10,20,50]"

echo ""
echo "All experiments complete. Results in results/"
