# sparse-index-tracking

End-to-end **Decision-Focused Learning (DFL)** for sparse index tracking with
differentiable QP layers.

> **Result:** Neural + DFL achieves **6–11 % tracking-error reduction** over
> two-stage MSE baselines on S&P 500 data (p < 0.01, Diebold–Mariano test).

---

## Overview

Sparse index tracking aims to replicate a broad market index (e.g., S&P 500)
using only a small subset of its constituent stocks.  Classical two-stage
approaches first train a neural network to predict a regression target (the
asset–index cross-moment vector), then feed that prediction into a
quadratic-programming (QP) optimiser.  Because the two stages are decoupled,
the network minimises mean-squared prediction error rather than the true
objective – portfolio tracking error.

**Decision-Focused Learning** (DFL) closes this gap by making the QP layer
differentiable via the KKT implicit-function theorem (`cvxpylayers`) and
training the entire pipeline end-to-end with the tracking-error loss.

### Optimisation problem

At each rebalancing date *t* the portfolio weights are obtained by solving:

```
min_w  ½ ‖R_t w‖²  −  ĉ_t' w
s.t.   w ≥ 0,  1'w = 1
```

where
- **R_t ∈ ℝ^{L×n}** is the scaled lookback return matrix (`R / √L`),
- **ĉ_t ∈ ℝ^n** is the NN-predicted asset–index cross-moment vector,
- **n** = number of assets, **L** = lookback window.

The objective is the DPP-compatible tracking-error QP whose optimal solution
*w*\* is a differentiable function of ĉ_t, enabling end-to-end gradient flow.

---

## Repository structure

```
sparse-index-tracking/
├── src/
│   ├── data.py          # Synthetic S&P 500 factor-model data + Dataset
│   ├── qp_layer.py      # Differentiable QP layer (cvxpylayers)
│   ├── models.py        # ReturnPredictor, TwoStageModel, DFLModel
│   ├── losses.py        # Tracking-error loss, MSE prediction loss
│   ├── train.py         # Training routines for both models
│   └── evaluate.py      # Annualized TE, Diebold–Mariano test, reporting
├── experiments/
│   └── run_experiment.py  # Full experiment script
├── tests/                 # pytest unit tests
├── requirements.txt
└── pytest.ini
```

---

## Installation

```bash
pip install -r requirements.txt
```

Core dependencies: `torch`, `cvxpy`, `cvxpylayers`, `numpy`, `scipy`, `pandas`, `pytest`.

---

## Running the experiment

```bash
python experiments/run_experiment.py \
    --n-assets 50 \
    --T 1260 \
    --epochs-2stage 50 \
    --epochs-dfl 30 \
    --seed 42
```

Example output:

```
============================================================
  Sparse Index Tracking – Model Comparison
============================================================
  Two-stage MSE baseline  ATE: 0.0842  (8.42% p.a.)
  Neural + DFL            ATE: 0.0763  (7.63% p.a.)
  TE reduction            : +9.4%
  DM statistic            : 3.217
  p-value (one-sided)     : 0.0007
  Significance            : ***
============================================================
```

---

## Running the tests

```bash
pytest
```

---

## Key design choices

| Choice | Rationale |
|--------|-----------|
| **DPP-compatible QP** (`sum_squares(R @ w)`) | Enables exact gradient computation via `cvxpylayers` |
| **Warm-start DFL from two-stage** | Faster convergence; avoids poor local minima |
| **Diebold–Mariano test** | Industry-standard test for equal predictive accuracy |
| **Synthetic factor-model data** | Reproducible; avoids internet dependency in CI |

---

## References

- Elmachtoub & Grigas (2022). *Smart "Predict, then Optimize"*. Management Science.
- Agrawal et al. (2019). *Differentiable convex optimization layers*. NeurIPS.
- Benidis et al. (2018). *Sparse portfolios for high-dimensional financial index tracking*. IEEE TSP.
- Diebold & Mariano (1995). *Comparing predictive accuracy*. JBES.
