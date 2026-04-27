# PPT Outline: Task-Aware Covariance Estimation for Sparse Index Tracking

**Course:** DSO 619 — End-to-End Learning and Optimization
**Team:** BrewNet (Aojie Yuan, Haiyue Zhang)
**Duration:** 15 min presentation + 5 min Q&A

---

## Figure Checklist

All generated figures are in `figures/` folder. Check marks = ready, TODO = needs manual creation.

| Figure File | Used in Slide | Status |
|-------------|---------------|--------|
| `fig12_cumulative_wealth.png` | Slide 2 (Problem) | READY |
| **[TODO] Two-Stage Pipeline Diagram** | **Slide 3 (PTO Gap)** | **NEED TO DRAW** |
| `fig14_covariance_attention.png` | Slide 4 (Gradient Bound) | READY |
| `fig10_gradient_alignment.png` | Slide 4 (Gradient Bound) | READY (backup) |
| **[TODO] DFL Pipeline Diagram (with backprop arrow)** | **Slide 5 (DFL Solution)** | **NEED TO DRAW** |
| `fig1_sparsity_frontier.png` | Slide 8 (Q1) | READY |
| `fig13_method_comparison.png` | Slide 9 (Q2) | READY |
| `fig11_training_convergence.png` | Slide 9 (Q2) | READY (backup) |
| `fig2_turnover_frontier.png` | Slide 11 (Robustness) | READY |
| `fig9_regime_analysis.png` | Slide 11 (Robustness) | READY |
| `fig3_te_boxplot_K20.png` | Slide 11 (Robustness) | READY |

**Backup figures** (for Q&A or appendix): `fig4`, `fig5`, `fig6`, `fig7`, `fig8`.

### TODO: Two diagrams need manual creation

**1. Two-Stage Pipeline (Slide 3):**
Draw a left-to-right flow diagram:
```
[Historical Returns] --> [Covariance Model (MSE loss)] --> [QP Solver] --> [Portfolio Weights]
                          predict Sigma                    min w'Sigma w
                          (trained on Frobenius norm)      (cardinality K, long-only)
```
- Add a red "X" on the arrow between QP and model: NO gradient flows backward
- Label: "Two-Stage Predict-then-Optimize"

**2. DFL Pipeline with Backprop (Slide 5):**
Same 3-box layout, but add backward gradient arrows:
```
[Historical Returns] --> [Covariance Model (theta)] --> [Differentiable QP] --> [Portfolio Weights]
                          predict Sigma                  cvxpylayers              |
                              ^                                                  v
                              |_____________ task loss gradient ________________ [Tracking Error Loss]
                                  (backprop through KKT conditions)
```
- Use green/blue for the backward arrow
- Label: "End-to-End Decision-Focused Learning"

**Tools:** Excalidraw (excalidraw.com), Google Slides shapes, or Canva.

---

## Narrative Arc

**Central thesis:** "Align your estimation objective with your decision objective."

| Act | Slides | Purpose |
|-----|--------|---------|
| THE PROBLEM | 2-4 | What is sparse tracking? Why does two-stage fail? |
| OUR SOLUTION | 5-6 | DFL pipeline and algorithm |
| EVIDENCE | 7-10 | Setup + three experimental questions |
| SYNTHESIS | 11-13 | Robustness, takeaways, questions |

---

## Slide 1: Title (30s)

- Title: **Task-Aware Covariance Estimation for Sparse Index Tracking**
- Subtitle: DSO 619 — End-to-End Learning and Optimization
- Team BrewNet: Aojie Yuan, Haiyue Zhang
- Spring 2026

---

## Slide 2: The Problem — Sparse Index Tracking (1.5 min)

**Hook:** "Can you replicate the S&P 500 with only 20 stocks?"

- Index funds hold all 500 stocks -> costly for small funds
- Sparse tracking: select K << N stocks, find weights to minimize tracking error
- TE = annualized std of return difference
- Real-world: ETFs, pension funds, UCITS compliance
- Our result: 20 stocks track S&P 500 over 9 years, TE under 3%

**Figure:** `fig12_cumulative_wealth.png`

**Transition:** "But how do we choose which 20 stocks and set their weights?"

---

## Slide 3: The Decision Problem + PTO Gap (2 min)

**Formal QP:**
$$\min_w \quad (w - w_{idx})^\top \Sigma (w - w_{idx}) \quad \text{s.t.} \quad \|w\|_0 \leq K, \; w \geq 0, \; \mathbf{1}^\top w = 1$$

- Decision variable: portfolio weights w
- Uncertain parameter: covariance matrix Sigma
- Constraints: cardinality K, long-only, fully invested

**Two-Stage Approach:**
- Stage 1: Estimate Sigma by minimizing MSE (Frobenius norm)
- Stage 2: Plug into QP solver
- Problem: no feedback from decision quality to the estimation model

**Figure:** **[TODO] Two-Stage Pipeline Diagram**

**Key observation:** QP uses only K^2 = 400 entries; MSE optimizes all N^2 = 9,409 entries equally. 95.7% of gradient effort is wasted.

---

## Slide 4: Why Sparsity Amplifies the Gap (1.5 min)

**Gradient Alignment Bound (Proposition 1):**
$$\cos(\nabla_\text{MSE}, \nabla_\text{task}) \leq \sqrt{\frac{2K}{N}}$$

| K/N | cos bound | Prediction |
|-----|-----------|------------|
| 5/97 | 0.32 | Severe misalignment |
| 20/97 | 0.64 | Moderate — significant gain possible |
| 50/97 | >= 1.0 | Well-aligned, MSE is near-optimal |

- QP Jacobian (from KKT differentiation) zeroes gradients for unselected stocks
- Testable prediction: DFL helps most when K/N is small, not at all when K/N > 0.5
- Our experiments confirm this exactly (see Slide 8)

**Figure:** `fig14_covariance_attention.png` (3-panel heatmap)

---

## Slide 5: Our Solution — Decision-Focused Learning (1.5 min)

**Figure:** **[TODO] DFL Pipeline Diagram with backprop arrow**

- Core idea: train covariance model to minimize downstream tracking error, not upstream prediction error
- Backward pass: differentiate through QP via KKT implicit differentiation (cvxpylayers)
- Stock selection is detached — only weight optimization is differentiable

**Two paths to alignment:**

| Path | When to use | Method | Time |
|------|-------------|--------|------|
| Grid search on val TE | 1-param models (LW alpha) | Enumerate, pick best | 0.6s |
| Differentiable QP (DFL) | 12+ param models | Backprop through KKT | 26s/fold |

**Key distinction:** DFL is not always necessary — but when the model has too many parameters for grid search, it's the only option.

---

## Slide 6: DFL Training Algorithm (1 min)

**Pseudocode** — see gamma_prompt.md for full formatted version.

Key steps:
1. Compute features from historical returns
2. Predict covariance: Sigma = f(x; theta)
3. Select top-K stocks (detached from gradient)
4. Solve differentiable QP -> portfolio weights
5. Compute realized tracking error over forward window
6. **KKT differentiation** gives dw*/dSigma analytically
7. Backpropagate through chain: dL/dtheta = dL/dw* · dw*/dSigma · dSigma/dtheta

**Key insight:** Step 6 — active constraints (w_i > 0 vs w_i = 0) determine which covariance entries receive gradient. Unselected stocks are automatically zeroed out.

---

## Slide 7: Experimental Setup (1 min)

- **Data:** S&P 500 top 100, daily prices 2006-2025 (19 years)
- **Protocol:** Rolling window — 3yr train / 6mo val / 1yr test -> 9 folds
- **Sparsity levels:** K in {5, 10, 20, 30, 50}
- **Baselines:** LW 2004, LW 2020, Val-Tuned grid search, SPO+ surrogate, Random search
- **Models:** Shrinkage (1p), Structured (12p), Conditional neural (385p)
- **Statistics:** Wilcoxon signed-rank test across 9 folds

"Three questions drive our experiments."

> Fast slide. Don't linger.

---

## Slide 8: Q1 — Alignment Helps, Sparsity Drives the Gain (1.5 min)

**ALL NUMBERS VERIFIED FROM `results/neural_*_results.json`:**

| K | Two-Stage TE | Task-Aware TE | Gain | p-value | Wins |
|---|-------------|--------------|------|---------|------|
| 5 | 0.0877 | 0.0794 | -9.5% | 0.002 | 9/9 |
| 10 | 0.0530 | 0.0495 | -6.6% | 0.002 | 9/9 |
| 20 | 0.0326 | 0.0287 | -11.9% | 0.002 | 9/9 |
| 30 | 0.0212 | 0.0202 | -4.5% | 0.002 | 8/9 |
| 50 | 0.0100 | 0.0100 | -0.2% | 0.102 | 5/9 |

**Key points:**
- Statistically significant at K <= 30 (p = 0.002)
- Pattern matches gradient bound prediction: gains vanish at K=50 (K/N > 0.5)
- Driver is K/N ratio, not model complexity

**Figure:** `fig1_sparsity_frontier.png`

> The sparsity frontier figure is the most powerful visual. Let it tell the story.

---

## Slide 9: Q2 — Decomposing the Mechanism (2 min)

**This is the core intellectual slide.**

**ALL NUMBERS VERIFIED FROM result JSON files (K=20):**

| Method | Params | TE | vs Two-Stage |
|--------|--------|-----|-------------|
| LW + MSE (two-stage) | 1 | 0.0325 | baseline |
| LW + Val-Tuned (grid search) | 1 | 0.0292 | -10.2% |
| LW + SPO+ surrogate | 1 | 0.0290 | -10.8% |
| LW + Task (DFL) | 1 | 0.0287 | -11.7% |
| LW 2020 (nonlinear shrinkage) | 1 | 0.0293 | -9.8% |
| Factor + MSE | ~10 | 0.0325 | 0.0% |
| Factor + Task (DFL) | ~10 | 0.0286 | -12.0% |
| Neural + MSE | 385 | 0.0326 | +0.3% |
| Neural + Task (DFL) | 385 | 0.0287 | -11.7% |

**Narrative (say this in order):**
1. "Val-Tuned grid search (0.6s) nearly matches DFL for 1-param models. DFL is not always necessary."
2. "The mechanism is **objective alignment** — any method achieving alignment (grid search, SPO+, DFL) closes the gap."
3. "All MSE models saturate at TE ~ 0.0325 regardless of complexity — the wrong objective is the bottleneck."
4. "LW 2020 (0.0293) is better statistics, not better alignment. Better estimation != better decisions."

**Figures:** `fig13_method_comparison.png` + `fig11_training_convergence.png` (backup)

---

## Slide 10: Q3 — DFL is Essential for Rich Models (1.5 min)

**ALL NUMBERS VERIFIED FROM result JSON files:**

| K | Method | TE | p-value | Wins |
|---|--------|-----|---------|------|
| 10 | Cond. Shrinkage + MSE | 0.0494 | — | — |
| 10 | Random Search (200 trials) | 0.0496 | — | — |
| 10 | **Cond. Shrinkage + Task** | **0.0490** | 0.014 | 7/9 |
| 20 | Struct. Shrinkage + MSE | 0.0287 | — | — |
| 20 | Random Search (500 trials) | 0.0295 | — | — |
| 20 | **Struct. Shrinkage + Task** | **0.0285** | 0.006 | 8/9 |

**Key points:**
- 500 random search trials cannot optimize 12+ parameters effectively
- DFL: p = 0.006 at K=20, winning 8 of 9 folds
- Scales to full S&P 500 (N = 478): still significant at p = 0.031
- Sweet spot: 12-param structured model (stable, sector-aware)

**Transition:** "This completes the logical chain: alignment is the mechanism (Q2), DFL is how you achieve it for complex models (Q3)."

---

## Slide 11: Robustness (1.5 min)

**Panel A — Turnover (all verified):**
| gamma | TE | Turnover | vs Two-Stage |
|-------|-----|----------|-------------|
| 0 | 0.0287 | 0.030 | -11.9% |
| 0.01 | 0.0299 | 0.003 | -8.2% |
| 0.1 | 0.0319 | 0.000 | -2.2% |

gamma = 0.01: MSE-level turnover, still 8.2% gain. DFL Pareto-dominates.

**Panel B — Market Regimes (all verified):**
| Period | Gain |
|--------|------|
| High-vol | -12.7% |
| Low-vol | -10.5% |
| 2021 (inflation) | -23.5% |
| 2022 (rate hikes) | -19.4% |

Gains in ALL regimes, largest during volatile transitions.

**Panel C — Risk & Concentration (all verified):**
- HHI identical (0.067 vs 0.068) — not a restructuring trick
- Worst-fold TE: -26.8% improvement
- TE std across folds: -59.4% — more consistent

**Figures:** `fig2_turnover_frontier.png` + `fig9_regime_analysis.png` + `fig3_te_boxplot_K20.png`

---

## Slide 12: Three Takeaways (1.5 min)

**1. Structural Principle**
Sparsity amplifies the PTO gap. cos <= sqrt(2K/N) predicts when alignment matters.

**2. Mechanism Decomposition**
The gain is from aligning the objective, not from DFL per se. For 1-param models, grid search suffices.

**3. DFL's Unique Role**
For 12+ param models, differentiable optimization is the only viable training path.

**Practical Decision Rule:**
- K/N > 0.3 -> minimal gain; add more stocks
- 1-param model -> grid search (0.6s)
- 12+ param model -> DFL via differentiable QP (26s/fold)

**Bottom line:** "Align your estimation objective with your decision objective — and use the simplest method that achieves alignment."

---

## Slide 13: Thank You + Questions

- "Thank you! Questions?"
- Team BrewNet: Aojie Yuan & Haiyue Zhang
- DSO 619 — End-to-End Learning and Optimization

---

## Anticipated Q&A from Prof. Gupta

| Question | Prepared Answer |
|----------|----------------|
| "Why not Bayesian optimization for 12 params?" | Viable but slower; at 385 params, only gradient-based DFL is practical. BO is another path to alignment — consistent with our mechanism claim. |
| "Is 5-12% TE reduction economically significant?" | At K=20, ~35 bps annual TE reduction. Typical ETF fee is 3-10 bps. Meaningful for index mandates with tight TE budgets. |
| "Val-Tuned matches DFL — isn't DFL unnecessary?" | For 1-param models, yes! That's our point. The mechanism is alignment, not DFL. DFL becomes essential at 12+ params (Q3). |
| "Why not just use LW2020?" | LW2020 (0.0293) is better statistics, still MSE-aligned. Task-aware LW2004 (0.0287) beats it with a worse estimator. Better statistics != better decisions. |
| "What about DCC-GARCH?" | Acknowledged limitation. Would expect same pattern: MSE-trained DCC-GARCH < task-aware DCC-GARCH. |
| "Gradient bound assumes fixed selection set?" | Yes, by design. Dynamic selection + DFL hurts (TE = 0.0346, 6x turnover). Validates detached selection. |

---

## Timing Summary

| Slide | Topic | Time | Cumulative |
|-------|-------|------|------------|
| 1 | Title | 0:30 | 0:30 |
| 2 | Problem | 1:30 | 2:00 |
| 3 | Decision Problem + PTO Gap | 2:00 | 4:00 |
| 4 | Sparsity Amplifies Gap | 1:30 | 5:30 |
| 5 | DFL Solution | 1:30 | 7:00 |
| 6 | Algorithm | 1:00 | 8:00 |
| 7 | Setup | 1:00 | 9:00 |
| 8 | Q1: Alignment Helps | 1:30 | 10:30 |
| 9 | Q2: Mechanism Decomposition | 2:00 | 12:30 |
| 10 | Q3: DFL Essential | 1:30 | 14:00 |
| 11 | Robustness | 1:00 | 15:00 |
| 12 | Takeaways | 1:00 | 16:00 |
| 13 | Thank You | — | — |

**Total: ~15 min** (1 min buffer)
