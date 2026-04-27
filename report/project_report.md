# Task-Aware Covariance Estimation for Sparse Index Tracking

## Project Report & Presentation Guide

**Team BrewNet** | Aojie Yuan · Haiyue Zhang  
CSCI 619: End-to-End Learning and Optimization | Spring 2026 | Prof. Vishal Gupta

---

## Part I: Project Report

---

### 1. Research Question

> **When a sparse portfolio decision uses only a K×K sub-matrix of an N×N covariance estimate, does aligning the estimation objective with the tracking-error objective improve out-of-sample performance — and when is differentiable optimization necessary to achieve this alignment?**

This question connects directly to the core theme of DSO 619: the predict-then-optimize paradigm and when end-to-end learning provides genuine advantages.

---

### 2. Problem & Motivation

#### 2.1 Sparse Index Tracking

Sparse index tracking replicates a benchmark index (e.g., S&P 500) using only K << N stocks. Smaller funds cannot hold all 500 constituents due to transaction costs, management overhead, and custody fees. The goal is to choose K stocks and their weights to minimize **tracking error** — the volatility of the return difference between the portfolio and the index.

**Decision problem (QP):**

Given a predicted covariance matrix Σ̂, solve:

```
min_{w_S}  w_S' Σ̂_SS w_S  -  2 w_S' σ̂_idx,S
s.t.       1'w_S = 1,  w_S ≥ 0
```

where S is the set of K selected stocks, |S| = K.

**Key elements:**
- **Decision variable:** w ∈ R^K (portfolio weights)
- **Objective:** minimize expected squared tracking error
- **Constraints:** fully invested, long-only
- **Uncertainty:** Σ must be estimated from historical return data
- **Timing:** observe features → estimate Σ̂ → solve QP → hold portfolio → observe realized TE

#### 2.2 The Predict-then-Optimize Gap

The standard two-stage approach:
1. **Predict:** Estimate Σ̂ by minimizing prediction error (MSE / Frobenius norm) against realized covariance — over all N² entries
2. **Optimize:** Plug Σ̂ into the QP to get portfolio weights

The structural problem: **the QP only uses the K×K sub-matrix Σ̂_SS**, but MSE distributes estimation effort uniformly across all N² entries. At K=20, N=100:
- QP uses 400 entries (4%)
- MSE optimizes all 10,000 entries
- **96% of estimation effort is wasted on decision-irrelevant entries**

This is a concrete, quantifiable instance of the predict-then-optimize gap studied throughout DSO 619. What makes sparse index tracking special is that **the sparsity constraint creates a hard boundary** — entries outside S×S have literally zero influence on the decision, not diminishing influence.

#### 2.3 Why This Problem for DFL

We chose sparse index tracking for three structural reasons:

1. **Tunable misalignment.** K/N directly controls how much of Σ̂ the decision uses. We can dial K from 5 to 50 and watch the DFL advantage appear, peak, and vanish. No other DFL application offers this clean experimental control.

2. **Hard zero boundary.** In most predict-then-optimize settings, all parameters have some influence on the decision. Here, (N-K)² entries have exactly zero influence. This creates the maximum possible gap between prediction and decision gradients, making the effect clearly measurable.

3. **Clean QP structure.** The optimization is a standard QP — differentiable via KKT conditions (cvxpylayers) with no approximations. This lets us study the DFL effect without confounding from optimization relaxation artifacts.

The result is that sparse index tracking serves as an **ideal experimental testbed** for studying when and why end-to-end training helps in predict-then-optimize pipelines.

---

### 3. Theoretical Foundation

#### 3.1 Gradient Alignment Bound

The QP Jacobian ∂w*/∂Σ̂, derived from KKT conditions, acts as a **sparsity-aware projection**: it zeros out gradient contributions from all entries Σ̂_ij where both i ∉ S and j ∉ S. This means:

- **Task gradient** g_task: supported only on T = {(i,j) : i ∈ S or j ∈ S}, zero elsewhere
- **MSE gradient** g_MSE: distributed across all N² entries

**Proposition 1 (Gradient Alignment Bound):**

Under isotropic MSE gradient energy:

```
cos(g_MSE, g_task)  ≤  √(2K/N − K²/N²)  ≈  √(2K/N)   for K << N
```

| K/N  | Max alignment | Wasted MSE effort | Predicted DFL value |
|------|---------------|-------------------|---------------------|
| 0.05 | 32%           | 68%               | High                |
| 0.10 | 45%           | 55%               | High                |
| 0.20 | 63%           | 37%               | Moderate            |
| 0.50 | 100%          | 0%                | None                |

**Proof sketch:** (1) KKT differentiation shows task gradient zero outside T, (2) Cauchy-Schwarz on restricted inner product, (3) isotropy gives energy ratio |T|/N² = 2K/N − K²/N².

**Why this matters:** This bound makes a **testable prediction** — DFL should help most when K/N is small, and the gain should vanish as K/N → 1. Our experiments precisely confirm this.

#### 3.2 QP Jacobian as Natural Attention

When backpropagating through the QP, the KKT Jacobian automatically:
- Concentrates 87.5% of gradient energy on the K×K selected-stock block
- Only 12.5% on the cross block S × S̄
- Zero outside

Under MSE, only K²/N² = 4.3% of gradient energy falls in S×S. The **20× concentration factor** confirms the QP Jacobian acts as an optimal information filter, directing learning exclusively to decision-relevant entries.

---

### 4. Solution: Two Paths to Alignment

We identify that the mechanism behind DFL's improvement is **objective alignment** — training the covariance model to minimize tracking error instead of prediction error. There are two paths to achieve this:

#### Path 1: Grid Search (Val-Tuned)

```
For each α ∈ {0.01, 0.02, ..., 1.0}:
    Σ̂(α) = (1−α)S + α·μ·I          # Ledoit-Wolf shrinkage
    w*(α) = solve_QP(Σ̂(α))          # standard QP (no diff)
    score(α) = realized_TE(w*, validation_returns)
Pick α* = argmin score(α)
```

- No differentiable optimization needed
- Works perfectly for 1-parameter models
- O(grid_size) QP solves — fast and simple
- **This IS task-aware estimation, achieved without DFL machinery**

#### Path 2: Differentiable QP (DFL / End-to-End)

```
θ ← initialize model parameters
For each epoch:
    For each rebalancing date t:
        Σ̂ = model(x_t; θ)
        w* = DifferentiableQP(Σ̂)     # cvxpylayers
        L = realized_TE(w*, r_actual)
        θ ← θ − η · ∂L/∂θ           # gradient through KKT
```

- Scales to arbitrary parameter count via backpropagation
- Can train conditional, per-observation covariance models
- O(params) per gradient step — efficient for high-dimensional θ

#### The Key Distinction

The contribution is NOT "DFL beats two-stage." It is:
1. **Objective alignment is the mechanism** — any method achieving alignment (grid search, SPO+, DFL) closes the gap
2. **DFL is a scalable means** — it becomes essential when the model has too many parameters for exhaustive search
3. The two components are **empirically separable** — our experiments cleanly decompose them

---

### 5. Experimental Design

#### 5.1 Data

| Item | Detail |
|------|--------|
| Universe | S&P 500, top 100 by market cap (N=97 after filtering) |
| Period | 2006–2025, daily returns (19 years) |
| Features | 21-day realized vol, 63-day momentum, GICS sector (one-hot), market-cap quintile |
| Benchmark | Cap-weighted index of same 100 stocks |

#### 5.2 Rolling Window Protocol

- 3-year train / 6-month validation / 1-year test
- Roll forward 1 year → **9 non-overlapping folds**
- Spans: 2008 crisis, 2015 China shock, 2020 COVID, 2022 rate cycle, 2024–25 AI rally

#### 5.3 Models (each trained under MSE and Task loss)

| Model | Params | Formula | Purpose |
|-------|--------|---------|---------|
| Shrinkage | 1 | (1−α)S + αμI | Minimal DFL test |
| Factor | O(NK_f) | BFB' + D | Controllable capacity |
| Neural | O(Nr) | MLP → LL' + D | Maximum flexibility |
| Structured shrinkage | 12 | (1−α)S + αT (per-sector) | Medium complexity |
| Conditional shrinkage | 385 | MLP(market_features) → α_t | Regime-adaptive |

#### 5.4 Baselines

1. **Naive market-cap** — top-K by index weight, no optimization
2. **Ledoit-Wolf 2004 + QP** — analytical linear shrinkage
3. **Ledoit-Wolf 2020 + QP** — analytical nonlinear shrinkage (statistical SOTA)
4. **Val-Tuned** — grid search 50 α values on validation TE (no diff QP)
5. **SPO+** — Elmachtoub & Grigas decision regret surrogate
6. **Random search** — 200–500 random configurations for multi-param models
7. **Direct weight MLP** — bypass QP, predict weights directly

#### 5.5 Three Experimental Questions

| # | Question | What it tests |
|---|----------|---------------|
| Q1 | Does alignment improve tracking? | Main effect of objective alignment |
| Q2 | Is DFL necessary, or does grid search suffice? | Mechanism decomposition |
| Q3 | When does DFL become essential? | Boundary condition for DFL's unique value |

#### 5.6 Statistics

- Wilcoxon signed-rank test (non-parametric, appropriate for 9 folds)
- 10,000-sample bootstrap confidence intervals
- All results out-of-sample

---

### 6. Results

#### 6.1 Q1: Alignment Helps — Sparsity is the Driver

**Neural model, task-aware vs. two-stage across K:**

| K  | Two-Stage | Task-Aware | Gain    | 95% CI           | p-value | Wins |
|----|-----------|------------|---------|------------------|---------|------|
| 5  | 0.0877    | 0.0794     | −9.5%   | [−12.7, −6.4]    | 0.002   | 9/9  |
| 10 | 0.0530    | 0.0495     | −6.6%   | [−10.4, −3.4]    | 0.002   | 9/9  |
| 20 | 0.0326    | 0.0287     | −11.9%  | [−18.1, −5.4]    | 0.002   | 9/9  |
| 30 | 0.0212    | 0.0202     | −4.5%   | [−9.4, −0.7]     | 0.002   | 8/9  |
| 50 | 0.0100    | 0.0100     | −0.2%   | [−0.5, −0.0]     | 0.102   | 5/9  |

**Findings:**
- K ≤ 30: statistically significant improvement (p = 0.002), 8–9/9 folds
- K = 50: no significant effect (p = 0.102, 5/9 wins)
- **Pattern matches gradient alignment bound prediction:** large gains at low K/N, vanishing at K/N = 0.5

**Additional evidence:**
- Factor model K×K_f interaction grid (288 runs): gain depends only on K, not on model capacity K_f
- Rolling 63-day TE: task-aware lower in 86% of windows

#### 6.2 Q2: Mechanism Decomposition — Alignment, Not Machinery

**Shrinkage model (1 param), K=20:**

| Method | TE | Needs diff QP? | How alignment is achieved |
|--------|-----|----------------|--------------------------|
| Ledoit-Wolf (analytical) | 0.0325 | No | No alignment (MSE-optimal α) |
| LW 2020 (nonlinear) | 0.0293 | No | Better estimation, still MSE |
| **Val-Tuned grid search** | **0.0292** | **No** | **Exhaustive search on val TE** |
| SPO+ surrogate | 0.0290 | Yes | Convex surrogate for regret |
| DFL (diff QP) | 0.0287 | Yes | Backprop through KKT |

**Key finding:** Val-Tuned, SPO+, and DFL all converge to TE ≈ 0.029. For a 1-parameter model, the optimization pathway doesn't matter — only alignment does.

**Why this is a strength, not a weakness:**
- It **decomposes** the DFL effect: alignment (mechanism) vs. diff optimization (means)
- It identifies the **sufficient condition** for improvement: any method achieving alignment works
- It sets up the next question: when does DFL become essential?

**Comprehensive comparison at K=20:**

| Method | TE | Turnover | Notes |
|--------|-----|----------|-------|
| Direct weight MLP | 0.0622 | 0.006 | No QP structure |
| Naive market-cap | 0.0462 | 0.000 | No optimization |
| LW 2004 + QP | 0.0325 | 0.000 | Linear shrinkage |
| LW 2020 + QP | 0.0293 | 0.000 | Nonlinear shrinkage (stat SOTA) |
| Factor (MSE) + QP | 0.0325 | 0.001 | Two-stage |
| Neural (MSE) + QP | 0.0326 | 0.003 | Two-stage |
| Val-Tuned Shrinkage | 0.0292 | 0.068 | Grid search α |
| SPO+ Surrogate | 0.0290 | 0.064 | Decision regret |
| Shrinkage + Task | 0.0287 | 0.046 | 1 param, diff QP |
| Factor + Task | 0.0286 | 0.051 | Task-aware |
| Neural + Task | 0.0287 | 0.030 | Most complex |
| Struct. Shrinkage + Task | **0.0285** | 0.052 | 12 params, per-sector |

**Observation:** All MSE-trained models saturate at TE ≈ 0.0325 regardless of complexity. All task-aware methods break through to TE ≈ 0.029. Better statistical estimation (LW2020 = 0.0293) narrows the gap but cannot close it — the residual is due to objective misalignment, not estimation quality.

#### 6.3 Q3: DFL is Essential for Richer Models

| Model | Params | Grid/RS | DFL | DFL unique? |
|-------|--------|---------|-----|-------------|
| Shrinkage | 1 | 0.0292 (grid) | 0.0290 | No — tied |
| Structured (per-sector) | 12 | 0.0295 (RS-500) | **0.0285** (p=0.006) | **Yes** |
| Conditional (MLP→α) | 385 | 0.0297 (RS-200) | 0.0287 (p=0.014 at K=10) | Yes (at K=10) |
| Neural | ~5000 | 0.0326 (MSE only) | 0.0287 | Yes |

**Key findings:**
- 1 param: grid search = DFL
- 12 params: DFL beats random search by 3.4%, p=0.006, 8/9 folds
- 385 params: DFL beats RS, but shows overfitting instability at K=20
- **Scaling to N=478:** Structured DFL achieves p=0.031 — significant at full S&P scale

**Practical sweet spot:** 12-parameter structured shrinkage — enough capacity to capture sector structure, stable enough to train reliably.

---

### 7. Robustness

#### 7.1 Turnover

| γ (turnover penalty) | TE | Turnover | vs Two-Stage |
|----------------------|-----|----------|--------------|
| Two-stage (no DFL) | 0.0326 | 0.003 | — |
| DFL, γ=0 | 0.0287 | 0.030 | −11.9% |
| **DFL, γ=0.01** | **0.0299** | **0.003** | **−8.2%** |
| DFL, γ=0.1 | 0.0319 | 0.000 | −2.2% |
| DFL, γ=1.0 | 0.0325 | 0.000 | −0.2% |

At γ=0.01: turnover matches MSE level, still 8.2% TE improvement. DFL Pareto-dominates MSE at every turnover level.

#### 7.2 Transaction Costs

- Low TC (10 bps): Val-tuned slightly better (grid search optimizes conservatively)
- High TC (100 bps): DFL wins — grid search over-shrinks to suppress turnover, DFL adapts per observation

#### 7.3 Market Regimes

| Regime | Two-Stage | Task-Aware | Gain |
|--------|-----------|------------|------|
| High-volatility | 0.0343 | 0.0300 | −12.7% |
| Low-volatility | 0.0308 | 0.0276 | −10.5% |
| 2021 (inflation) | 0.0394 | 0.0302 | −23.5% |
| 2022 (rate hikes) | 0.0412 | 0.0332 | −19.4% |
| 2017 (calm) | 0.0275 | 0.0270 | −2.0% |

Gains in both regimes, largest during volatile shifts (when estimation errors are largest).

#### 7.4 Risk Metrics

| Metric | Two-Stage | Task-Aware | Improvement |
|--------|-----------|------------|-------------|
| Annualized TE | 0.0326 | 0.0287 | −11.9% |
| Worst-fold TE | 0.0476 | 0.0348 | −26.8% |
| TE std across folds | 0.0062 | 0.0025 | −59.4% |
| CVaR 5% (daily) | 0.0047 | 0.0041 | −11.5% |

#### 7.5 Portfolio Characteristics

| Metric | Two-Stage | Task-Aware |
|--------|-----------|------------|
| HHI | 0.067 | 0.068 |
| Effective N | 15.0 | 14.7 |
| Max position | 17.3% | 17.3% |

Identical concentration — improvement is pure estimation quality, not portfolio restructuring.

#### 7.6 Other Robustness Checks

- **Position limits (UCITS 5/10/40):** gain preserved at w_max = 10% and 20%
- **Lookback windows (21–126 days):** gain stable at 10–12%
- **Rebalancing frequency (weekly/biweekly/monthly):** gain stable at ~12%
- **Cross-benchmark (Russell 2000, Tech sector):** 1-param findings replicate
- **Factor exposure:** no hidden factor bets (all |t| < 0.8, R² = 0.0003)
- **Direct weight prediction:** TE = 0.0622 — QP structure is essential inductive bias

---

### 8. Computational Cost

| Method | Time (per fold) | Notes |
|--------|-----------------|-------|
| Ledoit-Wolf + QP | < 0.1s | Closed-form |
| Grid search (50 α) | 0.6s | Forward QP only |
| Neural two-stage | 38.9s | 47 epochs (early stop) |
| Neural task-aware | 26.3s | 15 epochs (early stop) |

Task-aware converges faster (15 vs 47 epochs) — the task loss provides a more direct training signal.

---

### 9. Contributions

**Contribution 1: Structural Principle**
Sparsity in the downstream optimization amplifies the predict-then-optimize gap. The ratio K/N is a sufficient statistic governing whether task-aware estimation helps. Gradient alignment bound cos ≤ √(2K/N) provides a quantitative predictor, validated empirically.

**Contribution 2: Mechanism Decomposition**
The improvement comes from objective alignment — any method achieving alignment (grid search, SPO+, DFL) closes the gap equally for simple models. This cleanly separates the *what* (alignment) from the *how* (optimization pathway).

**Contribution 3: DFL's Unique Role**
Differentiable optimization becomes essential as model complexity grows. It is the only scalable path to alignment for structured/conditional covariance models, achieving significant gains at N=478 (p=0.031).

**Practical decision rule:**

```
if K/N > 0.3:      → just add more stocks, alignment gains minimal
if params ≤ 1:      → grid search over validation TE (0.6s, done)
if params 2–10:     → grid search may work, DFL is better
if params > 10:     → use DFL (differentiable QP) — only viable path
```

---

### 10. Limitations

1. **Survivorship bias:** Universe is current top-100, not point-in-time constituents. Affects absolute TE but not relative comparison.
2. **Transaction cost model:** Flat per-trade cost; real costs depend on trade size, bid-ask spread.
3. **Scope:** Richer models validated only on S&P 500. No comparison with DCC-GARCH.
4. **Conditional model instability:** 385-param model shows overfitting at K=20 (inconsistent across folds). 12-param structured model is the practical sweet spot.

---

### 11. Code & Reproducibility

- Repository: `github.com/[repo]`
- Key files:
  - `src/data/loader.py` — S&P 500 data pipeline
  - `src/data/features.py` — feature engineering + covariance estimators (incl. LW2020)
  - `src/models/shrinkage.py` — shrinkage/structured/conditional models
  - `src/models/neural_cov.py` — neural covariance model
  - `src/optimization/solver.py` — QP solver (cvxpylayers + SCS)
  - `src/training/trainer.py` — two-stage and task-aware training loops
  - `scripts/run_main.py` — main experiment runner
  - `scripts/run_nlshrink_baseline.py` — LW2020 baseline

---
---

## Part II: Presentation Flow (15 minutes)

---

### Slide-by-Slide Guide

#### Slide 1: Title (30s)
**Task-Aware Covariance Estimation for Sparse Index Tracking**
- Team BrewNet: Aojie Yuan, Haiyue Zhang
- CSCI 619 | Spring 2026

---

#### Slide 2: The Problem (1.5 min)
**Hook:** "Can you replicate the S&P 500 with only 20 stocks?"

Content:
- Sparse index tracking: K << N stocks, optimize weights, minimize tracking error
- Why sparse? Transaction costs, management overhead, custody fees
- Key metric: TE = std(w'r − r^idx) × √252 — how much does our portfolio deviate from the index?
- Visual: S&P 500 (500 stocks) → Our Portfolio (20 stocks), gap = TE

Transition: "The standard approach estimates covariance first, then optimizes. But this two-stage separation creates a structural problem."

---

#### Slide 3: The Decision Problem (1.5 min)
**Formal QP + Identify the gap**

Left half — QP formulation:
```
min  w'Σ̂w − 2w'σ̂_idx
s.t. 1'w = 1, w ≥ 0, |supp(w)| ≤ K
```
- Decision variable: w (K-dimensional weights)
- Unknown: Σ̂ estimated from data
- QP uses only Σ̂_SS — the K×K sub-matrix

Right half — two-stage pipeline diagram:
```
Returns → Cov Model → Σ̂ (N×N) → QP → w*
              ↑
         MSE Loss (all N² entries)
```

Key line: "The QP uses 4% of the covariance. MSE optimizes 100%. This is the predict-then-optimize gap."

---

#### Slide 4: Why Sparsity Amplifies the Gap (1.5 min)
**Matrix visual + gradient bound**

Visual: N×N matrix with K×K block highlighted green, rest gray.
- "96% of estimation effort is wasted on entries the decision ignores"

Gradient Alignment Bound: cos(g_MSE, g_task) ≤ √(2K/N)
- Table: K/N = 0.05 → 32% alignment | K/N = 0.5 → 100%
- "We can predict from K/N alone whether task-aware estimation is worth it"
- "The QP Jacobian acts as a natural attention mechanism — it tells the model which covariance entries matter for the decision"

---

#### Slide 5: Our Solution (1.5 min)
**Two paths to alignment**

DFL pipeline diagram:
```
Returns → Cov Model → Σ̂ → Diff QP → w* → Realized TE
              ↑                                |
              └───── gradient through KKT ─────┘
```

Key message: "We train the covariance model to produce Σ̂ that leads to low tracking error — not low prediction error."

Three models tested (each under MSE and Task loss):
| Shrinkage (1p) | Factor (O(NK)) | Neural (MLP) |
Quick mention of cvxpylayers + KKT differentiation.

---

#### Slide 6: Experimental Setup (1 min)
**Compact overview**

- Data: S&P 500 top 100, 2006–2025
- Protocol: rolling 3yr/6mo/1yr → 9 folds
- Sparsity: K ∈ {5, 10, 20, 30, 50}
- Baselines: Naive, LW2004, LW2020, Val-Tuned, SPO+
- Statistics: Wilcoxon signed-rank, all out-of-sample

"Three questions drive our experiments."

---

#### Slide 7: Result 1 — Alignment Helps (1.5 min)
**Sparsity frontier figure + table**

Show fig1 + key numbers:
- K=5: −9.5%, p=0.002, 9/9 wins
- K=20: −11.9%, p=0.002, 9/9 wins
- K=50: −0.2%, n.s.

"The gradient bound correctly predicts the boundary. At K/N = 0.5, MSE and task loss are already aligned — there's nothing for DFL to fix."

---

#### Slide 8: Key Finding — Decomposing the Mechanism (2 min)
**This is the intellectual core of the presentation**

Table comparing Val-Tuned / SPO+ / DFL at K=20:
- All three ≈ 0.029. LW2004 = 0.0325. LW2020 = 0.0293.

"Three methods, three different optimization approaches, same result. The mechanism is objective alignment, not differentiable optimization."

"But this raises the question: if grid search works, when do you actually need DFL?"

---

#### Slide 9: When DFL is Essential — Richer Models (1.5 min)
**The resolution**

Table: Shrinkage (1p) grid ≈ DFL | Structured (12p) only DFL works | Conditional (385p) only DFL works

"1 parameter: you can try every value. 12 parameters: 50¹² = impossible. DFL scales linearly via backpropagation."

"At N=478 (full S&P), structured DFL achieves p=0.031."

---

#### Slide 10: Robustness (1.5 min)
**Three panels, one slide**

Panel A — Turnover: γ=0.01 → turnover = MSE level, still 8.2% gain
Panel B — Regimes: gains in both high-vol (−12.7%) and low-vol (−10.5%)
Panel C — Concentration: HHI identical, UCITS compliant

"Task-aware estimation is a drop-in improvement — same portfolio structure, better tracking."

---

#### Slide 11: Takeaways (1.5 min)
**Three contributions**

1. **Structural principle:** Sparsity amplifies the predict-then-optimize gap. K/N is the sufficient statistic. Gradient bound predicts when alignment helps.

2. **Mechanism decomposition:** Gain = objective alignment. For simple models, grid search suffices. This separates the *what* from the *how*.

3. **DFL's unique role:** Scales to rich models (12+ params) that grid search cannot train. Only viable path to alignment at scale.

**Practical rule:**
- K/N > 0.3 → add more stocks
- 1 param → grid search
- 12+ params → use DFL

Bottom line: **"Align your estimation with your decision — and use the simplest method that achieves alignment."**

---

#### Slide 12: Thank You + Questions

---

### Presentation Tips

1. **Pacing:** ~1 slide/min, 12 slides for 15 min. Don't rush.
2. **The "Key Finding" slide (Slide 8) should take the most time.** This is where you demonstrate insight — the grid search = DFL result is counterintuitive and worth explaining carefully.
3. **Anticipate questions from Vishal:**
   - "Why not Bayesian optimization for 12 params?" → Could work, but DFL is more scalable and validated.
   - "Is 5-12% economically significant?" → 35 bps at K=20, meaningful for index funds (typical ETF fee is 3-10 bps).
   - "What about DCC-GARCH?" → Valid baseline we didn't test. Acknowledged as limitation.
4. **Don't apologize for the grid search result.** Frame it as a strength — it's a clean decomposition of the DFL effect.
5. **Use Vishal's language:** "predict-then-optimize," "decision-focused learning," "objective alignment," "downstream optimization." These are his course concepts.

---

### Task Division Suggestion

| Task | Owner | Deadline |
|------|-------|----------|
| Slides 1–5 (problem, theory, solution) | Justin | Apr 26 |
| Slides 6–9 (experiments, results) | Haiyue | Apr 26 |
| Slides 10–12 (robustness, takeaways) | Justin | Apr 26 |
| Slide design & polish (Gamma/Canva) | Both | Apr 26 evening |
| Dry run (timed, with transitions) | Both | Apr 27 morning |
| Code + README cleanup for submission | Both | Apr 27 noon |
| Submit slides + code by 2:00 PM | — | Apr 27 |
