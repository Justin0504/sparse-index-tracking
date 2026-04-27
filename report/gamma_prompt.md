Create a 13-slide academic research presentation for a graduate DFL course at USC Marshall. 15 minutes. Professor is an expert in optimization and quantitative finance.

RULES:
1. EXACTLY 13 slides. Each "--- SLIDE N ---" = one slide. Do NOT merge or skip.
2. Use ONLY the numbers below. Do NOT invent statistics. Copy numbers exactly.
3. No stock photos or AI images. Where I write [IMAGE], leave a large dashed-border empty area with the label.
4. Style: dark navy + white + light gray, sans-serif, clean academic. No gradients or decorations.
5. Formulas should be large and centered. Each slide: bold title + 3-5 concise bullets.

=== BEGIN SLIDES ===

--- SLIDE 1 ---
Title: Task-Aware Covariance Estimation for Sparse Index Tracking
Centered title slide.
DSO 619 — End-to-End Learning and Optimization
Team BrewNet: Aojie Yuan & Haiyue Zhang
USC Marshall School of Business, Spring 2026

--- SLIDE 2 ---
Title: Can You Replicate the S&P 500 with Only 20 Stocks?
Left text, right image.
- Index funds hold 500 stocks — expensive for small funds and ETF issuers
- Sparse index tracking: select K << N stocks, minimize tracking error
- Tracking error = annualized std of return difference vs index
- Real-world use: ETFs, pension funds, UCITS compliance
- Our result: 20 stocks track S&P 500 for 9 years, TE under 3%
[IMAGE: "fig12 — Cumulative wealth: 20 stocks vs S&P 500, 2016-2025"]
Bottom italic: "But how do we choose which 20 stocks and set the weights?"

--- SLIDE 3 ---
Title: The Standard Approach and Its Flaw
Large centered formula:
min_w (w - w_idx)^T Sigma (w - w_idx)  s.t. ||w||_0 <= K, w >= 0, 1^T w = 1
- Decision variable: portfolio weights w
- Uncertain parameter: covariance matrix Sigma (estimated from data)
- Two-Stage approach: (1) Estimate Sigma via MSE, (2) Plug into QP solver
- Problem: no feedback from decision quality to estimation model
Callout box: "QP uses only K^2 = 400 entries. MSE optimizes all N^2 = 9,409 equally — 95.7% of effort is wasted."

--- SLIDE 4 ---
Title: Why Sparsity Amplifies the Gap
Large centered formula:
Proposition 1: cos(g_MSE, g_task) <= sqrt(2K/N)
Table:
K=5, N=97: bound=0.32 — severe misalignment
K=20, N=97: bound=0.64 — moderate, significant gain possible
K=50, N=97: bound>=1.0 — well-aligned, MSE near-optimal
Callout: "Testable prediction: DFL helps most at low K/N, not at all at K/N > 0.5."
[IMAGE: "fig14 — Covariance attention heatmap: MSE (uniform) vs Task (K x K block) vs Wasted effort"]

--- SLIDE 5 ---
Title: Decision-Focused Learning — Train for What Matters
[IMAGE: "DFL Pipeline — Forward: Features -> Covariance Model -> Differentiable QP -> Weights. Backward: Task Loss gradient flows through QP via KKT back to model."]
- Replace MSE with downstream tracking error as training loss
- Differentiate through QP via KKT conditions (cvxpylayers)
- Stock selection is detached — only weight optimization is differentiable
- QP Jacobian = natural attention: zeroes gradients for unselected stocks
Two paths: Grid Search (1-param, 0.6s) vs Differentiable QP (12+ params, 26s/fold)

--- SLIDE 6 ---
Title: DFL Training Algorithm
Formatted pseudocode block:
Input: returns R, index weights w_idx, sparsity K
for each epoch:
  for each rebalancing date t:
    1. x_t = features(R_history)
    2. Sigma_t = f(x_t; theta)          // predict covariance
    3. S = TopK(w_idx, K)               // select stocks (detached)
    4. w* = QP_solve(Sigma[S,S])        // differentiable QP
    5. L = (252/H) sum((w*'r - r_idx)^2) // realized tracking error
    6. dw*/dSigma = KKT_differentiate   // KEY STEP
    7. theta -= lr * dL/dtheta          // chain rule through QP
  early stop if val_TE not improved for 10 epochs
Key insight: "Step 6: KKT conditions give dw*/dSigma analytically. Unselected stocks automatically receive zero gradient."

--- SLIDE 7 ---
Title: Experimental Setup
- Universe: S&P 500 top 100 by market cap (N=97 after filtering)
- Period: 2006-2025 (19 years, 4,800+ trading days)
- Protocol: 3yr train / 6mo val / 1yr test, 9 rolling folds
- Sparsity: K in {5, 10, 20, 30, 50}
- Baselines: LW 2004, LW 2020, Val-Tuned, SPO+, Random Search
- Statistics: Wilcoxon signed-rank test (9 folds)
Bottom italic: "Three questions drive our experiments."

--- SLIDE 8 ---
Title: Q1 — Alignment Helps, Sparsity Drives the Gain
Table (Neural model):
K=5:  Two-Stage=0.0877, Task-Aware=0.0794, Gain=-9.5%,  p=0.002, 9/9
K=10: Two-Stage=0.0530, Task-Aware=0.0495, Gain=-6.6%,  p=0.002, 9/9
K=20: Two-Stage=0.0326, Task-Aware=0.0287, Gain=-11.9%, p=0.002, 9/9
K=30: Two-Stage=0.0212, Task-Aware=0.0202, Gain=-4.5%,  p=0.002, 8/9
K=50: Two-Stage=0.0100, Task-Aware=0.0100, Gain=-0.2%,  p=0.102, 5/9
- Significant at K<=30, vanishes at K=50 — matches gradient bound prediction exactly
- Driver is K/N ratio, not model complexity
[IMAGE: "fig1 — Sparsity frontier: TE vs K"]

--- SLIDE 9 ---
Title: Q2 — The Mechanism is Objective Alignment
Table at K=20:
LW+MSE (baseline):    0.0325
LW+Val-Tuned (grid):  0.0292  (-10.2%)
LW+SPO+:              0.0290  (-10.8%)
LW+Task (DFL):        0.0287  (-11.7%)
LW 2020 (nonlinear):  0.0293  (-9.8%)
Neural+MSE:           0.0326  (+0.3%)
Neural+Task (DFL):    0.0287  (-11.7%)
- Grid search (0.6s) matches DFL for 1-param models — DFL not always necessary
- All MSE models saturate at ~0.0325 regardless of complexity — wrong objective is bottleneck
- LW 2020 = better statistics, not alignment. Better estimation != better decisions
[IMAGE: "fig13 — Method comparison bar chart at K=20"]

--- SLIDE 10 ---
Title: Q3 — DFL Essential for Rich Models
Table:
K=10: Cond.Shrinkage+MSE=0.0494, RandomSearch(200)=0.0496, Cond.Shrinkage+DFL=0.0490 (p=0.014, 7/9)
K=20: Struct.Shrinkage+MSE=0.0287, RandomSearch(500)=0.0295, Struct.Shrinkage+DFL=0.0285 (p=0.006, 8/9)
- Random search cannot optimize 12+ parameters effectively
- DFL: p=0.006, 8/9 folds at K=20
- Scales to full S&P 500 (N=478): p=0.031
- Sweet spot: 12-param structured model

--- SLIDE 11 ---
Title: Robustness — Gains Hold Everywhere
Three columns:
Turnover: gamma=0.01 drops turnover to baseline, still -8.2% TE. DFL Pareto-dominates.
Regimes: High-vol -12.7%, Low-vol -10.5%, 2021 inflation -23.5%, 2022 rate hikes -19.4%
Risk: HHI identical (0.067 vs 0.068). Worst-fold TE -26.8%. TE variability -59.4%.
[IMAGE: "fig2 Turnover frontier | fig9 Regime analysis | fig3 TE boxplot"]

--- SLIDE 12 ---
Title: Three Takeaways
Three numbered items:
1. Structural Principle: Sparsity amplifies the PTO gap. cos<=sqrt(2K/N) predicts when alignment matters.
2. Mechanism: The gain is from objective alignment. For 1-param models, grid search suffices.
3. DFL's Role: For 12+ param models, differentiable QP is the only viable training path.
Decision Rule box:
- K/N > 0.3: minimal gain
- 1-param: grid search (0.6s)
- 12+ params: DFL (26s/fold)
Bold bottom line: "Align your estimation objective with your decision objective."

--- SLIDE 13 ---
Title: Thank You & Questions
Centered, minimal.
"Thank you! We welcome your questions."
Team BrewNet: Aojie Yuan & Haiyue Zhang
DSO 619, USC Marshall School of Business

=== END SLIDES ===
