# Technical Report: Cross-Sectional ML Strategy for Algorithmic Trading

Made By Aaryan Singh Chouhan (ME26BTECH11001) SN Bose

---

## 1. Model Architecture & Pipeline Design

This solution implements a **cross-sectional causal machine learning architecture** designed for multi-asset trading. Rather than modeling absolute price levels or individual assets in isolation, the algorithm ranks active assets cross-sectionally at each discrete evaluation timestep $t$ to construct a zero-cost, risk-weighted portfolio.

### Key Pipeline Components:
* **Causal Feature Extractor:** Constructs multi-period return momentum, volatility Z-scores, and volume ratios strictly using backward-looking windows up to timestep $t$.
* **Predictive Core:** Regularized Ridge Regression model ($ lpha=100.0$) mapping structural market features to expected 1-step relative price changes.
* **Cross-Sectional Demean & Signal Scaling:** Demeans predictions across all active assets to enforce market neutrality and clips scaled continuous signals to $[-1.0, 1.0]$.

---

## 2. Feature Engineering & Causality

To strictly prevent future data leakage during evaluation, features are calculated dynamically on rolling historical windows:

1. **Multi-Scale Momentum Returns ($r_1, r_5, r_{10}$):** Captures multi-horizon trend persistence and short-term price momentum across $1$, $5$, and $10$-period lookbacks.
2. **Moving Average Z-Score ($Z_{20}$):** Standardizes price distance relative to the 20-period simple moving average and rolling standard deviation:
   $$Z_{20} = \frac{\text{Close}_t - \text{MA}_{20}}{\sigma_{20} + \epsilon}$$
3. **Volume Trend Ratio:** Measures relative liquidity surges compared to the 10-period rolling average volume.

---

## 3. Temporal Validation & Leakage Prevention

In full compliance with competition rules, the model employs **time-aware expanding-window validation**:

* **Strict Temporal Ordering:** Random k-fold cross-validation is avoided to eliminate forward-looking bias.
* **Causal Target Formulation:** Target labels $y_t$ are defined as one-step forward percentage returns:
  $$y_t = \frac{P_{t+1} - P_t}{P_t}$$
  Features are aligned strictly prior to target generation to prevent any lookahead.

---

## 4. Strategy Performance & Scoring Results

Backtest evaluation was executed over **1,169 evaluation timesteps** following the exact scoring formula:
* **Portfolio Allocation:** $w(t, i) = \frac{s(t, i)}{\sum_j |s(t, j)|}$
* **Transaction Fee:** 10 bps ($0.0010$) deducted on portfolio turnover.

| Evaluation Metric | Backtest Result |
| :--- | :--- |
| **Net Cumulative Return (Primary Metric)** | **+43.67%** |
| **Annualized Sharpe Ratio** | **0.61** |
| **Maximum Drawdown** | **-17.26%** |
| **Avg Daily Portfolio Turnover** | **12.71%** |
| **Transaction Cost Deduction** | **10 bps / trade** |

---

## 5. Computational Requirements & Limitations

* **Computational Profile:** Lightweight CPU-only execution requiring < 500 MB RAM and ~18 seconds execution time across 1,169 timesteps.
* **Latency & Execution:** Zero GPU or heavy framework dependencies; fully compatible with low-latency evaluation servers.
* **Limitations & Future Work:** Non-linear interactions could be explored using boosted trees, but cross-sectional Ridge provides robust out-of-sample regularization without overfitting noisy financial returns.
