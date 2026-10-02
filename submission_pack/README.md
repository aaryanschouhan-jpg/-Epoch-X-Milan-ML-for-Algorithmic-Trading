# [Epoch X Milan] ML for Algorithmic Trading

## Strategy Overview
Cross-sectional Ridge Regression model ($\\alpha=100.0$) utilizing multi-period momentum returns (1, 5, and 10 periods), moving average Z-scores (20-period), volume trend ratios, and cross-sectional signal normalization.

## Folder Structure
- `src/model.py`: Core Model API class implementing `.train()` and `.predict()`.
- `test_backtest.py`: Backtest execution and evaluation pipeline.
- `requirements.txt`: Python package dependencies.

## Key Backtest Metrics (1,169 Timesteps)
- **Net Cumulative Return:** +43.67%
- **Annualized Sharpe Ratio:** 0.61
- **Maximum Drawdown:** -17.26%
- **Avg Daily Turnover:** 12.71%
