---
name: time-series-setup
description: Configure GAFIME Engine v0.4.5 time-series candidate search for temporal/sequential data. Use when the user has temporal data, transaction logs, time-stamped records, or asks things like "set up time series features", "time series feature engineering", "velocity features", "rolling windows", or "regime features".
---

# Time-Series Setup

Guide users through configuring Engine-level time-series candidate search.

## Instructions

1. Ask the user about their data:
   - What is the **group column**? (e.g., `customer_id`, `user_id`, `account_id`)
   - What is the **time column**? (e.g., `date`, `timestamp`, `transaction_date`)
   - What is the **target**? (e.g., `churn`, `fraud`, `default`)
   - What is the **time granularity**? (hourly, daily, weekly, monthly)

2. If they have the data file, run the detector:

   ```bash
   python .claude/skills/time-series-setup/scripts/detect_time_structure.py "<file_path>"
   ```

3. Based on the data's time span and granularity, recommend window sizes:

   | Granularity | Recommended Windows | Rationale |
   |-------------|-------------------|-----------|
   | Hourly | [6, 12, 24, 48, 168] | 6h, 12h, 1d, 2d, 1w |
   | Daily | [7, 14, 30, 60, 90, 180, 360] | 1w, 2w, 1m, 2m, 3m, 6m, 1y |
   | Weekly | [4, 8, 13, 26, 52] | 1m, 2m, 1q, 6m, 1y |
   | Monthly | [3, 6, 12, 24] | 1q, 6m, 1y, 2y |

4. Recommend whether to enable calculus features:
   - **Velocity** (1st derivative): Always useful — captures rate of change
   - **Acceleration** (2nd derivative): Useful for detecting trend reversals
   - **Momentum** (jerk): Useful for volatile financial data
   - **Integral**: Useful for cumulative metrics
   - **Volatility ratios**: Great for financial data
   - **Trend strength**: Good for detecting regime changes

5. Generate the complete setup code:

   ```python
   from gafime import ComputeBudget, EngineConfig, GafimeEngine

   config = EngineConfig(
       backend="auto",
       metric_names=("pearson", "r2"),
       enable_time_series_functions=True,
       time_series_lags=(1, 2, 4, 8, 16),
       time_series_windows=(4, 8, 16, 32),
       budget=ComputeBudget(max_time_series_candidates=100_000),
   )
   report = GafimeEngine(config).analyze(X_train, y_train, feature_names)
   ```

6. Warn about common pitfalls:
   - Windows longer than the data span → produces all NaN
   - Too many windows × features → combinatorial explosion
   - Missing timestamps → Polars will error on sort
   - Non-numeric feature columns → need encoding first
   - If v0.4.x discrete functions are enabled after time-series aggregation,
     thresholds must be fit only on training folds.

7. If the user wants threshold or regime-style time-series signals, recommend
   running `GafimeEngine` with `enable_discrete_functions=True` after
   aggregation. On GPU, use `discrete_mode="soft"`; hard mode is C++ Core only.

## Example

**User says:** "I have daily transaction data per customer, need to predict churn"

**Result:** "For daily data, I recommend windows [7, 14, 30, 60, 90, 180, 360]. Enable calculus features — velocity and acceleration are great for detecting customers who are slowing down before churning. Here's your setup code: ..."
