---
name: time-series-setup
description: Configure GAFIME's TimeSeriesPreprocessor for time-series feature engineering. Use when the user has temporal/sequential data, transaction logs, time-stamped records, or asks things like "set up time series features", "configure preprocessor", "I have transaction data", "time series feature engineering", "calculus features", "velocity features", "rolling windows", or "aggregate to entity level".
---

# Time-Series Setup

Guide users through configuring `TimeSeriesPreprocessor` for their temporal data.

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
   from gafime.preprocessors import TimeSeriesPreprocessor

   tsp = TimeSeriesPreprocessor(
       group_col='<detected>',
       time_col='<detected>',
       windows=<recommended>,
       enable_calculus=True,
   )
   df_processed = tsp.transform(raw_df)
   df_features = tsp.aggregate_to_entity(df_processed, target_df, '<target>')
   ```

6. Warn about common pitfalls:
   - Windows longer than the data span → produces all NaN
   - Too many windows × features → combinatorial explosion
   - Missing timestamps → Polars will error on sort
   - Non-numeric feature columns → need encoding first

## Example

**User says:** "I have daily transaction data per customer, need to predict churn"

**Result:** "For daily data, I recommend windows [7, 14, 30, 60, 90, 180, 360]. Enable calculus features — velocity and acceleration are great for detecting customers who are slowing down before churning. Here's your setup code: ..."
