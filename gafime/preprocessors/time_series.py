"""
GAFIME Time Series Preprocessor - Full Calculus Edition

Creates continuity-aware features using Polars (fast, vectorized):

BASIC FEATURES:
- Lags: f[n-k] as separate columns
- Differentials: f[n] - f[n-k] for various k
- Rolling statistics: mean, std, sum, min, max over windows

CALCULUS FEATURES (Advanced):
- Velocity: First derivative df/dt = (f_new - f_old) / Δt
- Acceleration: Second derivative d²f/dt² = (v_new - v_old) / Δt
- Momentum: Rate of acceleration change (jerk)
- Integral: Cumulative sum (discrete integration)
- Limit Extrapolation: Predict future from current trend
- Volatility Ratios: Short-term vs long-term variance
- Trend Strength: Recent vs historical average ratios

These become regular columns that GAFIME kernel can process
with its existing operations (mul, add, sub, div).

Usage:
    from gafime.preprocessors.time_series import TimeSeriesPreprocessor
    
    tsp = TimeSeriesPreprocessor(
        group_col='cust_id',
        time_col='date',
        windows=[7, 14, 30, 60, 90, 180, 360],
        enable_calculus=True  # Enable advanced calculus features
    )
    
    df_processed = tsp.transform(df)
    df_aggregated = tsp.aggregate_to_entity(df_processed)
"""

import polars as pl
import numpy as np
from typing import List, Optional, Union
from dataclasses import dataclass, field


@dataclass
class TimeSeriesConfig:
    """Configuration for time series preprocessing."""
    group_col: str = 'cust_id'      # Entity column (customer, user, etc.)
    time_col: str = 'date'          # Time column
    
    # Time windows for aggregation (days before reference date)
    windows: List[int] = field(default_factory=lambda: [7, 14, 30, 60, 90, 180, 360])
    
    # Lag features: f[n-k] for each k
    lags: List[int] = field(default_factory=lambda: [1, 7, 14, 30])
    
    # Differential features: f[n] - f[n-k] for each k
    differentials: List[int] = field(default_factory=lambda: [1, 7, 14, 30])
    
    # Rolling window statistics
    rolling_windows: List[int] = field(default_factory=lambda: [7, 14, 30, 60, 90])
    
    # Enable advanced calculus features
    enable_calculus: bool = True
    
    # Which columns to process (None = all numeric)
    feature_cols: Optional[List[str]] = None
    
    # Max columns to process (for speed)
    max_cols: int = 10


class TimeSeriesPreprocessor:
    """
    Creates continuity-aware features for GAFIME.
    
    Separates time-series logic from GAFIME kernel:
    - This layer: creates lags, differentials, calculus features using Polars
    - GAFIME kernel: finds interactions between these features
    
    Full calculus support:
    - Velocity (first derivative)
    - Acceleration (second derivative)
    - Momentum (jerk)
    - Integral (cumulative)
    - Limit extrapolation
    - Volatility ratios
    - Trend strength
    """
    
    def __init__(self, config: TimeSeriesConfig = None, **kwargs):
        if config:
            self.config = config
        else:
            self.config = TimeSeriesConfig(**kwargs)
    
    def transform(self, df: Union[pl.DataFrame, 'pd.DataFrame']) -> pl.DataFrame:
        """
        Transform raw time series into GAFIME-ready features.
        
        Args:
            df: DataFrame with entity, time, and feature columns
        
        Returns:
            DataFrame with original + derived features
        """
        # Convert to Polars if pandas
        if hasattr(df, 'to_pandas'):  # Already polars
            lf = df.lazy()
        else:
            lf = pl.from_pandas(df).lazy()
        
        # Get feature columns
        if self.config.feature_cols:
            feature_cols = self.config.feature_cols
        else:
            # All numeric columns except group and time
            exclude_cols = {self.config.group_col, self.config.time_col}
            all_cols = lf.collect().columns
            feature_cols = [c for c in all_cols 
                           if c not in exclude_cols 
                           and lf.collect()[c].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]]
        
        # Limit columns for speed
        feature_cols = feature_cols[:self.config.max_cols]
        
        # Sort by group and time
        lf = lf.sort([self.config.group_col, self.config.time_col])
        
        # Create derived features
        derived_exprs = []
        
        for col in feature_cols:
            # 1. Lag features
            for lag in self.config.lags:
                derived_exprs.append(
                    pl.col(col).shift(lag).over(self.config.group_col).alias(f"{col}_lag{lag}")
                )
            
            # 2. Differential features (discrete derivative)
            for diff in self.config.differentials:
                derived_exprs.append(
                    (pl.col(col) - pl.col(col).shift(diff).over(self.config.group_col))
                    .alias(f"{col}_diff{diff}")
                )
            
            # 3. Rolling statistics
            for window in self.config.rolling_windows:
                derived_exprs.append(
                    pl.col(col).rolling_mean(window).over(self.config.group_col)
                    .alias(f"{col}_rmean{window}")
                )
                derived_exprs.append(
                    pl.col(col).rolling_std(window).over(self.config.group_col)
                    .alias(f"{col}_rstd{window}")
                )
        
        # Apply all expressions
        result = lf.with_columns(derived_exprs).collect()
        
        return result
    
    def aggregate_to_entity(
        self, 
        df: pl.DataFrame, 
        target_df: 'pd.DataFrame' = None,
        target_col: str = None
    ) -> pl.DataFrame:
        """
        Aggregate time series to 1 row per entity with FULL calculus features.
        
        Creates:
        - Window statistics (mean, std, sum, min, max, first, last, count)
        - Velocity (first derivative)
        - Acceleration (second derivative)
        - Momentum (jerk)
        - Integral (cumulative)
        - Limit extrapolation
        - Volatility ratios
        - Trend strength
        """
        group_col = self.config.group_col
        time_col = self.config.time_col
        windows = self.config.windows
        
        # Get numeric feature columns
        exclude = {group_col, time_col}
        numeric_cols = [c for c in df.columns 
                       if c not in exclude 
                       and df[c].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]]
        numeric_cols = numeric_cols[:self.config.max_cols]
        
        # Setup reverse time projection
        max_dates = df.group_by(group_col).agg(pl.col(time_col).max().alias("ref_date"))
        df = df.join(max_dates, on=group_col)
        df = df.with_columns(
            (pl.col("ref_date") - pl.col(time_col)).dt.total_days().alias("days_before")
        )
        
        # Start with unique entities
        all_features = df.select(group_col).unique()
        
        # === WINDOW-BASED STATISTICS ===
        for window in windows:
            w_data = df.filter(pl.col("days_before") <= window)
            
            agg_exprs = []
            for col in numeric_cols:
                agg_exprs.extend([
                    pl.col(col).mean().alias(f"{col}_w{window}_mean"),
                    pl.col(col).std().alias(f"{col}_w{window}_std"),
                    pl.col(col).sum().alias(f"{col}_w{window}_sum"),
                    pl.col(col).min().alias(f"{col}_w{window}_min"),
                    pl.col(col).max().alias(f"{col}_w{window}_max"),
                    pl.col(col).last().alias(f"{col}_w{window}_last"),
                    pl.col(col).first().alias(f"{col}_w{window}_first"),
                    pl.col(col).count().alias(f"{col}_w{window}_cnt"),
                ])
            
            agg = w_data.group_by(group_col).agg(agg_exprs)
            all_features = all_features.join(agg, on=group_col, how="left")
        
        # === CALCULUS FEATURES ===
        if self.config.enable_calculus:
            for col in numeric_cols:
                # Get available windows
                w_short = windows[0] if len(windows) > 0 else 7
                w_mid = windows[2] if len(windows) > 2 else 30
                w_long = windows[-1] if len(windows) > 0 else 360
                
                # --- VELOCITY (First Derivative) ---
                # df/dt = (f_last - f_first) / window
                for w in [w_short, w_mid, w_long]:
                    all_features = all_features.with_columns([
                        ((pl.col(f"{col}_w{w}_last") - pl.col(f"{col}_w{w}_first")) / w)
                        .alias(f"{col}_velocity_{w}")
                    ])
                
                # --- ACCELERATION (Second Derivative) ---
                # d²f/dt² = (velocity_short - velocity_long) / Δwindow
                all_features = all_features.with_columns([
                    ((pl.col(f"{col}_velocity_{w_short}") - pl.col(f"{col}_velocity_{w_mid}")) 
                     / (w_mid - w_short)).alias(f"{col}_accel_short"),
                    ((pl.col(f"{col}_velocity_{w_mid}") - pl.col(f"{col}_velocity_{w_long}")) 
                     / (w_long - w_mid)).alias(f"{col}_accel_long"),
                ])
                
                # --- MOMENTUM (Jerk - rate of acceleration change) ---
                all_features = all_features.with_columns([
                    (pl.col(f"{col}_accel_short") - pl.col(f"{col}_accel_long"))
                    .alias(f"{col}_momentum")
                ])
                
                # --- INTEGRAL (Cumulative sum as proxy) ---
                for w in [w_mid, w_long]:
                    all_features = all_features.with_columns([
                        pl.col(f"{col}_w{w}_sum").alias(f"{col}_integral_{w}")
                    ])
                
                # --- LIMIT EXTRAPOLATION ---
                # Predict future: current + velocity * lookahead
                for lookahead in [7, 30]:
                    all_features = all_features.with_columns([
                        (pl.col(f"{col}_w{w_short}_last") + 
                         pl.col(f"{col}_velocity_{w_short}") * lookahead)
                        .alias(f"{col}_predict_{lookahead}d")
                    ])
                
                # --- VOLATILITY RATIOS ---
                all_features = all_features.with_columns([
                    (pl.col(f"{col}_w{w_short}_std") / 
                     (pl.col(f"{col}_w{w_long}_std") + 0.001))
                    .alias(f"{col}_vol_ratio_{w_short}vs{w_long}"),
                ])
                
                # --- TREND STRENGTH ---
                all_features = all_features.with_columns([
                    (pl.col(f"{col}_w{w_short}_mean") / 
                     (pl.col(f"{col}_w{w_long}_mean") + 0.001))
                    .alias(f"{col}_trend_{w_short}vs{w_long}"),
                ])
                
                # --- RANGE ---
                for w in [w_short, w_mid]:
                    all_features = all_features.with_columns([
                        (pl.col(f"{col}_w{w}_max") - pl.col(f"{col}_w{w}_min"))
                        .alias(f"{col}_range_{w}")
                    ])
        
        # Fill nulls
        all_features = all_features.fill_null(0)
        all_features = all_features.fill_nan(0)
        
        # Join target if provided
        if target_df is not None and target_col:
            import pandas as pd
            target_pl = pl.from_pandas(target_df[[group_col, target_col]])
            all_features = all_features.join(target_pl, on=group_col, how='inner')
        
        return all_features


def create_calculus_features(
    df,
    group_col: str = 'cust_id',
    time_col: str = 'date',
    windows: List[int] = [7, 14, 30, 60, 90, 180, 360],
    target_df = None,
    target_col: str = None
) -> pl.DataFrame:
    """
    Convenience function to create full calculus features.
    
    Args:
        df: Input DataFrame (pandas or polars) with time series data
        group_col: Column to group by (entity ID)
        time_col: Time column for sorting
        windows: List of window sizes in days
        target_df: Optional target DataFrame with labels
        target_col: Name of target column
    
    Returns:
        Polars DataFrame with calculus features (1 row per entity)
    """
    config = TimeSeriesConfig(
        group_col=group_col,
        time_col=time_col,
        windows=windows,
        enable_calculus=True
    )
    preprocessor = TimeSeriesPreprocessor(config)
    transformed = preprocessor.transform(df)
    return preprocessor.aggregate_to_entity(transformed, target_df, target_col)


# Legacy function for backwards compatibility
def create_differential_features(
    df,
    group_col: str = 'cust_id',
    time_col: str = 'date',
    lags: List[int] = [1, 7, 14, 30],
    differentials: List[int] = [1, 7, 14, 30]
) -> pl.DataFrame:
    """
    Convenience function to create differential features (legacy).
    
    Args:
        df: Input DataFrame (pandas or polars)
        group_col: Column to group by (entity ID)
        time_col: Time column for sorting
        lags: List of lag values
        differentials: List of differential orders
    
    Returns:
        Polars DataFrame with derived features
    """
    config = TimeSeriesConfig(
        group_col=group_col,
        time_col=time_col,
        lags=lags,
        differentials=differentials,
        rolling_windows=[],
        enable_calculus=False
    )
    preprocessor = TimeSeriesPreprocessor(config)
    return preprocessor.transform(df)
