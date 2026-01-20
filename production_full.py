"""
PRODUCTION-GRADE: Full Calculus Time Series + GAFIME 150 Features

Complete implementation:
1. All time windows (7, 14, 30, 60, 90, 180, 360 days)
2. Full statistics (mean, std, sum, min, max, first, last, count)
3. First derivative (velocity): Δf/Δt
4. Second derivative (acceleration): Δ²f/Δt²
5. Integral (cumulative): Σf
6. Momentum: rate of acceleration change
7. Limit extrapolation: predict future from trend
8. Volatility ratios: short-term vs long-term variance
9. GAFIME feature interactions (130 features)
10. XGBoost with tuned hyperparameters
"""
import pandas as pd
import numpy as np
import polars as pl
import sys
import gc
from pathlib import Path
from itertools import combinations, product
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import xgboost as xgb

sys.path.insert(0, str(Path(r'c:\Users\Hamza\Desktop\GAFIME')))
from gafime.backends.fused_kernel import (
    StaticBucket, UnaryOp, InteractionType,
    compute_pearson_from_stats, create_fold_mask
)

BASE_DIR = Path(r"c:\Users\Hamza\Desktop\GAFIME\ing-hubs-turkiye-datathon")

print("=" * 70)
print("PRODUCTION-GRADE: FULL CALCULUS + GAFIME 150 FEATURES")
print("=" * 70)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n[1/8] Loading data...")
history = pl.read_csv(BASE_DIR / "customer_history.csv")
train_ref = pl.read_csv(BASE_DIR / "referance_data.csv")
test_ref = pl.read_csv(BASE_DIR / "referance_data_test.csv")

history = history.with_columns(pl.col("date").str.to_datetime())
history = history.sort(["cust_id", "date"])

numeric_cols = [c for c in history.columns if c not in ['cust_id', 'date']]
print(f"     Records: {history.shape[0]:,}")
print(f"     Numeric columns: {numeric_cols}")

# ============================================================================
# 2. REVERSE TIME PROJECTION
# ============================================================================
print("\n[2/8] Setting up reverse time projection...")
max_dates = history.group_by("cust_id").agg(pl.col("date").max().alias("ref_date"))
history = history.join(max_dates, on="cust_id")
history = history.with_columns(
    (pl.col("ref_date") - pl.col("date")).dt.total_days().alias("days_before")
)

# ============================================================================
# 3. CREATE BASE WINDOW FEATURES
# ============================================================================
print("\n[3/8] Creating window-based features...")

WINDOWS = [7, 14, 30, 60, 90, 180, 360]
all_features = history.select("cust_id").unique()

for window in WINDOWS:
    print(f"     Window {window} days...", end=" ")
    w_data = history.filter(pl.col("days_before") <= window)
    
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
    
    agg = w_data.group_by("cust_id").agg(agg_exprs)
    all_features = all_features.join(agg, on="cust_id", how="left")
    print("done")

print(f"     Features so far: {len(all_features.columns)-1}")

# ============================================================================
# 4. CALCULUS FEATURES
# ============================================================================
print("\n[4/8] Creating calculus features...")

for col in numeric_cols:
    # === FIRST DERIVATIVE (velocity) ===
    # df/dt = (f_recent - f_old) / Δt
    all_features = all_features.with_columns([
        ((pl.col(f"{col}_w7_last") - pl.col(f"{col}_w7_first")) / 7).alias(f"{col}_velocity_7"),
        ((pl.col(f"{col}_w30_last") - pl.col(f"{col}_w30_first")) / 30).alias(f"{col}_velocity_30"),
        ((pl.col(f"{col}_w90_last") - pl.col(f"{col}_w90_first")) / 90).alias(f"{col}_velocity_90"),
    ])
    
    # === SECOND DERIVATIVE (acceleration) ===
    # d²f/dt² = (velocity_recent - velocity_old) / Δt
    all_features = all_features.with_columns([
        ((pl.col(f"{col}_velocity_7") - pl.col(f"{col}_velocity_30")) / 23).alias(f"{col}_accel_short"),
        ((pl.col(f"{col}_velocity_30") - pl.col(f"{col}_velocity_90")) / 60).alias(f"{col}_accel_long"),
    ])
    
    # === MOMENTUM (jerk - rate of acceleration change) ===
    all_features = all_features.with_columns([
        (pl.col(f"{col}_accel_short") - pl.col(f"{col}_accel_long")).alias(f"{col}_momentum")
    ])
    
    # === INTEGRAL (cumulative sum proxy) ===
    all_features = all_features.with_columns([
        (pl.col(f"{col}_w30_sum")).alias(f"{col}_integral_30"),
        (pl.col(f"{col}_w90_sum")).alias(f"{col}_integral_90"),
        (pl.col(f"{col}_w360_sum")).alias(f"{col}_integral_360"),
    ])
    
    # === LIMIT EXTRAPOLATION ===
    # Predict future: current + velocity * lookahead
    all_features = all_features.with_columns([
        (pl.col(f"{col}_w7_last") + pl.col(f"{col}_velocity_7") * 7).alias(f"{col}_predict_7d"),
        (pl.col(f"{col}_w7_last") + pl.col(f"{col}_velocity_7") * 30).alias(f"{col}_predict_30d"),
    ])
    
    # === VOLATILITY RATIOS ===
    all_features = all_features.with_columns([
        (pl.col(f"{col}_w7_std") / (pl.col(f"{col}_w90_std") + 0.001)).alias(f"{col}_vol_ratio_7vs90"),
        (pl.col(f"{col}_w30_std") / (pl.col(f"{col}_w360_std") + 0.001)).alias(f"{col}_vol_ratio_30vs360"),
    ])
    
    # === RANGE FEATURES ===
    all_features = all_features.with_columns([
        (pl.col(f"{col}_w7_max") - pl.col(f"{col}_w7_min")).alias(f"{col}_range_7"),
        (pl.col(f"{col}_w30_max") - pl.col(f"{col}_w30_min")).alias(f"{col}_range_30"),
    ])
    
    # === TREND STRENGTH ===
    # Ratio of recent to historical average
    all_features = all_features.with_columns([
        (pl.col(f"{col}_w7_mean") / (pl.col(f"{col}_w360_mean") + 0.001)).alias(f"{col}_trend_7vs360"),
        (pl.col(f"{col}_w30_mean") / (pl.col(f"{col}_w90_mean") + 0.001)).alias(f"{col}_trend_30vs90"),
    ])

# Fill nulls and handle inf
all_features = all_features.fill_null(0)
all_features = all_features.fill_nan(0)

print(f"     Total features: {len(all_features.columns)-1}")

# ============================================================================
# 5. SPLIT TRAIN/TEST
# ============================================================================
print("\n[5/8] Preparing train/test data...")

train_features = all_features.join(train_ref.select(["cust_id", "churn"]), on="cust_id", how="inner")
test_features = all_features.join(test_ref.select(["cust_id"]), on="cust_id", how="inner")

feature_cols = [c for c in train_features.columns if c not in ['cust_id', 'churn']]

X = train_features.select(feature_cols).to_numpy().astype(np.float32)
y = train_features.select("churn").to_numpy().flatten().astype(np.float32)
X_test = test_features.select(feature_cols).to_numpy().astype(np.float32)
test_ids = test_features.select("cust_id").to_numpy().flatten()

# Clean data
X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
X_test = np.nan_to_num(X_test, nan=0, posinf=0, neginf=0)

n_samples, n_features = X.shape
print(f"     Train: {n_samples:,} x {n_features}")
print(f"     Test: {len(test_ids):,}")

# ============================================================================
# 6. GAFIME FEATURE INTERACTIONS
# ============================================================================
print("\n[6/8] Running GAFIME (target: 130 interactions)...")

# Find valid features (non-zero variance, no inf)
valid_idx = []
for i in range(n_features):
    if np.std(X[:, i]) > 1e-6 and not np.any(np.isinf(X[:, i])):
        valid_idx.append(i)

print(f"     Valid features: {len(valid_idx)}/{n_features}")

mask = create_fold_mask(n_samples, n_folds=5, seed=42)
OPS = [UnaryOp.IDENTITY, UnaryOp.LOG, UnaryOp.SQRT, UnaryOp.SQUARE]
INTERACTIONS = [InteractionType.MULT, InteractionType.ADD, InteractionType.SUB, InteractionType.DIV]

discovered = []
search_idx = valid_idx[:60]  # Top 60 valid features

for f0, f1 in combinations(search_idx, 2):
    try:
        bucket = StaticBucket(n_samples, 2)
        bucket.upload_feature(0, X[:, f0])
        bucket.upload_feature(1, X[:, f1])
        bucket.upload_target(y)
        bucket.upload_mask(mask)
        
        for op0, op1 in product(OPS, repeat=2):
            for interact in INTERACTIONS:
                try:
                    stats = bucket.compute([0, 1], [op0, op1], interact, 0)
                    train_r, val_r = compute_pearson_from_stats(stats)
                    
                    if not np.isnan(val_r) and abs(val_r) >= 0.03 and abs(val_r) < 0.999:
                        discovered.append({
                            'f0': f0, 'f1': f1,
                            'ops': (int(op0), int(op1)),
                            'int': int(interact),
                            'r': float(val_r)
                        })
                except:
                    pass
        del bucket
    except:
        pass

discovered.sort(key=lambda x: abs(x['r']), reverse=True)
discovered = discovered[:130]
print(f"     GAFIME found: {len(discovered)} interactions")
if discovered:
    print(f"     Best r: {discovered[0]['r']:.4f}")

# Apply GAFIME features
def apply_op(x, op):
    if op == 0: return x
    elif op == 1: return np.log(np.abs(x) + 1e-8)
    elif op == 3: return np.sqrt(np.abs(x))
    elif op == 6: return x ** 2
    else: return x

def apply_int(a, b, i):
    if i == 0: return a * b
    elif i == 1: return a + b
    elif i == 2: return a - b
    elif i == 3: return a / (b + 1e-8)
    else: return a * b

gafime_tr = []
gafime_te = []
for d in discovered:
    gafime_tr.append(apply_int(apply_op(X[:, d['f0']], d['ops'][0]), 
                                apply_op(X[:, d['f1']], d['ops'][1]), d['int']))
    gafime_te.append(apply_int(apply_op(X_test[:, d['f0']], d['ops'][0]),
                                apply_op(X_test[:, d['f1']], d['ops'][1]), d['int']))

if gafime_tr:
    X_full = np.column_stack([X] + gafime_tr)
    X_test_full = np.column_stack([X_test] + gafime_te)
else:
    X_full = X
    X_test_full = X_test

X_full = np.nan_to_num(X_full, nan=0, posinf=0, neginf=0)
X_test_full = np.nan_to_num(X_test_full, nan=0, posinf=0, neginf=0)

print(f"     Final features: {X_full.shape[1]} (base: {n_features} + GAFIME: {len(discovered)})")

# ============================================================================
# 7. TRAIN MODEL
# ============================================================================
print("\n[7/8] Training XGBoost...")

X_tr, X_val, y_tr, y_val = train_test_split(X_full, y, test_size=0.2, random_state=42, stratify=y)

model = xgb.XGBClassifier(
    max_depth=7,
    n_estimators=500,
    learning_rate=0.02,
    subsample=0.8,
    colsample_bytree=0.6,
    min_child_weight=3,
    reg_alpha=0.1,
    reg_lambda=1.0,
    use_label_encoder=False,
    eval_metric='auc',
    early_stopping_rounds=50,
    verbosity=0,
    random_state=42,
    n_jobs=-1
)
model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

preds_val = model.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y_val, preds_val)
print(f"     VALIDATION AUC: {auc:.4f}")

# ============================================================================
# 8. SAVE SUBMISSION
# ============================================================================
print("\n[8/8] Saving submission...")

preds_test = model.predict_proba(X_test_full)[:, 1]
sub = pd.DataFrame({'cust_id': test_ids, 'churn': preds_test})

output_dir = BASE_DIR / "submissions_comparison"
output_dir.mkdir(exist_ok=True)
sub.to_csv(output_dir / "production_full_calculus_gafime.csv", index=False)

print(f"     Saved: production_full_calculus_gafime.csv")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("PRODUCTION RUN COMPLETE")
print("=" * 70)
print(f"""
Features Breakdown:
  - Window features: {len(WINDOWS)} windows x {len(numeric_cols)} cols x 8 stats
  - Calculus: velocity, acceleration, momentum, integral, predict, volatility, range, trend
  - GAFIME interactions: {len(discovered)}
  - TOTAL: {X_full.shape[1]}

Results:
  - Validation AUC: {auc:.4f}
  - Winner target: ~0.7600

Submission: production_full_calculus_gafime.csv
""")
print("=" * 70)
