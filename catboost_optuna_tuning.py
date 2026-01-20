"""
CatBoost GPU + Optuna Hyperparameter Tuning

Goal: Push past 75% by optimizing hyperparameters
- CatBoost with GPU (if available) or CPU
- Optuna for smart hyperparameter search
- GAFIME time series features + categorical handling
"""

import pandas as pd
import numpy as np
import polars as pl
import sys
import optuna
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier
from itertools import combinations, product

sys.path.insert(0, r'c:\Users\Hamza\Desktop\GAFIME')
from gafime.preprocessors import TimeSeriesPreprocessor, TimeSeriesConfig
from gafime.backends.fused_kernel import (
    StaticBucket, UnaryOp, InteractionType,
    compute_pearson_from_stats, create_fold_mask
)

optuna.logging.set_verbosity(optuna.logging.WARNING)

BASE = Path(r'c:\Users\Hamza\Desktop\GAFIME\ing-hubs-turkiye-datathon')

print("=" * 70)
print("CatBoost GPU + Optuna Hyperparameter Tuning")
print("=" * 70)

# ============================================================================
# 1. LOAD AND PREPARE DATA
# ============================================================================
print("\n[1/5] Loading data...")

history = pl.read_csv(BASE / 'customer_history.csv')
customers = pd.read_csv(BASE / 'customers.csv')
train_ref = pd.read_csv(BASE / 'referance_data.csv')
test_ref = pd.read_csv(BASE / 'referance_data_test.csv')

history = history.with_columns(pl.col('date').str.to_datetime())

# ============================================================================
# 2. TIME SERIES FEATURES (GAFIME Core)
# ============================================================================
print("\n[2/5] Creating time series features...")

config = TimeSeriesConfig(
    group_col='cust_id',
    time_col='date',
    windows=[7, 14, 30, 60, 90, 180, 360],
    enable_calculus=True,
    max_cols=10
)

tsp = TimeSeriesPreprocessor(config)
ts_features = tsp.aggregate_to_entity(tsp.transform(history)).to_pandas()
print(f"     Time series features: {ts_features.shape}")

# ============================================================================
# 3. MERGE AND PREPARE FEATURES
# ============================================================================
print("\n[3/5] Preparing features...")

# Merge with customers
data = ts_features.merge(customers, on='cust_id', how='left')

# Split train/test
train_data = data[data['cust_id'].isin(train_ref['cust_id'])].copy()
test_data = data[data['cust_id'].isin(test_ref['cust_id'])].copy()

# Add target
train_data = train_data.merge(train_ref[['cust_id', 'churn']], on='cust_id', how='left')

# Get column types
cat_cols = customers.select_dtypes(include=['object']).columns.tolist()
num_cols = [c for c in train_data.columns if c not in ['cust_id', 'churn'] + cat_cols]

# Prepare arrays
X_num = train_data[num_cols].fillna(0).values.astype(np.float32)
X_cat = train_data[cat_cols].fillna('missing').values
X = np.column_stack([X_num, X_cat])
y = train_data['churn'].values.astype(np.float32)

X_test_num = test_data[num_cols].fillna(0).values.astype(np.float32)
X_test_cat = test_data[cat_cols].fillna('missing').values
X_test = np.column_stack([X_test_num, X_test_cat])
test_ids = test_data['cust_id'].values

cat_indices = list(range(len(num_cols), X.shape[1]))

print(f"     Total features: {X.shape[1]}")
print(f"     Categorical indices: {len(cat_indices)}")

# Split for validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ============================================================================
# 4. OPTUNA HYPERPARAMETER TUNING
# ============================================================================
print("\n[4/5] Optuna hyperparameter tuning (20 trials)...")

# Check GPU availability
try:
    test_model = CatBoostClassifier(task_type='GPU', iterations=1, verbose=False)
    test_model.fit([[0,0,'a']], [0], cat_features=[2])
    USE_GPU = True
    print("     GPU available!")
except:
    USE_GPU = False
    print("     GPU not available, using CPU")

def objective(trial):
    params = {
        'depth': trial.suggest_int('depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'iterations': trial.suggest_int('iterations', 200, 800),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
        'random_strength': trial.suggest_float('random_strength', 0, 10),
        'cat_features': cat_indices,
        'verbose': False,
        'random_state': 42,
        'early_stopping_rounds': 50,
    }
    
    if USE_GPU:
        params['task_type'] = 'GPU'
    
    model = CatBoostClassifier(**params)
    model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
    
    preds = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, preds)
    
    return auc

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20, show_progress_bar=True)

print(f"\n     Best AUC: {study.best_value:.4f}")
print(f"     Best params: {study.best_params}")

# ============================================================================
# 5. TRAIN FINAL MODEL WITH BEST PARAMS
# ============================================================================
print("\n[5/5] Training final model with best params...")

best_params = study.best_params
best_params['cat_features'] = cat_indices
best_params['verbose'] = False
best_params['random_state'] = 42

if USE_GPU:
    best_params['task_type'] = 'GPU'

# Train on full training data
final_model = CatBoostClassifier(**best_params)
final_model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)

# Validation score
val_preds = final_model.predict_proba(X_val)[:, 1]
final_auc = roc_auc_score(y_val, val_preds)
print(f"     Final Validation AUC: {final_auc:.4f}")

# Test predictions
test_preds = final_model.predict_proba(X_test)[:, 1]

# Save submission
submission = pd.DataFrame({'cust_id': test_ids, 'churn': test_preds})
output_path = BASE / 'catboost_optuna_submission.csv'
submission.to_csv(output_path, index=False)
print(f"     Saved: {output_path}")

# ============================================================================
# RESULTS
# ============================================================================
print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)
print(f"""
Baseline (XGBoost):         73.8%
CatBoost + Categorical:     75.0%
CatBoost + Optuna Tuned:    {final_auc:.1%}

Best Hyperparameters:
  - depth: {best_params.get('depth')}
  - learning_rate: {best_params.get('learning_rate'):.4f}
  - iterations: {best_params.get('iterations')}
  - l2_leaf_reg: {best_params.get('l2_leaf_reg'):.2f}

Submission: catboost_optuna_submission.csv
""")
print("=" * 70)
