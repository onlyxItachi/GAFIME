import pytest
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

from gafime.sklearn import GafimeSelector

@pytest.fixture
def dummy_data():
    np.random.seed(42)
    n_samples, n_features = 100, 10
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = np.random.randint(0, 2, size=n_samples).astype(np.float32)
    return X, y

def test_gafime_selector_fit_transform(dummy_data):
    X, y = dummy_data
    
    # We want top 3 interactions using Python backend to avoid requiring CUDA in basic CI for this test
    selector = GafimeSelector(k=3, backend='python', operator='multiply')
    
    # 1. Test fit
    selector.fit(X, y)
    assert hasattr(selector, 'top_interactions_')
    assert len(selector.top_interactions_) == 3
    assert selector.n_features_in_ == 10
    
    # 2. Test transform
    X_transformed = selector.transform(X)
    
    # Original (10) + k augmented features (3)
    assert X_transformed.shape[1] == 13
    assert X_transformed.shape[0] == 100

def test_gafime_pipeline_integration(dummy_data):
    X, y = dummy_data
    
    pipe = Pipeline([
        ('interaction_miner', GafimeSelector(k=5, backend='python')),
        ('classifier', LogisticRegression())
    ])
    
    # This proves that our Scikit-Learn wrapper plays nicely with sklearn.pipeline.Pipeline
    pipe.fit(X, y)
    
    preds = pipe.predict(X)
    assert preds.shape == (100,)

def test_gafime_gridsearchcv(dummy_data):
    X, y = dummy_data
    
    pipe = Pipeline([
        ('miner', GafimeSelector(backend='python')),
        ('classifier', LogisticRegression())
    ])
    
    param_grid = {
        'miner__k': [2, 4],
        'miner__metric': ['pearson']
    }
    
    grid = GridSearchCV(pipe, param_grid, cv=2)
    grid.fit(X, y)
    
    assert grid.best_estimator_ is not None
    assert grid.best_params_['miner__k'] in [2, 4]
