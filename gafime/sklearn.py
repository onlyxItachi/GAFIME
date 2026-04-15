from typing import Optional, List, Tuple
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, check_X_y, check_array

from gafime.engine import GafimeEngine
from gafime.config import EngineConfig, ComputeBudget


class GafimeSelector(BaseEstimator, TransformerMixin):
    """
    Scikit-Learn compatible wrapper for GAFIME.
    
    This transformer evaluates all pairwise feature interactions during `fit()`, 
    finds the top `k` most predictive interactions based on the provided metric,
    and then appends these generated interaction features to the input data during `transform()`.
    
    Args:
        k: The number of top feature interactions to retain and append.
        backend: The computing backend to use ('auto', 'cuda', 'metal', 'cpu', 'rust', 'python').
        metric: The evaluation metric ('pearson' or 'spearman' or 'r2').
        operator: The arithmetic operator used to combine features ('multiply', 'add', 'subtract', 'divide').
        n_jobs: Number of OpenMP threads to use for CPU backends (-1 means all cores).
        verbose: If True, prints diagnostic information from the engine.
    """
    
    def __init__(self, 
                 k: int = 10, 
                 backend: str = 'auto', 
                 metric: str = 'pearson', 
                 operator: str = 'multiply',
                 n_jobs: int = -1,
                 verbose: bool = False):
        self.k = k
        self.backend = backend
        self.metric = metric
        self.operator = operator
        self.n_jobs = n_jobs
        self.verbose = verbose

    def fit(self, X, y):
        """
        Fit the transformer by evaluating all feature interactions and selecting the top k.
        """
        # Validate inputs
        X, y = check_X_y(X, y, accept_sparse=False, dtype=[np.float32, np.float64])
        
        # Initialize the underlying GAFIME engine with a strict budget to save time if we only want small combinations
        config = EngineConfig(
            backend=self.backend,
            metric_names=(self.metric,),
            budget=ComputeBudget(max_comb_size=2)  # Currently wrapper focuses on pairwise
        )
        engine = GafimeEngine(config=config)
        
        # Evaluate all interactions
        report = engine.analyze(X, y)
        
        # Extract the highest scoring pairs
        # InteractionResult has .combo (Tuple[int, ...]) and .metrics (Dict[str, float])
        # We sort by the absolute value of the chosen metric score
        sorted_interactions = sorted(
            report.interactions, 
            key=lambda x: abs(x.metrics.get(self.metric, 0.0)), 
            reverse=True
        )
        
        # Filter for only 2-way interactions to apply our operator pairwise
        pairwise = [x.combo for x in sorted_interactions if len(x.combo) == 2]
        
        self.top_interactions_ = pairwise[:self.k]
        self.n_features_in_ = X.shape[1]
        
        return self

    def transform(self, X):
        """
        Apply the learned feature interactions to X and append them as new columns.
        """
        check_is_fitted(self, ['top_interactions_', 'n_features_in_'])
        X = check_array(X, accept_sparse=False, dtype=[np.float32, np.float64])
        
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"Shape of input is different from what was seen in `fit` "
                             f"({X.shape[1]} != {self.n_features_in_})")
            
        n_samples = X.shape[0]
        augmented_features = np.empty((n_samples, len(self.top_interactions_)), dtype=X.dtype)
        
        # Generate the interaction features mathematically on the CPU.
        # Note: In a production pipeline, this transformation typically applies to 
        # validation/test sets which are smaller, so numpy broadcasting is efficient enough.
        for idx, (feat_i, feat_j) in enumerate(self.top_interactions_):
            col_i = X[:, feat_i]
            col_j = X[:, feat_j]
            
            if self.operator == 'multiply':
                augmented_features[:, idx] = col_i * col_j
            elif self.operator == 'add':
                augmented_features[:, idx] = col_i + col_j
            elif self.operator == 'subtract':
                augmented_features[:, idx] = col_i - col_j
            elif self.operator == 'divide':
                # Add tiny epsilon to avoid division by zero
                augmented_features[:, idx] = col_i / (col_j + 1e-8)
            else:
                raise ValueError(f"Unsupported operator encountered during transform: {self.operator}")
                
        # Horizontally stack the original features and the new interaction features
        return np.hstack((X, augmented_features))
