import numpy as np
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, mutual_info_classif


class FeatureSelectionOperator:
    """Wrapper for feature selection methods"""
    
    def __init__(self, method='variance_threshold', k=10):
        """
        Available methods: 'none', 'variance_threshold', 'k_best', 'mutual_info'
        """
        self.method = f"feature_selection_{method}"  # Method with prefix for DiffPrep
        self.feature_selection_method = method  # Original method name for internal use
        self.selector = None
        self.k = k
        
    def fit(self, X, y=None):
        if self.feature_selection_method == 'none':
            return
        elif self.feature_selection_method == 'variance_threshold':
            self.selector = VarianceThreshold(threshold=0.01)
            self.selector.fit(X)
        elif self.feature_selection_method == 'k_best':
            k = min(self.k, X.shape[1])
            self.selector = SelectKBest(f_classif, k=k)
            self.selector.fit(X, y)
        elif self.feature_selection_method == 'mutual_info':
            k = min(self.k, X.shape[1])
            self.selector = SelectKBest(mutual_info_classif, k=k)
            self.selector.fit(X, y)
    
    def transform(self, X):
        if self.feature_selection_method == 'none' or self.selector is None:
            return X
        return self.selector.transform(X)
    
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)
