import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD


class DimensionalityReductionOperator:
    """Wrapper for dimensionality reduction methods"""
    
    def __init__(self, method='pca', n_components=10):
        """
        Available methods: 'none', 'pca', 'svd'
        """
        self.method = f"dimensionality_reduction_{method}"  # Method with prefix for DiffPrep
        self.dimensionality_reduction_method = method  # Original method name for internal use
        self.reducer = None
        self.n_components = n_components
        
    def fit(self, X, y=None):
        if self.dimensionality_reduction_method == 'none':
            return
        
        n_components = min(self.n_components, X.shape[0], X.shape[1])
        
        if self.dimensionality_reduction_method == 'pca':
            self.reducer = PCA(n_components=n_components)
        elif self.dimensionality_reduction_method == 'svd':
            self.reducer = TruncatedSVD(n_components=n_components)
        else:
            return
        
        self.reducer.fit(X)
    
    def transform(self, X):
        if self.dimensionality_reduction_method == 'none' or self.reducer is None:
            return X if isinstance(X, np.ndarray) else np.array(X)
        X_reduced = self.reducer.transform(X)
        # Ensure numpy array output
        if not isinstance(X_reduced, np.ndarray):
            X_reduced = np.array(X_reduced)
        return X_reduced
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
