import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


class OutlierRemovalOperator:
    """Wrapper for outlier removal methods"""
    
    def __init__(self, method='iqr'):
        """
        Available methods: 'none', 'iqr', 'zscore', 'lof', 'isolation_forest'
        """
        self.method = f"outlier_removal_{method}"  # Method with prefix for DiffPrep
        self.outlier_removal_method = method  # Original method name for internal use
        self.fitted = False
        
    def fit(self, X, y=None):
        self.fitted = True
        # No fitting needed for most methods, they detect on-the-fly
        
    def transform(self, X):
        """
        Note: This returns the data unchanged - outlier removal should only 
        happen during training, not during test transform
        """
        if self.outlier_removal_method == 'none':
            return X if isinstance(X, np.ndarray) else np.array(X)
        # For transform (test time), don't remove outliers
        # Ensure numpy array output
        return X if isinstance(X, np.ndarray) else np.array(X)
    
    def fit_transform_with_y(self, X, y=None):
        """
        Special method for training that actually removes outliers
        Returns: X_cleaned, y_cleaned (if y provided)
        """
        if self.outlier_removal_method == 'none':
            return X, y
        
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        
        # Compute mask for outliers
        if self.outlier_removal_method == 'iqr':
            mask = pd.Series(True, index=X.index)
            for col in X.columns:
                Q1, Q3 = X[col].quantile([0.25, 0.75])
                IQR = Q3 - Q1
                if IQR > 0:
                    mask &= (X[col] >= Q1 - 1.5 * IQR) & (X[col] <= Q3 + 1.5 * IQR)
        
        elif self.outlier_removal_method == 'zscore':
            Z = np.abs(zscore(X))
            mask = pd.Series((Z < 3).all(axis=1), index=X.index)
        
        elif self.outlier_removal_method == 'lof':
            lof = LocalOutlierFactor(n_neighbors=min(20, len(X)-1))
            mask = pd.Series(lof.fit_predict(X) == 1, index=X.index)
        
        elif self.outlier_removal_method == 'isolation_forest':
            iso = IsolationForest(contamination=0.05, random_state=42)
            mask = pd.Series(iso.fit_predict(X) == 1, index=X.index)
        
        # Apply mask
        X_cleaned = X.loc[mask].reset_index(drop=True).values
        
        if y is not None:
            y_cleaned = pd.Series(y).loc[mask].reset_index(drop=True).values
            return X_cleaned, y_cleaned
        
        return X_cleaned, None
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
