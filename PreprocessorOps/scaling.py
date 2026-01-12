import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler


class ScalingOperator:
    """Wrapper for scaling methods"""
    
    def __init__(self, method='standard'):
        """
        Available methods: 'none', 'standard', 'minmax', 'robust', 'maxabs'
        """
        self.method = f"scaling_{method}"  # Store with prefix for identification
        self.scaling_method = method
        self.scaler = None
        
    def fit(self, X, y=None):
        if self.scaling_method == 'none':
            return
        elif self.scaling_method == 'standard':
            self.scaler = StandardScaler()
        elif self.scaling_method == 'minmax':
            self.scaler = MinMaxScaler()
        elif self.scaling_method == 'robust':
            self.scaler = RobustScaler()
        elif self.scaling_method == 'maxabs':
            self.scaler = MaxAbsScaler()
        else:
            self.scaler = StandardScaler()
        
        self.scaler.fit(X)
    
    def transform(self, X):
        if self.scaling_method == 'none' or self.scaler is None:
            return X if isinstance(X, np.ndarray) else np.array(X)
        X_trans = self.scaler.transform(X)
        # Ensure numpy array output (sklearn might return DataFrame)
        if not isinstance(X_trans, np.ndarray):
            X_trans = np.array(X_trans)
        # Clip to prevent extreme values
        X_trans = np.clip(X_trans, -1e10, 1e10)
        return X_trans
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
