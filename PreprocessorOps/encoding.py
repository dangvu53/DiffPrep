import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


class EncodingOperator:
    """Wrapper for encoding methods"""
    
    def __init__(self, method='onehot'):
        """
        Available methods: 'none', 'onehot'
        """
        self.method = f"encoding_{method}"
        self.encoding_method = method
        self.encoder = None
        
    def fit(self, X):
        if self.encoding_method == 'none':
            return
        elif self.encoding_method == 'onehot':
            try:
                # sklearn >= 1.2
                self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            except TypeError:
                # sklearn < 1.2
                self.encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
        
        if self.encoder is not None:
            self.encoder.fit(X)
    
    def transform(self, X):
        if self.encoding_method == 'none' or self.encoder is None:
            return X
        return self.encoder.transform(X)
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
