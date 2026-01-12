"""
Imputation operators for DiffPrep pipeline.

Compatible with DiffPrep's FirstTransformer interface - uses sklearn directly.
"""

import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer


class ImputationOperator:
    """
    Imputation operator compatible with DiffPrep FirstTransformer.
    Handles numerical and categorical data separately using sklearn.
    """
    
    def __init__(self, method='mean'):
        self.method = f"imputation_{method}"
        self.imputation_method = method
        self.input_type = "mixed"  # Handles both numerical and categorical
        self.num_imputer = None
        self.cat_imputer = None
        
    def fit(self, X_num, X_cat):
        """Fit imputers for numerical and categorical data"""
        # Numerical imputer
        if X_num.shape[1] > 0:
            if self.imputation_method == 'mean':
                self.num_imputer = SimpleImputer(strategy='mean')
            elif self.imputation_method == 'median':
                self.num_imputer = SimpleImputer(strategy='median')
            elif self.imputation_method == 'most_frequent':
                self.num_imputer = SimpleImputer(strategy='most_frequent')
            elif self.imputation_method == 'knn':
                self.num_imputer = KNNImputer(n_neighbors=5)
            elif self.imputation_method == 'constant':
                self.num_imputer = SimpleImputer(strategy='constant', fill_value=0)
            else:
                self.num_imputer = SimpleImputer(strategy='mean')
            
            self.num_imputer.fit(X_num)
        
        # Categorical imputer
        if X_cat.shape[1] > 0:
            if self.imputation_method == 'constant':
                # Use 0 for numeric, will work even if categorical is stored as float
                self.cat_imputer = SimpleImputer(strategy='constant', fill_value=0)
            else:
                self.cat_imputer = SimpleImputer(strategy='most_frequent')
            
            self.cat_imputer.fit(X_cat)
    
    def transform(self, X_num, X_cat):
        """Transform numerical and categorical data"""
        X_num_trans = X_num
        X_cat_trans = X_cat
        
        if self.num_imputer is not None and X_num.shape[1] > 0:
            X_num_trans = self.num_imputer.transform(X_num)
        
        if self.cat_imputer is not None and X_cat.shape[1] > 0:
            X_cat_trans = self.cat_imputer.transform(X_cat)
        
        return X_num_trans, X_cat_trans
    
    def fit_transform(self, X_num, X_cat):
        """Fit and transform"""
        self.fit(X_num, X_cat)
        return self.transform(X_num, X_cat)



