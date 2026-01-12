"""
Example usage of PreprocessorOps with DiffPrep

This script shows how to create preprocessing operators from the PreprocessorOps folder
and use them in DiffPrep's search space.
"""


class IdentityOperator(object):
    """Identity Transformer that does nothing - passes data through unchanged"""
    def __init__(self):
        self.method = "identity"
    
    def fit(self, X):
        return self
    
    def transform(self, X):
        return X
    
    def fit_transform(self, X):
        return X


from PreprocessorOps.imputation import ImputationOperator
from PreprocessorOps.scaling import ScalingOperator
from PreprocessorOps.encoding import EncodingOperator
from PreprocessorOps.feature_selection import FeatureSelectionOperator
from PreprocessorOps.outlier_removal import OutlierRemovalOperator
from PreprocessorOps.dimensionality_reduction import DimensionalityReductionOperator


def create_imputation_operators():
    """Create all imputation operators"""
    return [
        ImputationOperator('none'),
        ImputationOperator('mean'),
        ImputationOperator('median'),
        ImputationOperator('most_frequent'),
        ImputationOperator('knn'),
        ImputationOperator('constant'),
    ]


def create_scaling_operators():
    """Create all scaling operators"""
    return [
        ScalingOperator('none'),
        ScalingOperator('standard'),
        ScalingOperator('minmax'),
        ScalingOperator('robust'),
        ScalingOperator('maxabs'),
    ]


def create_encoding_operators():
    """Create all encoding operators"""
    return [
        EncodingOperator('none'),
        EncodingOperator('onehot'),
    ]


def create_feature_selection_operators(k=10):
    """Create all feature selection operators"""
    return [
        FeatureSelectionOperator('none'),
        FeatureSelectionOperator('variance_threshold'),
        FeatureSelectionOperator('k_best', k=k),
        FeatureSelectionOperator('mutual_info', k=k),
    ]


def create_outlier_removal_operators():
    """Create all outlier removal operators"""
    return [
        OutlierRemovalOperator('none'),
        OutlierRemovalOperator('iqr'),
        OutlierRemovalOperator('zscore'),
        OutlierRemovalOperator('lof'),
        OutlierRemovalOperator('isolation_forest'),
    ]


def create_dimensionality_reduction_operators(n_components=10):
    """Create all dimensionality reduction operators"""
    return [
        DimensionalityReductionOperator('none'),
        DimensionalityReductionOperator('pca', n_components=n_components),
        DimensionalityReductionOperator('svd', n_components=n_components),
    ]


def create_all_operators(k=10, n_components=10):
    """
    Create all operators for DiffPrep search space
    
    Returns:
        dict: Dictionary with operator categories and their instances
    """
    return {
        'imputation': create_imputation_operators(),
        'scaling': create_scaling_operators(),
        'encoding': create_encoding_operators(),
        'feature_selection': create_feature_selection_operators(k=k),
        'outlier_removal': create_outlier_removal_operators(),
        'dimensionality_reduction': create_dimensionality_reduction_operators(n_components=n_components),
    }


if __name__ == "__main__":
    # Example: Create all operators
    operators = create_all_operators(k=10, n_components=10)
    
    print("=" * 60)
    print("PreprocessorOps - All Available Operators")
    print("=" * 60)
    
    for category, ops in operators.items():
        print(f"\n{category.upper()}:")
        for op in ops:
            print(f"  - {op.method}")
    
    print("\n" + "=" * 60)
    print(f"Total operators: {sum(len(ops) for ops in operators.values())}")
    print("=" * 60)
