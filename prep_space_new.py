"""
Preprocessing space definition using new PreprocessorOps operators.

Follows preprocessor.py structure:
1. imputation (handled by FirstTransformer for num/cat separately)
2. scaling
3. encoding (separate stage)
4. feature_selection
5. outlier_removal  
6. dimensionality_reduction
"""

from PreprocessorOps.imputation import ImputationOperator
from PreprocessorOps.scaling import ScalingOperator
from PreprocessorOps.encoding import EncodingOperator
from PreprocessorOps.feature_selection import FeatureSelectionOperator
from PreprocessorOps.outlier_removal import OutlierRemovalOperator
from PreprocessorOps.dimensionality_reduction import DimensionalityReductionOperator


# Define the search space for DiffPrep
space = [
    {
        "name": "missing_value_imputation",
        "num_tf_options": [
            ImputationOperator("mean"),
            ImputationOperator("median"),
            ImputationOperator("most_frequent"),
            ImputationOperator("knn"),
        ],
        "cat_tf_options": [
            ImputationOperator("most_frequent"),
            ImputationOperator("constant"),
        ],
        "default": [ImputationOperator("mean"), ImputationOperator("most_frequent")],
        "init": [(ImputationOperator("mean"), 0.5), (ImputationOperator("most_frequent"), 0.5)]
    },
    {
        "name": "scaling",
        "tf_options": [
            ScalingOperator("none"),
            ScalingOperator("standard"),
            ScalingOperator("minmax"),
            ScalingOperator("robust"),
            ScalingOperator("maxabs"),
        ],
        "default": ScalingOperator("standard"),
        "init": (ScalingOperator("standard"), 0.5)
    },
    # Note: Encoding stage removed - OneHotEncoder changes dimensions (incompatible with DiffPrep Transformer)
    {
        "name": "feature_selection",
        "tf_options": [
            FeatureSelectionOperator("none"),
            FeatureSelectionOperator("variance_threshold"),
            FeatureSelectionOperator("k_best", k=10),
            FeatureSelectionOperator("mutual_info", k=10),
        ],
        "default": FeatureSelectionOperator("none"),
        "init": (FeatureSelectionOperator("none"), 0.5)
    },
    {
        "name": "outlier_removal",
        "tf_options": [
            OutlierRemovalOperator("none"),
            OutlierRemovalOperator("iqr"),
            OutlierRemovalOperator("zscore"),
            OutlierRemovalOperator("lof"),
            OutlierRemovalOperator("isolation_forest"),
        ],
        "default": OutlierRemovalOperator("none"),
        "init": (OutlierRemovalOperator("none"), 0.5)
    },
    {
        "name": "dimensionality_reduction",
        "tf_options": [
            DimensionalityReductionOperator("none"),
            DimensionalityReductionOperator("pca"),
            DimensionalityReductionOperator("svd"),
        ],
        "default": DimensionalityReductionOperator("none"),
        "init": (DimensionalityReductionOperator("none"), 0.5)
    },
]


