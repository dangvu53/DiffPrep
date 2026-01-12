# PreprocessorOps - New Preprocessing Operators

This folder contains modular preprocessing operators based on your new `preprocessor.py` design.

## Structure

Each operator wraps one preprocessing step with all its options:

### 1. **Imputation** (`imputation.py`)

- `none`: No imputation
- `mean`: Mean imputation
- `median`: Median imputation
- `most_frequent`: Mode imputation
- `knn`: KNN imputation
- `constant`: Constant value imputation

### 2. **Scaling** (`scaling.py`)

- `none`: No scaling
- `standard`: StandardScaler (z-score)
- `minmax`: MinMaxScaler [0,1]
- `robust`: RobustScaler (median-based)
- `maxabs`: MaxAbsScaler

### 3. **Encoding** (`encoding.py`)

- `none`: No encoding
- `onehot`: One-hot encoding

### 4. **Feature Selection** (`feature_selection.py`)

- `none`: No selection
- `variance_threshold`: Remove low-variance features
- `k_best`: Select k best features (F-test)
- `mutual_info`: Select by mutual information

### 5. **Outlier Removal** (`outlier_removal.py`)

- `none`: No removal
- `iqr`: IQR-based outlier removal
- `zscore`: Z-score based (|z| > 3)
- `lof`: Local Outlier Factor
- `isolation_forest`: Isolation Forest

### 6. **Dimensionality Reduction** (`dimensionality_reduction.py`)

- `none`: No reduction
- `pca`: Principal Component Analysis
- `svd`: Truncated SVD

## Usage Example

```python
from PreprocessorOps.imputation import Imputation
from PreprocessorOps.scaling import Scaling

# Create operators with different methods
imputer_mean = Imputation(method='mean')
imputer_knn = Imputation(method='knn')
scaler_std = Scaling(method='standard')
scaler_minmax = Scaling(method='minmax')

# Use in DiffPrep pipeline
from pipeline.diffprep_v2_pipeline import DiffPrepV2Pipeline

pipeline = DiffPrepV2Pipeline(
    imputation_options=[
        Imputation('mean'),
        Imputation('median'),
        Imputation('knn'),
        Imputation('none')
    ],
    scaling_options=[
        Scaling('standard'),
        Scaling('minmax'),
        Scaling('robust'),
        Scaling('none')
    ],
    # ... other options
)
```

## Integration with DiffPrep

Create a new pipeline file `pipeline/diffprep_v2_pipeline.py` that uses these operators instead of the old TFs folder operators.

## Advantages

1. ✅ **Consistent API**: All operators follow sklearn's fit/transform pattern
2. ✅ **Modular**: Each preprocessing step is independent
3. ✅ **Extensible**: Easy to add new methods to each operator
4. ✅ **Compatible**: Drop-in replacement for TFs operators
5. ✅ **All Options**: Covers all 6 preprocessing categories from preprocessor.py
