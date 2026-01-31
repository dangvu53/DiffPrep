"""
Apply a saved preprocessing pipeline configuration to a dataset.

Usage:
    python apply_pipeline.py --dataset eeg --method diffprep_fix --output preprocessed_eeg.csv
"""

import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


class PreprocessingPipeline:
    """Apply preprocessing transformations based on a pipeline configuration."""
    
    def __init__(self, config_path):
        """Load the pipeline configuration."""
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        self.fitted = False
        self.scalers = {}
        
    def fit(self, X, y=None):
        """Fit the pipeline on training data."""
        X_transformed = X.copy()
        
        # Apply each transformation stage
        for transformer in self.config['transformers']:
            if transformer['name'] == 'missing_value_imputation':
                X_transformed = self._fit_missing_value_imputation(X_transformed, transformer)
            elif transformer['name'] == 'normalization':
                X_transformed = self._fit_normalization(X_transformed, transformer)
            elif transformer['name'] == 'cleaning_outliers':
                X_transformed = self._fit_outlier_cleaning(X_transformed, transformer)
            elif transformer['name'] == 'discretization':
                X_transformed = self._fit_discretization(X_transformed, transformer)
        
        self.fitted = True
        return X_transformed
    
    def transform(self, X):
        """Transform data using the fitted pipeline."""
        if not self.fitted:
            raise ValueError("Pipeline must be fitted before transform. Call fit() first.")
        
        X_transformed = X.copy()
        
        # Apply each transformation stage
        for transformer in self.config['transformers']:
            if transformer['name'] == 'missing_value_imputation':
                X_transformed = self._transform_missing_value_imputation(X_transformed, transformer)
            elif transformer['name'] == 'normalization':
                X_transformed = self._transform_normalization(X_transformed, transformer)
            elif transformer['name'] == 'cleaning_outliers':
                X_transformed = self._transform_outlier_cleaning(X_transformed, transformer)
            elif transformer['name'] == 'discretization':
                X_transformed = self._transform_discretization(X_transformed, transformer)
        
        return X_transformed
    
    def fit_transform(self, X, y=None):
        """Fit and transform in one step."""
        return self.fit(X, y)
    
    # Missing Value Imputation Methods
    def _fit_missing_value_imputation(self, X, transformer):
        """Fit missing value imputation."""
        X_result = X.copy()
        
        for feature_config in transformer['features']:
            feature_name = feature_config['feature_name']
            transformation = feature_config['transformation']
            
            if transformation == 'num_mv_identity':
                # Identity: do nothing (no missing values or keep as is)
                pass
            elif transformation == 'num_mv_mean':
                # Mean imputation
                if feature_name not in self.scalers:
                    self.scalers[feature_name] = {}
                self.scalers[feature_name]['mean'] = X_result[feature_name].mean()
                X_result[feature_name].fillna(self.scalers[feature_name]['mean'], inplace=True)
            elif transformation == 'num_mv_median':
                # Median imputation
                if feature_name not in self.scalers:
                    self.scalers[feature_name] = {}
                self.scalers[feature_name]['median'] = X_result[feature_name].median()
                X_result[feature_name].fillna(self.scalers[feature_name]['median'], inplace=True)
            elif transformation == 'num_mv_mode':
                # Mode imputation
                if feature_name not in self.scalers:
                    self.scalers[feature_name] = {}
                self.scalers[feature_name]['mode'] = X_result[feature_name].mode()[0]
                X_result[feature_name].fillna(self.scalers[feature_name]['mode'], inplace=True)
        
        return X_result
    
    def _transform_missing_value_imputation(self, X, transformer):
        """Transform with fitted missing value imputation."""
        X_result = X.copy()
        
        for feature_config in transformer['features']:
            feature_name = feature_config['feature_name']
            transformation = feature_config['transformation']
            
            if transformation == 'num_mv_identity':
                pass
            elif transformation in ['num_mv_mean', 'num_mv_median', 'num_mv_mode']:
                # Use fitted statistics
                stat_name = transformation.split('_')[-1]  # 'mean', 'median', or 'mode'
                if feature_name in self.scalers and stat_name in self.scalers[feature_name]:
                    X_result[feature_name].fillna(self.scalers[feature_name][stat_name], inplace=True)
        
        return X_result
    
    # Normalization Methods
    def _fit_normalization(self, X, transformer):
        """Fit normalization."""
        X_result = X.copy()
        
        # Get the transformation type (assumes all features use the same transformation)
        transformation = transformer['features'][0]['transformation']
        
        if transformation == 'identity':
            # No normalization
            pass
        elif transformation == 'ZS':
            # Z-score normalization (StandardScaler)
            numerical_cols = X_result.select_dtypes(include=[np.number]).columns
            scaler = StandardScaler()
            X_result[numerical_cols] = scaler.fit_transform(X_result[numerical_cols])
            self.scalers['normalization'] = scaler
        elif transformation == 'MM':
            # Min-Max normalization
            numerical_cols = X_result.select_dtypes(include=[np.number]).columns
            scaler = MinMaxScaler()
            X_result[numerical_cols] = scaler.fit_transform(X_result[numerical_cols])
            self.scalers['normalization'] = scaler
        elif transformation == 'robust':
            # Robust scaling
            numerical_cols = X_result.select_dtypes(include=[np.number]).columns
            scaler = RobustScaler()
            X_result[numerical_cols] = scaler.fit_transform(X_result[numerical_cols])
            self.scalers['normalization'] = scaler
        
        return X_result
    
    def _transform_normalization(self, X, transformer):
        """Transform with fitted normalization."""
        X_result = X.copy()
        
        transformation = transformer['features'][0]['transformation']
        
        if transformation == 'identity':
            pass
        elif transformation in ['ZS', 'MM', 'robust']:
            if 'normalization' in self.scalers:
                numerical_cols = X_result.select_dtypes(include=[np.number]).columns
                X_result[numerical_cols] = self.scalers['normalization'].transform(X_result[numerical_cols])
        
        return X_result
    
    # Outlier Cleaning Methods
    def _fit_outlier_cleaning(self, X, transformer):
        """Fit outlier cleaning."""
        X_result = X.copy()
        
        transformation = transformer['features'][0]['transformation']
        
        if transformation == 'identity':
            # No outlier cleaning
            pass
        elif transformation == 'clip':
            # Clip outliers using IQR method
            numerical_cols = X_result.select_dtypes(include=[np.number]).columns
            self.scalers['outlier_bounds'] = {}
            
            for col in numerical_cols:
                Q1 = X_result[col].quantile(0.25)
                Q3 = X_result[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                self.scalers['outlier_bounds'][col] = (lower_bound, upper_bound)
                X_result[col] = X_result[col].clip(lower=lower_bound, upper=upper_bound)
        
        return X_result
    
    def _transform_outlier_cleaning(self, X, transformer):
        """Transform with fitted outlier cleaning."""
        X_result = X.copy()
        
        transformation = transformer['features'][0]['transformation']
        
        if transformation == 'identity':
            pass
        elif transformation == 'clip':
            if 'outlier_bounds' in self.scalers:
                for col, (lower, upper) in self.scalers['outlier_bounds'].items():
                    if col in X_result.columns:
                        X_result[col] = X_result[col].clip(lower=lower, upper=upper)
        
        return X_result
    
    # Discretization Methods
    def _fit_discretization(self, X, transformer):
        """Fit discretization."""
        X_result = X.copy()
        
        transformation = transformer['features'][0]['transformation']
        
        if transformation == 'identity':
            # No discretization
            pass
        elif transformation == 'equal_width':
            # Equal-width binning
            numerical_cols = X_result.select_dtypes(include=[np.number]).columns
            self.scalers['discretization_bins'] = {}
            
            for col in numerical_cols:
                _, bins = pd.cut(X_result[col], bins=5, retbins=True)
                self.scalers['discretization_bins'][col] = bins
                X_result[col] = pd.cut(X_result[col], bins=bins, labels=False, include_lowest=True)
        elif transformation == 'equal_frequency':
            # Equal-frequency binning
            numerical_cols = X_result.select_dtypes(include=[np.number]).columns
            self.scalers['discretization_bins'] = {}
            
            for col in numerical_cols:
                _, bins = pd.qcut(X_result[col], q=5, retbins=True, duplicates='drop')
                self.scalers['discretization_bins'][col] = bins
                X_result[col] = pd.cut(X_result[col], bins=bins, labels=False, include_lowest=True)
        
        return X_result
    
    def _transform_discretization(self, X, transformer):
        """Transform with fitted discretization."""
        X_result = X.copy()
        
        transformation = transformer['features'][0]['transformation']
        
        if transformation == 'identity':
            pass
        elif transformation in ['equal_width', 'equal_frequency']:
            if 'discretization_bins' in self.scalers:
                for col, bins in self.scalers['discretization_bins'].items():
                    if col in X_result.columns:
                        X_result[col] = pd.cut(X_result[col], bins=bins, labels=False, include_lowest=True)
        
        return X_result


def main():
    parser = argparse.ArgumentParser(description='Apply saved preprocessing pipeline to dataset')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name (e.g., eeg, abalone)')
    parser.add_argument('--method', type=str, default='diffprep_fix', help='Method name (default: diffprep_fix)')
    parser.add_argument('--config_dir', type=str, default='saved_pipelines', help='Directory containing saved pipelines')
    parser.add_argument('--data_dir', type=str, default='data', help='Directory containing datasets')
    parser.add_argument('--output', type=str, help='Output file path (default: data/{dataset}/preprocessed_data.csv)')
    
    args = parser.parse_args()
    
    # Construct paths
    config_path = Path(args.config_dir) / args.method / args.dataset / 'pipeline_config.json'
    data_path = Path(args.data_dir) / args.dataset / 'data.csv'
    
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(args.data_dir) / args.dataset / 'preprocessed_data.csv'
    
    # Check if files exist
    if not config_path.exists():
        print(f"[ERROR] Configuration not found at: {config_path}")
        return
    
    if not data_path.exists():
        print(f"[ERROR] Data not found at: {data_path}")
        return
    
    print(f"[INFO] Loading configuration from: {config_path}")
    print(f"[INFO] Loading data from: {data_path}")
    
    # Load data
    df = pd.read_csv(data_path)
    print(f"\n[INFO] Data shape: {df.shape}")
    print(f"[INFO] Columns: {df.columns.tolist()}")
    
    # Separate features and target (assuming last column is target)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    print(f"\n[INFO] Features shape: {X.shape}")
    print(f"[INFO] Target shape: {y.shape}")
    
    # Create and apply pipeline
    pipeline = PreprocessingPipeline(config_path)
    
    print("\n[INFO] Applying preprocessing pipeline...")
    X_transformed = pipeline.fit_transform(X)
    
    # Combine with target
    df_transformed = pd.concat([X_transformed, y], axis=1)
    
    # Save result
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_transformed.to_csv(output_path, index=False)
    
    print(f"\n[SUCCESS] Preprocessed data saved to: {output_path}")
    print(f"[INFO] Output shape: {df_transformed.shape}")
    
    # Show statistics
    print("\n" + "="*80)
    print("PREPROCESSING SUMMARY")
    print("="*80)
    
    print("\nOriginal data statistics:")
    print(df.describe())
    
    print("\nPreprocessed data statistics:")
    print(df_transformed.describe())


if __name__ == '__main__':
    main()
