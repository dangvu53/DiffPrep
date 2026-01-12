"""
Script to download datasets from OpenML and save them in the data folder.
Usage: python download_openml_dataset.py <dataset_id> [dataset_name]
Example: python download_openml_dataset.py 44 credit-g
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle


def load_openml_dataset(dataset_id, test_dataset_ids=None, max_samples=5000):
    """Load OpenML dataset with error handling and automatic problem type detection
    
    Args:
        dataset_id: OpenML dataset ID
        test_dataset_ids: List of dataset IDs to use larger sample size
        max_samples: Maximum number of samples to keep (None for full dataset)
    """
    try:
        try:
            dataset = fetch_openml(data_id=dataset_id, as_frame=True, parser='auto')
        except ValueError as e:
            if "Sparse ARFF" in str(e):
                print(f"Retrying dataset {dataset_id} with as_frame=False...")
                dataset = fetch_openml(data_id=dataset_id, as_frame=False, parser='auto')
            else:
                raise e

        X = dataset.data.copy()
        y = dataset.target
        dataset_name = dataset.details.get('name', f'dataset_{dataset_id}')

        # Handle categorical features properly
        if isinstance(X, pd.DataFrame):
            for col in X.select_dtypes(include=['object', 'category']).columns:
                X[col] = X[col].astype(str)

        # Handle target encoding
        if y.dtype == 'object' or y.dtype.name == 'category':
            le = LabelEncoder()
            y = pd.Series(le.fit_transform(y), index=y.index)

        # Drop invalid samples
        X = X.dropna(axis=1, how='all')
        mask = ~pd.isna(y)
        X = X[mask].reset_index(drop=True)
        y = y[mask].reset_index(drop=True)

        # Detect problem type
        if y.nunique() > 50 and y.dtype.kind in "iufc":
            task_type = 'regression'
        else:
            task_type = 'classification'
            y = y.astype(int)

        # Remove rare classes (<5 samples)
        if task_type == 'classification':
            class_counts = y.value_counts()
            valid_classes = class_counts[class_counts >= 5].index
            mask = y.isin(valid_classes)
            X = X[mask].reset_index(drop=True)
            y = y[mask].reset_index(drop=True)

        # Limit dataset size for efficiency
        if max_samples is not None:
            if test_dataset_ids and dataset_id in test_dataset_ids:
                max_samples = 100000
            if len(X) > max_samples:
                X, y = shuffle(X, y, n_samples=max_samples, random_state=42)
                X = X.reset_index(drop=True)
                y = pd.Series(y).reset_index(drop=True)

        print(f"Loaded dataset {dataset_id}")
        print(f"  Shape: {X.shape}")
        print(f"  Task: {task_type}")
        print(f"  Target classes: {len(np.unique(y)) if task_type=='classification' else 'N/A'}")

        return {
            'id': dataset_id,
            'name': dataset_name,
            'X': X,
            'y': y,
            'task_type': task_type
        }
    except Exception as e:
        print(f"Failed to load dataset {dataset_id}: {e}")
        return None


def save_dataset(dataset_data, dataset_name, data_folder='data'):
    """Save dataset to data folder with data.csv and info.json"""
    if dataset_data is None:
        print("No dataset data to save.")
        return False
    
    # Create folder for dataset
    dataset_folder = os.path.join(data_folder, dataset_name)
    os.makedirs(dataset_folder, exist_ok=True)
    
    X = dataset_data['X']
    y = dataset_data['y']
    task_type = dataset_data['task_type']
    
    # Combine X and y into a single DataFrame
    # Use 'label' as the target column name (consistent with existing datasets)
    label_column = 'label'
    
    if isinstance(X, pd.DataFrame):
        df = X.copy()
    else:
        df = pd.DataFrame(X)
    
    df[label_column] = y
    
    # Calculate missing values (before saving CSV)
    num_missing_values = int(df.isnull().sum().sum())
    missing_percentage = (num_missing_values / (df.shape[0] * df.shape[1])) * 100
    
    # Calculate outliers using z-score for numeric columns (excluding label)
    # Count samples (rows) that have at least one outlier value, not individual outlier values
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if label_column in numeric_cols:
        numeric_cols.remove(label_column)
    
    num_outlier_samples = 0
    if len(numeric_cols) > 0:
        # Create boolean mask for samples with outliers (using Z-score > 3, same as ZSOutlierDetector with nstd=3)
        has_outlier = np.zeros(len(df), dtype=bool)
        for col in numeric_cols:
            col_data = df[col].values
            valid_mask = ~np.isnan(col_data)
            if valid_mask.sum() > 0:
                mean = np.nanmean(col_data)
                std = np.nanstd(col_data)
                if std > 0:
                    z_scores = np.abs((col_data - mean) / std)
                    # Mark rows with outliers (z-score > 3)
                    has_outlier = has_outlier | (z_scores > 3)
        
        num_outlier_samples = int(has_outlier.sum())
    
    outlier_percentage = (num_outlier_samples / df.shape[0]) * 100 if df.shape[0] > 0 else 0
    
    # Save data.csv
    csv_path = os.path.join(dataset_folder, 'data.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved data.csv with {len(df)} rows and {len(df.columns)} columns")
    
    # Create info.json
    info = {
        'label': label_column,
        'dataset_name': dataset_data['name'],
        'task_type': task_type,
        'openml_id': dataset_data['id'],
        'num_samples': len(df),
        'num_features': len(df.columns) - 1,  # Exclude label
        'num_classes': int(y.nunique()) if task_type == 'classification' else None,
        'num_missing_values': num_missing_values,
        'missing_percentage': round(missing_percentage, 2),
        'num_outlier_samples': num_outlier_samples,
        'outlier_percentage': round(outlier_percentage, 2)
    }
    
    info_path = os.path.join(dataset_folder, 'info.json')
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    print(f"Saved info.json")
    print(f"  - Missing values: {num_missing_values} ({missing_percentage:.2f}%)")
    print(f"  - Outlier samples (z-score > 3): {num_outlier_samples} ({outlier_percentage:.2f}%)")
    
    print(f"\nDataset saved successfully to: {dataset_folder}")
    print(f"  - {csv_path}")
    print(f"  - {info_path}")
    
    return True


def main():
    if len(sys.argv) < 2:
        print("Usage: python download_openml_dataset.py <dataset_id> [--full]")
        print("Example: python download_openml_dataset.py 44")
        print("Example: python download_openml_dataset.py 44 --full  (download full dataset)")
        sys.exit(1)
    
    dataset_id = int(sys.argv[1])
    
    # Check if --full flag is present
    full_dataset = '--full' in sys.argv
    max_samples = None if full_dataset else 5000
    
    # Use dataset ID as folder name
    folder_name = str(dataset_id)
    
    print(f"Downloading OpenML dataset {dataset_id}...")
    print(f"Dataset will be saved in folder: {folder_name}")
    print(f"Mode: {'Full dataset' if full_dataset else f'Limited to {max_samples} samples'}")
    print("-" * 60)
    
    # Load dataset from OpenML
    dataset_data = load_openml_dataset(dataset_id, max_samples=max_samples)
    
    if dataset_data:
        print(f"Dataset name: {dataset_data['name']}")
        # Save to data folder
        success = save_dataset(dataset_data, folder_name)
        if success:
            print("\n" + "=" * 60)
            print("SUCCESS! Dataset is ready to use.")
            print("=" * 60)
        else:
            print("\nFailed to save dataset.")
            sys.exit(1)
    else:
        print("\nFailed to load dataset from OpenML.")
        sys.exit(1)


if __name__ == "__main__":
    main()
