"""
Script to fetch datasets from OpenML and save them in the data folder
with the same format as existing datasets (data.csv and info.json)
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle


def load_openml_dataset(dataset_id, test_dataset_ids=None):
    """Load OpenML dataset with error handling and automatic problem type detection"""
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
        max_samples = 100000 if (test_dataset_ids and dataset_id in test_dataset_ids) else 5000
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
            'name': dataset.details.get('name', f"Dataset_{dataset_id}"),
            'X': X,
            'y': y,
            'task_type': task_type
        }
    except Exception as e:
        print(f"Failed to load dataset {dataset_id}: {e}")
        return None


def calculate_dataset_stats(X, y, task_type):
    """Calculate statistics about the dataset"""
    
    # Count missing values
    num_missing = X.isna().sum().sum()
    total_values = X.shape[0] * X.shape[1]
    missing_percentage = round((num_missing / total_values) * 100, 2) if total_values > 0 else 0.0
    
    # Count outliers using IQR method (only for numeric columns)
    num_outliers = 0
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        for col in numeric_cols:
            if X[col].notna().sum() > 0:  # Only if column has non-null values
                Q1 = X[col].quantile(0.25)
                Q3 = X[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = ((X[col] < (Q1 - 1.5 * IQR)) | (X[col] > (Q3 + 1.5 * IQR))) & X[col].notna()
                num_outliers += outliers.sum()
    
    outlier_percentage = round((num_outliers / total_values) * 100, 2) if total_values > 0 else 0.0
    
    # Number of classes (only for classification)
    num_classes = len(np.unique(y)) if task_type == 'classification' else None
    
    return {
        'num_samples': int(X.shape[0]),
        'num_features': int(X.shape[1]),
        'num_classes': int(num_classes) if num_classes is not None else None,
        'num_missing_values': int(num_missing),
        'missing_percentage': float(missing_percentage),
        'num_outlier_cells': int(num_outliers),
        'outlier_percentage': float(outlier_percentage)
    }


def save_dataset(dataset_info, save_dir='data'):
    """
    Save dataset in the format matching existing datasets in the data folder
    
    Args:
        dataset_info: Dictionary containing 'id', 'name', 'X', 'y', 'task_type'
        save_dir: Base directory to save datasets (default: 'data')
    """
    dataset_id = dataset_info['id']
    dataset_name = dataset_info['name']
    X = dataset_info['X']
    y = dataset_info['y']
    task_type = dataset_info['task_type']
    
    # Create directory for this dataset
    dataset_dir = os.path.join(save_dir, str(dataset_id))
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Combine X and y into a single DataFrame with 'label' column for target
    data_df = X.copy()
    data_df['label'] = y
    
    # Save data.csv
    csv_path = os.path.join(dataset_dir, 'data.csv')
    data_df.to_csv(csv_path, index=False)
    print(f"  Saved data.csv to {csv_path}")
    
    # Calculate statistics
    stats = calculate_dataset_stats(X, y, task_type)
    
    # Create info.json
    info = {
        'label': 'label',
        'dataset_name': dataset_name,
        'task_type': task_type,
        'openml_id': int(dataset_id),
        'num_samples': stats['num_samples'],
        'num_features': stats['num_features'],
    }
    
    if task_type == 'classification':
        info['num_classes'] = stats['num_classes']
    
    info.update({
        'num_missing_values': stats['num_missing_values'],
        'missing_percentage': stats['missing_percentage'],
        'num_outlier_cells': stats['num_outlier_cells'],
        'outlier_percentage': stats['outlier_percentage']
    })
    
    # Save info.json
    info_path = os.path.join(dataset_dir, 'info.json')
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    print(f"  Saved info.json to {info_path}")
    
    return dataset_dir


def fetch_and_save_datasets(dataset_ids, save_dir='data', test_dataset_ids=None):
    """
    Fetch multiple datasets from OpenML and save them
    
    Args:
        dataset_ids: List of OpenML dataset IDs to fetch
        save_dir: Directory to save datasets (default: 'data')
        test_dataset_ids: Optional list of dataset IDs that should have max 100k samples
    """
    print(f"Fetching {len(dataset_ids)} datasets from OpenML...")
    print("=" * 60)
    
    successful = []
    failed = []
    
    for dataset_id in dataset_ids:
        print(f"\nProcessing dataset {dataset_id}...")
        
        # Check if data.csv already exists
        dataset_dir = os.path.join(save_dir, str(dataset_id))
        csv_path = os.path.join(dataset_dir, 'data.csv')
        
        if os.path.exists(csv_path):
            print(f"  data.csv exists. Regenerating info.json...")
            try:
                # Load existing data
                data_df = pd.read_csv(csv_path)
                X = data_df.drop(columns=['label'])
                y = data_df['label']
                
                # Load existing info to get metadata
                info_path = os.path.join(dataset_dir, 'info.json')
                if os.path.exists(info_path):
                    with open(info_path, 'r') as f:
                        info = json.load(f)
                    dataset_name = info.get('dataset_name', str(dataset_id))
                    task_type = info.get('task_type', 'classification')
                else:
                    dataset_name = str(dataset_id)
                    task_type = 'classification'
                
                # Create dataset_info dict
                dataset_info = {
                    'id': dataset_id,
                    'name': dataset_name,
                    'X': X,
                    'y': y,
                    'task_type': task_type
                }
                
                # Regenerate info.json with correct formulas
                save_dataset(dataset_info, save_dir)
                successful.append(dataset_id)
                print(f"  Successfully regenerated info.json")
            except Exception as e:
                print(f"  Failed to regenerate info.json: {e}")
                failed.append(dataset_id)
            continue
        
        # Load dataset from OpenML
        dataset_info = load_openml_dataset(dataset_id, test_dataset_ids)
        
        if dataset_info is None:
            failed.append(dataset_id)
            continue
        
        # Save dataset
        try:
            saved_dir = save_dataset(dataset_info, save_dir)
            successful.append(dataset_id)
            print(f"  Successfully saved to {saved_dir}")
        except Exception as e:
            print(f"  Failed to save dataset {dataset_id}: {e}")
            failed.append(dataset_id)
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Successfully processed: {len(successful)}/{len(dataset_ids)}")
    print(f"Failed: {len(failed)}/{len(dataset_ids)}")
    
    if successful:
        print(f"\nSuccessful dataset IDs: {successful}")
    if failed:
        print(f"\nFailed dataset IDs: {failed}")
    
    return successful, failed


if __name__ == "__main__":
    # Example usage: Fetch some sample datasets
    # You can modify this list to include the dataset IDs you want to fetch
    
    # Example dataset IDs (you can add more)
    dataset_ids_to_fetch = [
        40975, 1233, 1115, 1466, 248, 279, 40740, 803, 942, 373, 
        1518, 737, 1396, 1399, 823, 253, 922, 7, 1066, 1164, 932,
        974, 1047, 991, 244, 1400, 862, 40520, 2, 40663, 1054,
        1387, 1397, 1401, 1393, 728, 876, 1358, 75, 18
    ]
    
    # Or fetch specific test datasets with larger sample limit
    test_dataset_ids = [
        40975, 1233, 1115, 1466, 248, 279, 40740, 803, 942, 373, 
        1518, 737, 1396, 1399, 823, 253, 922, 7, 1066, 1164, 932,
        974, 1047, 991, 244, 1400, 862, 40520, 2, 40663, 1054,
        1387, 1397, 1401, 1393, 728, 876, 1358, 75, 18    
    ]
    
    if not dataset_ids_to_fetch:
        print("No dataset IDs specified.")
        print("\nUsage example:")
        print("  1. Edit this script and add dataset IDs to 'dataset_ids_to_fetch' list")
        print("  2. Run: python fetch_and_save_openml_datasets.py")
        print("\nAlternatively, use the function in your code:")
        print("  from fetch_and_save_openml_datasets import fetch_and_save_datasets")
        print("  fetch_and_save_datasets([31, 3, 1485])")
    else:
        # Fetch and save datasets
        fetch_and_save_datasets(
            dataset_ids_to_fetch, 
            save_dir='data',
            test_dataset_ids=test_dataset_ids
        )
