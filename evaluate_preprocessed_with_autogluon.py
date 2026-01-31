"""
Evaluate preprocessed data with AutoGluon using the same split method as the repo.

Usage:
    python evaluate_preprocessed_with_autogluon.py --dataset eeg --method diffprep_fix
"""

import argparse
import pandas as pd
import numpy as np
import os
import json
from pathlib import Path
import sys
import tempfile
import shutil

# Add experiment directory to path
sys.path.append('experiment')
from experiment_utils import split

from autogluon.tabular import TabularPredictor, FeatureMetadata
from autogluon.tabular.configs.hyperparameter_configs import get_hyperparameter_config
from autogluon.features.generators import IdentityFeatureGenerator


def load_preprocessed_data(dataset, data_dir='data'):
    """Load the preprocessed data."""
    preprocessed_path = Path(data_dir) / dataset / 'preprocessed_data.csv'
    
    if not preprocessed_path.exists():
        raise FileNotFoundError(f"Preprocessed data not found at: {preprocessed_path}")
    
    df = pd.read_csv(preprocessed_path)
    return df


def load_dataset_info(dataset, data_dir='data'):
    """Load dataset info to get the label column name."""
    info_path = Path(data_dir) / dataset / 'info.json'
    
    with open(info_path, 'r') as f:
        info = json.load(f)
    
    return info


def evaluate_with_autogluon(dataset, method='diffprep_fix', time_limit=600, random_state=1):
    """
    Evaluate preprocessed data with AutoGluon.
    
    Args:
        dataset: Dataset name (e.g., 'eeg')
        method: Preprocessing method name (e.g., 'diffprep_fix')
        time_limit: Training time limit in seconds
        random_state: Random seed for reproducibility
    """
    print("="*80)
    print(f"AUTOGLUON EVALUATION - {dataset.upper()} ({method})")
    print("="*80)
    
    # Load dataset info
    info = load_dataset_info(dataset)
    label_column = info['label']
    
    print(f"\n[INFO] Label column: {label_column}")
    
    # Load preprocessed data
    print(f"[INFO] Loading preprocessed data...")
    df = load_preprocessed_data(dataset)
    print(f"[INFO] Data shape: {df.shape}")
    
    # Separate features and target
    feature_columns = [c for c in df.columns if c != label_column]
    X = df[feature_columns]
    y = df[label_column]
    
    print(f"[INFO] Features: {X.shape}")
    print(f"[INFO] Target: {y.shape}")
    print(f"[INFO] Target classes: {sorted(y.unique())}")
    
    # Split data using repo's split method (60/20/20)
    print(f"\n[INFO] Splitting data (train/val/test = 60/20/20)...")
    X_train, y_train, X_val, y_val, X_test, y_test = split(
        X, y, val_ratio=0.2, test_ratio=0.2, random_state=random_state
    )
    
    print(f"[INFO] Train size: {len(X_train)}")
    print(f"[INFO] Val size: {len(X_val)}")
    print(f"[INFO] Test size: {len(X_test)}")
    
    # Combine features and target for AutoGluon
    train_data = pd.concat([X_train.reset_index(drop=True), 
                           pd.Series(y_train, name=label_column).reset_index(drop=True)], axis=1)
    val_data = pd.concat([X_val.reset_index(drop=True), 
                         pd.Series(y_val, name=label_column).reset_index(drop=True)], axis=1)
    test_data = pd.concat([X_test.reset_index(drop=True), 
                          pd.Series(y_test, name=label_column).reset_index(drop=True)], axis=1)
    
    # Create output directory for results
    output_dir = f'autogluon_results/{method}/{dataset}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Create temp directory for AutoGluon training (to avoid "already fit" error)
    os.makedirs('D:/temp_autogluon', exist_ok=True)
    temp_dir = tempfile.mkdtemp(prefix='autogluon_', dir='D:/temp_autogluon')
    
    try:
        # Train AutoGluon
        print(f"\n[INFO] Training AutoGluon (time_limit={time_limit}s)...")
        print(f"[INFO] Using IdentityFeatureGenerator (no additional feature engineering)...")
        print(f"[INFO] Temporary training directory: {temp_dir}")
        
        predictor = TabularPredictor(
            label=label_column,
            path=temp_dir,
            eval_metric='accuracy',
            verbosity=2
        )
        predictor.fit(
            train_data=train_data,
            # Not using tuning_data to avoid "Learner is already fit" error with DyStack
            # tuning_data=val_data,
            time_limit=time_limit,
            presets='best_quality',
            feature_generator=IdentityFeatureGenerator(),  # Disable AutoGluon's feature engineering
            ag_args_fit={'num_gpus': 0}  # Use CPU
        )
        
        # Evaluate on validation set
        print("\n" + "="*80)
        print("VALIDATION SET RESULTS")
        print("="*80)
        val_pred = predictor.predict(val_data)
        val_accuracy = (val_pred == val_data[label_column]).mean()
        print(f"Validation Accuracy: {val_accuracy:.4f}")
        
        # Get detailed metrics
        val_metrics = predictor.evaluate(val_data, detailed_report=True)
        print("\nDetailed validation metrics:")
        for metric, value in val_metrics.items():
            if isinstance(value, (int, float)):
                print(f"  {metric}: {value:.4f}")
        
        # Evaluate on test set
        print("\n" + "="*80)
        print("TEST SET RESULTS")
        print("="*80)
        test_pred = predictor.predict(test_data)
        test_accuracy = (test_pred == test_data[label_column]).mean()
        print(f"Test Accuracy: {test_accuracy:.4f}")
        
        # Get detailed metrics
        test_metrics = predictor.evaluate(test_data, detailed_report=True)
        print("\nDetailed test metrics:")
        for metric, value in test_metrics.items():
            if isinstance(value, (int, float)):
                print(f"  {metric}: {value:.4f}")
        
        # Get model leaderboard
        print("\n" + "="*80)
        print("MODEL LEADERBOARD")
        print("="*80)
        leaderboard = predictor.leaderboard(test_data, silent=True)
        print(leaderboard.to_string())
        
        # Save results
        results = {
            'dataset': dataset,
            'method': method,
            'random_state': random_state,
            'data_shape': df.shape,
            'train_size': len(X_train),
            'val_size': len(X_val),
            'test_size': len(X_test),
            'val_accuracy': float(val_accuracy),
            'test_accuracy': float(test_accuracy),
            'val_metrics': {k: float(v) if isinstance(v, (int, float, np.number)) else str(v) 
                            for k, v in val_metrics.items()},
            'test_metrics': {k: float(v) if isinstance(v, (int, float, np.number)) else str(v) 
                             for k, v in test_metrics.items()},
            'best_model': predictor.model_best
        }
        
        results_path = os.path.join(output_dir, 'evaluation_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n[SUCCESS] Results saved to: {results_path}")
        
        # Save leaderboard
        leaderboard_path = os.path.join(output_dir, 'leaderboard.csv')
        leaderboard.to_csv(leaderboard_path)
        print(f"[SUCCESS] Leaderboard saved to: {leaderboard_path}")
        
        return results
        
    finally:
        # Clean up temp directory
        print(f"\n[INFO] Cleaning up temporary directory: {temp_dir}")
        try:
            shutil.rmtree(temp_dir)
        except:
            pass


def main():
    parser = argparse.ArgumentParser(description='Evaluate preprocessed data with AutoGluon')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name (e.g., eeg)')
    parser.add_argument('--method', type=str, default='diffprep_fix', help='Preprocessing method')
    parser.add_argument('--time_limit', type=int, default=300, help='Training time limit in seconds')
    parser.add_argument('--random_state', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    try:
        results = evaluate_with_autogluon(
            dataset=args.dataset,
            method=args.method,
            time_limit=args.time_limit,
            random_state=args.random_state
        )
        
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print(f"Dataset: {results['dataset']}")
        print(f"Method: {results['method']}")
        print(f"Validation Accuracy: {results['val_accuracy']:.4f}")
        print(f"Test Accuracy: {results['test_accuracy']:.4f}")
        print(f"Best Model: {results['best_model']}")
        
    except Exception as e:
        print(f"\n[ERROR] Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
