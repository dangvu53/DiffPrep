"""
Evaluate saved preprocessing pipeline using AutoGluon on the test split.

Usage:
    # First, extract and save the best pipeline:
    python extract_and_save_pipeline.py --dataset abalone --method diffprep_fix
    
    # Then evaluate with AutoGluon:
    python evaluate_with_autogluon_v2.py --dataset abalone --method diffprep_fix
"""

import argparse
import os
import json
import pandas as pd
import numpy as np
import tempfile
import shutil
import pickle
import torch
from autogluon.tabular import TabularPredictor


def load_saved_pipeline(saved_pipeline_dir, dataset, method):
    """Load a previously saved pipeline"""
    pipeline_dir = os.path.join(saved_pipeline_dir, method, dataset)
    
    # Load metadata
    metadata_path = os.path.join(pipeline_dir, 'metadata.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Load pipeline
    pipeline_path = os.path.join(pipeline_dir, 'pipeline.pkl')
    with open(pipeline_path, 'rb') as f:
        pipeline = pickle.load(f)
    
    # Load data splits
    data_path = os.path.join(pipeline_dir, 'data_splits.pkl')
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    print("="*60)
    print(f"Loaded {method} pipeline for {dataset}")
    print("="*60)
    print(f"Original experiment:")
    print(f"  Best epoch: {metadata['best_epoch']}")
    print(f"  Test Acc: {metadata['original_test_acc']:.4f}")
    
    return pipeline, data, metadata


def preprocess_data(pipeline, X_train, X_val, X_test):
    """Transform data using the pipeline"""
    print("\nTransforming data using learned pipeline...")
    
    # Fit the pipeline on training data first (this initializes internal transformers)
    if not pipeline.is_fitted:
        print("Fitting pipeline...")
        pipeline.fit(X_train)
    
    # Use the pipeline's transform method with max_only=True to get deterministic output
    with torch.no_grad():
        X_train_transformed = pipeline.transform(X_train, X_type='train', max_only=True, resample=False)
        X_val_transformed = pipeline.transform(X_val, X_type='val', max_only=True, resample=False)
        X_test_transformed = pipeline.transform(X_test, X_type='test', max_only=True, resample=False)
    
    # Convert back to pandas
    X_train_df = pd.DataFrame(X_train_transformed.numpy())
    X_val_df = pd.DataFrame(X_val_transformed.numpy())
    X_test_df = pd.DataFrame(X_test_transformed.numpy())
    
    print(f"✓ Transformed train: {X_train_df.shape}")
    print(f"✓ Transformed val: {X_val_df.shape}")
    print(f"✓ Transformed test: {X_test_df.shape}")
    
    return X_train_df, X_val_df, X_test_df


def train_autogluon(X_train, y_train, X_val, y_val, X_test, y_test, time_limit, eval_metric='accuracy'):
    """Train AutoGluon on preprocessed data"""
    print(f"\nTraining AutoGluon (time limit: {time_limit}s)...")
    
    # Convert tensors to numpy if needed
    if isinstance(y_train, torch.Tensor):
        y_train = y_train.numpy()
    if isinstance(y_val, torch.Tensor):
        y_val = y_val.numpy()
    if isinstance(y_test, torch.Tensor):
        y_test = y_test.numpy()
    
    # Combine train and val for AutoGluon
    X_train_combined = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
    y_train_combined = np.concatenate([y_train, y_val])
    
    # Create training DataFrame
    train_data = X_train_combined.copy()
    train_data['label'] = y_train_combined
    
    # Create test DataFrame
    test_data = X_test.copy()
    test_data['label'] = y_test
    
    # Ensure parent directory exists
    os.makedirs('D:/temp_autogluon', exist_ok=True)
    
    # Create temp directory
    temp_dir = tempfile.mkdtemp(prefix='autogluon_', dir='D:/temp_autogluon')
    
    try:
        # Create AutoGluon predictor
        predictor = TabularPredictor(
            label='label',
            eval_metric=eval_metric,
            path=temp_dir,
            verbosity=0,
        )
        
        # Train
        predictor.fit(
            train_data=train_data,
            time_limit=time_limit,
            presets='best_quality',
            verbosity=0
        )
        
        # Evaluate on test set
        test_results = predictor.evaluate(test_data)
        print(f"\nAutoGluon Test Accuracy: {test_results['accuracy']:.4f}")
        
        # Get leaderboard
        leaderboard = predictor.leaderboard(test_data, silent=True)
        
        return test_results, leaderboard, predictor
        
    finally:
        # Clean up temp directory
        try:
            shutil.rmtree(temp_dir)
        except:
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--method', required=True, choices=['diffprep_fix', 'diffprep_flex', 'default', 'random'])
    parser.add_argument('--saved_pipeline_dir', default='saved_pipelines')
    parser.add_argument('--output_dir', default='autogluon_results')
    parser.add_argument('--time_limit', type=int, default=300, help='AutoGluon training time limit in seconds')
    args = parser.parse_args()
    
    # Load saved pipeline
    pipeline, data, metadata = load_saved_pipeline(
        args.saved_pipeline_dir,
        args.dataset,
        args.method
    )
    
    # Extract data
    X_train = data['X_train']
    y_train = data['y_train']
    X_val = data['X_val']
    y_val = data['y_val']
    X_test = data['X_test']
    y_test = data['y_test']
    
    # Preprocess data
    X_train_transformed, X_val_transformed, X_test_transformed = preprocess_data(
        pipeline, X_train, X_val, X_test
    )
    
    # Train AutoGluon
    test_results, leaderboard, predictor = train_autogluon(
        X_train_transformed, y_train,
        X_val_transformed, y_val,
        X_test_transformed, y_test,
        args.time_limit
    )
    
    # Save results
    output_dir = os.path.join(args.output_dir, args.method, args.dataset)
    os.makedirs(output_dir, exist_ok=True)
    
    results = {
        'dataset': args.dataset,
        'method': args.method,
        'original_test_acc': metadata['original_test_acc'],
        'autogluon_test_acc': test_results['accuracy'],
        'improvement': test_results['accuracy'] - metadata['original_test_acc']
    }
    
    results_path = os.path.join(output_dir, 'autogluon_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    leaderboard_path = os.path.join(output_dir, 'leaderboard.csv')
    leaderboard.to_csv(leaderboard_path, index=False)
    
    print("\n" + "="*60)
    print("AUTOGLUON EVALUATION COMPLETE")
    print("="*60)
    print(f"Original Test Acc: {metadata['original_test_acc']:.4f}")
    print(f"AutoGluon Test Acc: {test_results['accuracy']:.4f}")
    print(f"Improvement: {results['improvement']:.4f}")
    print(f"\nResults saved to: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
