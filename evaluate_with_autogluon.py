"""
Evaluate best preprocessing pipeline using AutoGluon
After finding the best pipeline, transform the data and train AutoGluon on it.
"""

import argparse
import os
import json
import torch
import pandas as pd
import numpy as np
import tempfile
import shutil
from sklearn.preprocessing import LabelEncoder
from autogluon.tabular import TabularPredictor
from autogluon.features.generators import IdentityFeatureGenerator

from prep_space_new import space  # Using new PreprocessorOps operators
from experiment.experiment_utils import load_data, build_data, min_max_normalize
from pipeline.diffprep_fix_pipeline import DiffPrepFixPipeline
from pipeline.diffprep_flex_pipeline import DiffPrepFlexPipeline

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='Dataset name')
parser.add_argument('--data_dir', default="data")
parser.add_argument('--result_dir', default="result")
parser.add_argument('--method', default="diffprep_fix", choices=["diffprep_fix", "diffprep_flex"])
parser.add_argument('--split_seed', default=1, type=int)
parser.add_argument('--time_limit', default=600, type=int, help='Time limit for AutoGluon training (seconds)')
parser.add_argument('--output_dir', default="autogluon_results", help='Directory to save AutoGluon results')
args = parser.parse_args()


def load_best_pipeline(result_dir, dataset, method, prep_space):
    """Load the best preprocessing pipeline from saved results"""
    pipeline_dir = os.path.join(result_dir, method, dataset)
    
    # Load parameters
    params_path = os.path.join(pipeline_dir, "params.json")
    with open(params_path, 'r') as f:
        params = json.load(f)
    
    # Load results to see best performance
    result_path = os.path.join(pipeline_dir, "result.json")
    with open(result_path, 'r') as f:
        result = json.load(f)
    
    print(f"Best pipeline found at epoch {result['best_epoch']}")
    print(f"  Val Acc: {result['best_val_acc']:.4f}")
    print(f"  Test Acc (DiffPrep): {result['best_test_acc']:.4f}")
    
    # Initialize pipeline
    if method == "diffprep_fix":
        prep_pipeline = DiffPrepFixPipeline(
            prep_space, 
            temperature=params["temperature"],
            use_sample=params["sample"],
            diff_method=params["diff_method"],
            init_method=params["init_method"]
        )
    elif method == "diffprep_flex":
        prep_pipeline = DiffPrepFlexPipeline(
            prep_space,
            temperature=params["temperature"],
            use_sample=params["sample"],
            diff_method=params["diff_method"],
            init_method=params["init_method"]
        )
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return prep_pipeline, params, result


def transform_data_with_pipeline(prep_pipeline, X_train, X_val, X_test, params, result_dir, dataset, method):
    """Transform data using the best preprocessing pipeline"""
    print("\nTransforming data with best pipeline...")
    
    # First, initialize pipeline parameters to create the correct structure
    print("Initializing pipeline structure...")
    prep_pipeline.init_parameters(X_train, X_val, X_test)
    
    # Now load saved pipeline weights (after structure is created)
    pipeline_path = os.path.join(result_dir, method, dataset, "prep_pipeline.pth")
    if os.path.exists(pipeline_path):
        print(f"Loading pipeline weights from {pipeline_path}")
        prep_pipeline.load_state_dict(torch.load(pipeline_path))
        print("Pipeline weights loaded successfully!")
    else:
        print(f"Warning: No saved pipeline weights found at {pipeline_path}")
        print("Using initialized pipeline with default parameters.")
    
    # Fit pipeline on training data (to compute statistics like mean, std, etc.)
    print("Fitting pipeline on training data...")
    prep_pipeline.fit(X_train)
    
    # Transform all datasets using the learned transformation probabilities
    # Use max_only=True to select the most probable transformation
    print("Transforming train data...")
    X_train_trans = prep_pipeline.transform(X_train, X_type='train', max_only=True, resample=False, require_grad=False)
    print("Transforming val data...")
    X_val_trans = prep_pipeline.transform(X_val, X_type='val', max_only=True, resample=False, require_grad=False)
    print("Transforming test data...")
    X_test_trans = prep_pipeline.transform(X_test, X_type='test', max_only=True, resample=False, require_grad=False)
    
    # Convert to numpy and then to DataFrame
    X_train_trans = pd.DataFrame(X_train_trans.detach().numpy())
    X_val_trans = pd.DataFrame(X_val_trans.detach().numpy())
    X_test_trans = pd.DataFrame(X_test_trans.detach().numpy())
    
    print(f"Transformed shapes: Train {X_train_trans.shape}, Val {X_val_trans.shape}, Test {X_test_trans.shape}")
    
    return X_train_trans, X_val_trans, X_test_trans


def train_autogluon(X_train, y_train, X_val, y_val, X_test, y_test, time_limit):
    """Train AutoGluon on preprocessed data"""
    print(f"\nTraining AutoGluon (time limit: {time_limit}s)...")
    
    # Combine train and val for AutoGluon (it will split internally)
    X_train_combined = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
    y_train_combined = torch.cat([y_train, y_val], dim=0)
    
    # Add label column
    X_train_combined['label'] = y_train_combined.numpy()
    X_test_df = X_test.copy()
    X_test_df['label'] = y_test.numpy()
    
    # Use temporary directory on D: drive to avoid cross-drive issues with DyStack
    # AutoGluon's dynamic stacking fails when temp is on C: but data is on D:
    # Ensure parent directory exists FIRST
    os.makedirs('D:/temp_autogluon', exist_ok=True)
    
    # Now create temp directory
    temp_dir = tempfile.mkdtemp(prefix='autogluon_', dir='D:/temp_autogluon')
    
    try:
        # Create AutoGluon predictor with fresh path
        predictor = TabularPredictor(
            label='label',
            problem_type='multiclass',
            eval_metric='accuracy',
            path=temp_dir,
            verbosity=0
        )
        
        # Train - disable dynamic_stacking to avoid cross-drive path issues
        predictor.fit(
            train_data=X_train_combined,
            time_limit=time_limit,
            presets='best_quality',
            verbosity=0,
            dynamic_stacking=False,  # Disable to avoid cross-drive path errors
            feature_generator=IdentityFeatureGenerator()
        )
        
        # Evaluate on test set
        test_acc = predictor.evaluate(X_test_df)
        print(f"\nAutoGluon Test Accuracy: {test_acc['accuracy']:.4f}")
        
        # Get leaderboard
        leaderboard = predictor.leaderboard(X_test_df, silent=True)
        print("\nModel Leaderboard:")
        print(leaderboard)
        
        # Extract leaderboard as dict for saving
        leaderboard_dict = leaderboard.to_dict('records')
        
        return test_acc, leaderboard_dict
        
    finally:
        # Clean up temporary directory
        try:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
                print(f"\nCleaned up temporary AutoGluon models from: {temp_dir}")
        except Exception as e:
            print(f"Warning: Failed to clean up temp directory: {e}")


def main():
    print("="*60)
    print(f"Evaluating {args.method} pipeline on {args.dataset} with AutoGluon")
    print("="*60)
    
    # Load and split data
    print(f"\nLoading dataset: {args.dataset}")
    X, y = load_data(args.data_dir, args.dataset)
    X_train, y_train, X_val, y_val, X_test, y_test = build_data(X, y, random_state=args.split_seed)
    
    # Pre-normalize for diffprep_flex
    if args.method == "diffprep_flex":
        X_train, X_val, X_test = min_max_normalize(X_train, X_val, X_test)
    
    print(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Load best pipeline
    prep_pipeline, params, original_result = load_best_pipeline(
        args.result_dir, args.dataset, args.method, space
    )
    
    # Transform data with best pipeline
    X_train_trans, X_val_trans, X_test_trans = transform_data_with_pipeline(
        prep_pipeline, X_train, X_val, X_test, params,
        args.result_dir, args.dataset, args.method
    )
    
    # Create output directory
    output_dir = os.path.join(args.output_dir, args.method, args.dataset)
    os.makedirs(output_dir, exist_ok=True)
    
    # Train AutoGluon (models saved temporarily and deleted after)
    test_acc, leaderboard_dict = train_autogluon(
        X_train_trans, y_train, 
        X_val_trans, y_val,
        X_test_trans, y_test,
        args.time_limit
    )
    
    # Save comparison results
    comparison = {
        'dataset': args.dataset,
        'method': args.method,
        'split_seed': args.split_seed,
        'original_test_acc': original_result['best_test_acc'],
        'autogluon_test_acc': test_acc['accuracy'],
        'improvement': test_acc['accuracy'] - original_result['best_test_acc'],
        'time_limit': args.time_limit
    }
    
    comparison_path = os.path.join(output_dir, 'comparison.json')
    with open(comparison_path, 'w') as f:
        json.dump(comparison, f, indent=4)
    
    print("\n" + "="*60)
    print("RESULTS COMPARISON")
    print("="*60)
    print(f"Original (Logistic Regression): {original_result['best_test_acc']:.4f}")
    print(f"AutoGluon on Preprocessed Data: {test_acc['accuracy']:.4f}")
    print(f"Improvement: {comparison['improvement']:.4f}")
    print(f"\nResults saved to: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
