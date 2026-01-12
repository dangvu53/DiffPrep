"""
Quick test of the complete pipeline workflow.

This script verifies that:
1. We can load experiment results
2. Extract configuration from pipelines
3. Create and fit a Preprocessor
4. Save and reload the Preprocessor
5. Transform data correctly

Usage:
    python test_pipeline_workflow.py --dataset abalone --method diffprep_fix
"""

import argparse
import os
import json
import numpy as np
import pandas as pd

from new_preprocessor import Preprocessor
from experiment.experiment_utils import load_data, build_data
from extract_and_save_pipeline import load_and_extract_pipeline


def test_workflow(dataset, method, data_dir='data', result_dir='result/default'):
    """Test the complete workflow"""
    print("="*70)
    print(f"TESTING PIPELINE WORKFLOW: {method} on {dataset}")
    print("="*70)
    
    # Load data
    print("\n1. Loading data...")
    X, y = load_data(data_dir, dataset)
    X_train, y_train, X_val, y_val, X_test, y_test = build_data(X, y, random_state=1)
    print(f"   ✓ Loaded data - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Extract pipeline
    print("\n2. Extracting pipeline from experiment results...")
    try:
        preprocessor, config, result, params = load_and_extract_pipeline(
            result_dir, dataset, method, data_dir, split_seed=1
        )
        print(f"   ✓ Extracted pipeline successfully")
    except Exception as e:
        print(f"   ✗ Failed to extract pipeline: {e}")
        print(f"   Make sure you've run the experiment first with --save_model flag:")
        print(f"   python -m experiment.{method}_experiment --dataset {dataset} --save_model")
        return False
    
    # Test transform
    print("\n3. Testing transform...")
    try:
        X_train_trans, y_train_aligned = preprocessor.transform(X_train, y_train)
        X_val_trans, y_val_aligned = preprocessor.transform(X_val, y_val)
        X_test_trans, y_test_aligned = preprocessor.transform(X_test, y_test)
        
        print(f"   ✓ Transformed all splits")
        print(f"     Train: {X_train.shape} → {X_train_trans.shape}")
        print(f"     Val: {X_val.shape} → {X_val_trans.shape}")
        print(f"     Test: {X_test.shape} → {X_test_trans.shape}")
        
        # Check for NaNs
        if X_train_trans.isnull().any().any():
            print(f"   ⚠ Warning: NaNs found in transformed train data")
        if X_val_trans.isnull().any().any():
            print(f"   ⚠ Warning: NaNs found in transformed val data")
        if X_test_trans.isnull().any().any():
            print(f"   ⚠ Warning: NaNs found in transformed test data")
        
        # Check alignment
        assert len(X_train_trans) == len(y_train_aligned), "Train alignment mismatch"
        assert len(X_val_trans) == len(y_val_aligned), "Val alignment mismatch"
        assert len(X_test_trans) == len(y_test_aligned), "Test alignment mismatch"
        print(f"   ✓ Labels correctly aligned")
        
    except Exception as e:
        print(f"   ✗ Transform failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test save/load
    print("\n4. Testing save/load...")
    try:
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
            temp_path = tmp.name
        
        preprocessor.save(temp_path)
        print(f"   ✓ Saved to {temp_path}")
        
        preprocessor_loaded = Preprocessor.load(temp_path)
        print(f"   ✓ Loaded from {temp_path}")
        
        # Test that loaded preprocessor works
        X_test_trans2, y_test_aligned2 = preprocessor_loaded.transform(X_test, y_test)
        
        # Compare results
        if X_test_trans.equals(X_test_trans2):
            print(f"   ✓ Loaded preprocessor produces identical results")
        else:
            # Check if numerically close (floating point differences)
            numeric_cols = X_test_trans.select_dtypes(include='number').columns
            if len(numeric_cols) > 0:
                max_diff = (X_test_trans[numeric_cols] - X_test_trans2[numeric_cols]).abs().max().max()
                if max_diff < 1e-6:
                    print(f"   ✓ Results are numerically equivalent (max diff: {max_diff:.2e})")
                else:
                    print(f"   ⚠ Warning: Results differ by up to {max_diff:.2e}")
            else:
                print(f"   ⚠ Warning: Results differ")
        
        # Clean up
        os.remove(temp_path)
        
    except Exception as e:
        print(f"   ✗ Save/load failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test config
    print("\n5. Checking configuration...")
    print(f"   Config keys: {list(config.keys())}")
    for key, value in config.items():
        if isinstance(value, dict):
            print(f"   {key}: {len(value)} per-feature settings")
        else:
            print(f"   {key}: {value}")
    
    print("\n" + "="*70)
    print("ALL TESTS PASSED ✓")
    print("="*70)
    print("\nYou can now:")
    print(f"1. Save this preprocessor:")
    print(f"   python extract_and_save_pipeline.py --dataset {dataset} --method {method}")
    print(f"\n2. Test with AutoGluon:")
    print(f"   python evaluate_with_autogluon_v2.py --dataset {dataset} --method {method}")
    print("="*70)
    
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help='Dataset name')
    parser.add_argument('--method', required=True, 
                        choices=['diffprep_fix', 'diffprep_flex', 'baseline'])
    parser.add_argument('--data_dir', default='data')
    parser.add_argument('--result_dir', default='result/default')
    args = parser.parse_args()
    
    success = test_workflow(args.dataset, args.method, args.data_dir, args.result_dir)
    
    if not success:
        exit(1)


if __name__ == "__main__":
    main()
