"""
Extract and save the best learned pipeline for AutoGluon testing.

Usage:
    python extract_and_save_pipeline.py --dataset abalone --method diffprep_fix
"""

import argparse
import os
import json
import torch
import pickle
from prep_space import space
from experiment.experiment_utils import load_data, build_data, min_max_normalize
from pipeline.diffprep_fix_pipeline import DiffPrepFixPipeline
from pipeline.diffprep_flex_pipeline import DiffPrepFlexPipeline
from pipeline.baseline_pipeline import BaselinePipeline


def load_best_pipeline(result_dir, dataset, method, data_dir, split_seed=1):
    """Load the best trained pipeline."""
    print("="*60)
    print(f"Loading {method} pipeline for {dataset}")
    print("="*60)
    
    # Load data
    X, y = load_data(data_dir, dataset)
    X_train, y_train, X_val, y_val, X_test, y_test = build_data(X, y, random_state=split_seed)
    
    # Load parameters
    pipeline_dir = os.path.join(result_dir, method, dataset)
    params_path = os.path.join(pipeline_dir, "params.json")
    with open(params_path, 'r') as f:
        params = json.load(f)
    
    # Load results
    result_path = os.path.join(pipeline_dir, "result.json")
    with open(result_path, 'r') as f:
        result = json.load(f)
    
    print(f"\nBest pipeline from epoch {result['best_epoch']}")
    print(f"  Val Acc: {result['best_val_acc']:.4f}")
    print(f"  Test Acc: {result['best_test_acc']:.4f}")
    
    # Initialize and load pipeline
    if method == "diffprep_fix":
        prep_pipeline = DiffPrepFixPipeline(
            space,
            temperature=params["temperature"],
            use_sample=params["sample"],
            diff_method=params["diff_method"],
            init_method=params["init_method"]
        )
        
        # Initialize structure
        prep_pipeline.init_parameters(X_train, X_val, X_test)
        
        # Load weights
        pipeline_path = os.path.join(pipeline_dir, "prep_pipeline.pth")
        if os.path.exists(pipeline_path):
            prep_pipeline.load_state_dict(torch.load(pipeline_path))
            print(f"✓ Loaded pipeline weights")
        
    elif method == "diffprep_flex":
        X_train, X_val, X_test = min_max_normalize(X_train, X_val, X_test)
        
        prep_pipeline = DiffPrepFlexPipeline(
            space,
            temperature=params["temperature"],
            use_sample=params["sample"],
            diff_method=params["diff_method"],
            init_method=params["init_method"]
        )
        
        prep_pipeline.init_parameters(X_train, X_val, X_test)
        
        pipeline_path = os.path.join(pipeline_dir, "prep_pipeline.pth")
        if os.path.exists(pipeline_path):
            prep_pipeline.load_state_dict(torch.load(pipeline_path))
            print(f"✓ Loaded pipeline weights")
        
    elif method == "default" or method == "random":
        prep_pipeline = BaselinePipeline(space, init_method=params["init_method"])
        prep_pipeline.init_parameters(X_train, X_val, X_test)
        print("✓ Using baseline configuration")
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return prep_pipeline, X_train, y_train, X_val, y_val, X_test, y_test, result, params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--method', required=True, choices=['diffprep_fix', 'diffprep_flex', 'default', 'random'])
    parser.add_argument('--data_dir', default='data')
    parser.add_argument('--result_dir', default='result')
    parser.add_argument('--output_dir', default='saved_pipelines')
    parser.add_argument('--split_seed', type=int, default=1)
    args = parser.parse_args()
    
    # Load pipeline
    prep_pipeline, X_train, y_train, X_val, y_val, X_test, y_test, result, params = load_best_pipeline(
        args.result_dir,
        args.dataset,
        args.method,
        args.data_dir,
        args.split_seed
    )
    
    # Create output directory
    output_dir = os.path.join(args.output_dir, args.method, args.dataset)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save pipeline
    pipeline_path = os.path.join(output_dir, 'pipeline.pkl')
    with open(pipeline_path, 'wb') as f:
        pickle.dump(prep_pipeline, f)
    print(f"\n✓ Saved pipeline to {pipeline_path}")
    
    # Save data splits
    data_path = os.path.join(output_dir, 'data_splits.pkl')
    with open(data_path, 'wb') as f:
        pickle.dump({
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test
        }, f)
    print(f"✓ Saved data splits to {data_path}")
    
    # Save metadata
    metadata = {
        'dataset': args.dataset,
        'method': args.method,
        'split_seed': args.split_seed,
        'original_test_acc': result['best_test_acc'],
        'best_epoch': result['best_epoch'],
        'params': params
    }
    metadata_path = os.path.join(output_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved metadata to {metadata_path}")
    
    print("\n" + "="*60)
    print("EXTRACTION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
