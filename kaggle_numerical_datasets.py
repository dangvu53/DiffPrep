"""
Kaggle script: Run DiffPrep (fix and flex) on numerical ID datasets
Copy and paste this entire script into a Kaggle notebook cell and run
"""

import subprocess
import json
from datetime import datetime
import time
import os

def get_numerical_datasets(data_dir='data'):
    """Get list of all numerical ID dataset folders"""
    all_items = os.listdir(data_dir)
    numerical_datasets = []
    
    for item in all_items:
        # Check if item is a directory and has a numeric name
        item_path = os.path.join(data_dir, item)
        if os.path.isdir(item_path) and item.replace('-', '').isdigit():
            # Check if it has data.csv
            if os.path.exists(os.path.join(item_path, 'data.csv')):
                numerical_datasets.append(item)
    
    return sorted(numerical_datasets, key=lambda x: int(x.replace('-', '')))

# Get all numerical datasets
datasets = get_numerical_datasets()

# Configuration
METHODS = ['diffprep_fix', 'diffprep_flex']  # Both methods
SPLIT_SEED = 42
TRAIN_SEED = 1

results = {}
start_time = time.time()

print(f"{'='*80}")
print(f"Starting Numerical Dataset Experiments")
print(f"Methods: {', '.join(METHODS)}")
print(f"Split seed: {SPLIT_SEED}")
print(f"Train seed: {TRAIN_SEED}")
print(f"Total datasets: {len(datasets)}")
print(f"{'='*80}\n")

print(f"Datasets: {', '.join(datasets)}\n")

total_experiments = len(datasets) * len(METHODS)
completed = 0

for method in METHODS:
    results[method] = {}
    
    print(f"\n{'*'*80}")
    print(f"* METHOD: {method}")
    print(f"{'*'*80}\n")
    
    for idx, dataset in enumerate(datasets, 1):
        # Convert dataset to string to handle numeric IDs
        dataset_str = str(dataset)
        
        print(f"\n{'#'*80}")
        print(f"# [{completed+1}/{total_experiments}] Dataset {idx}/{len(datasets)}: {dataset_str}")
        print(f"# Completed: {completed}/{total_experiments}")
        print(f"{'#'*80}\n")
        
        dataset_start = time.time()
        
        try:
            # Run DiffPrep experiment
            print(f"Running {method} experiment...")
            subprocess.run([
                'python', 'main.py',
                '--dataset', dataset_str,
                '--data_dir', 'data',
                '--result_dir', 'result',
                '--model', 'log',
                '--method', method,
                '--train_seed', str(TRAIN_SEED),
                '--split_seed', str(SPLIT_SEED)
            ], check=True, timeout=3600)
            
            elapsed = time.time() - dataset_start
            results[method][dataset_str] = {
                'status': 'SUCCESS',
                'time': elapsed
            }
            
            print(f"✓ SUCCESS - {dataset_str} with {method} ({elapsed:.1f}s)")
            
        except subprocess.TimeoutExpired:
            elapsed = time.time() - dataset_start
            results[method][dataset_str] = {
                'status': 'TIMEOUT',
                'time': elapsed
            }
            print(f"✗ TIMEOUT - {dataset_str} with {method}")
            
        except Exception as e:
            elapsed = time.time() - dataset_start
            results[method][dataset_str] = {
                'status': 'ERROR',
                'time': elapsed,
                'error': str(e)
            }
            print(f"✗ ERROR - {dataset_str} with {method}: {str(e)}")
        
        completed += 1
        
        # Save progress
        with open('numerical_results.json', 'w') as f:
            json.dump(results, f, indent=2)

# Print summary
total_time = time.time() - start_time
print(f"\n{'='*80}")
print(f"EXPERIMENT COMPLETE")
print(f"{'='*80}")
print(f"Total time: {total_time/60:.1f} minutes")
print(f"Total experiments: {total_experiments}")

for method in METHODS:
    success = sum(1 for r in results[method].values() if r['status'] == 'SUCCESS')
    print(f"\n{method}:")
    print(f"  Success: {success}/{len(datasets)}")
    print(f"  Failed: {len(datasets) - success}")

print(f"\nResults saved to: numerical_results.json")
print(f"{'='*80}")
