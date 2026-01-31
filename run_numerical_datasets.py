"""
Script to run diffprep_fix and diffprep_flex on all numerical ID datasets
"""
import os
import subprocess
import time
from datetime import datetime
import json

# Get all numerical ID datasets
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


# Methods to run
METHODS = ['diffprep_fix', 'diffprep_flex']

# Configuration
SPLIT_SEED = 42
TRAIN_SEED = 1
MODEL = 'log'

def run_experiment(dataset, method, model='log', train_seed=1, split_seed=42):
    """Run a single experiment"""
    cmd = [
        'python', 'main.py',
        '--dataset', dataset,
        '--data_dir', 'data',
        '--result_dir', 'result',
        '--model', model,
        '--method', method,
        '--train_seed', str(train_seed),
        '--split_seed', str(split_seed)
    ]
    
    print(f"\n{'='*80}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Running: {dataset} with {method}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout per dataset
        )
        elapsed = time.time() - start_time
        
        print(f"✓ SUCCESS - Completed in {elapsed:.1f}s")
        return {
            'status': 'SUCCESS',
            'time': elapsed,
            'stdout': result.stdout[-500:] if len(result.stdout) > 500 else result.stdout
        }
    
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        print(f"✗ TIMEOUT - Exceeded time limit ({elapsed:.1f}s)")
        return {
            'status': 'TIMEOUT',
            'time': elapsed,
            'error': 'Execution exceeded timeout'
        }
    
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"✗ ERROR - Failed after {elapsed:.1f}s")
        print(f"Error: {e.stderr[-500:]}")
        return {
            'status': 'ERROR',
            'time': elapsed,
            'error': e.stderr[-500:] if e.stderr else str(e)
        }
    
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"✗ EXCEPTION - {str(e)}")
        return {
            'status': 'EXCEPTION',
            'time': elapsed,
            'error': str(e)
        }


def main():
    # Get all numerical datasets
    numerical_datasets = get_numerical_datasets()
    
    print(f"\n{'#'*80}")
    print(f"# Running Experiments on Numerical ID Datasets")
    print(f"# Total datasets: {len(numerical_datasets)}")
    print(f"# Methods: {', '.join(METHODS)}")
    print(f"# Split seed: {SPLIT_SEED}")
    print(f"# Train seed: {TRAIN_SEED}")
    print(f"{'#'*80}\n")
    
    print(f"Datasets to process: {', '.join(numerical_datasets)}\n")
    
    # Track results
    results = {}
    start_time = time.time()
    
    total_experiments = len(numerical_datasets) * len(METHODS)
    completed = 0
    
    for method in METHODS:
        results[method] = {}
        
        print(f"\n{'*'*80}")
        print(f"* STARTING METHOD: {method}")
        print(f"{'*'*80}\n")
        
        for idx, dataset in enumerate(numerical_datasets, 1):
            print(f"\n[{completed+1}/{total_experiments}] Dataset {idx}/{len(numerical_datasets)}: {dataset}")
            
            result = run_experiment(
                dataset=dataset,
                method=method,
                model=MODEL,
                train_seed=TRAIN_SEED,
                split_seed=SPLIT_SEED
            )
            
            results[method][dataset] = result
            completed += 1
            
            # Save intermediate results
            with open('numerical_datasets_results.json', 'w') as f:
                json.dump(results, f, indent=2)
    
    # Print summary
    total_time = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Total experiments: {total_experiments}")
    print(f"")
    
    for method in METHODS:
        print(f"\n{method}:")
        success = sum(1 for r in results[method].values() if r['status'] == 'SUCCESS')
        failed = len(results[method]) - success
        avg_time = sum(r['time'] for r in results[method].values()) / len(results[method]) if results[method] else 0
        
        print(f"  Success: {success}/{len(results[method])}")
        print(f"  Failed: {failed}")
        print(f"  Average time: {avg_time:.1f}s")
    
    print(f"\nResults saved to: numerical_datasets_results.json")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
