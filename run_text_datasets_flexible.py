"""
Flexible script to run experiments on text-named datasets
with various configuration options
"""
import os
import subprocess
import argparse
import time
from datetime import datetime

# All text-named datasets
ALL_TEXT_DATASETS = [
    'abalone',
    'ada_prior',
    'avila',
    'connect-4',
    'eeg',
    'google',
    'house_prices',
    'jungle_chess_2pcs_raw_endgame_complete',
    'microaggregation2',
    'mozilla4',
    'obesity',
    'page-blocks',
    'pbcseq',
    'pol',
    'Run_or_walk_information',
    'shuttle',
    'USCensus',
    'wall-robot-navigation'
]

ALL_METHODS = ['diffprep_fix', 'diffprep_flex', 'default', 'random']

def run_experiment(dataset, method, model='log', train_seed=1, split_seed=1, data_dir='data', result_dir='result'):
    """Run a single experiment"""
    cmd = [
        'python', 'main.py',
        '--dataset', dataset,
        '--data_dir', data_dir,
        '--result_dir', result_dir,
        '--model', model,
        '--method', method,
        '--train_seed', str(train_seed),
        '--split_seed', str(split_seed)
    ]
    
    print(f"\n{'='*80}")
    print(f"Running: {' '.join(cmd)}")
    print(f"Time: {datetime.now()}")
    print(f"{'='*80}\n")
    sys.stdout.flush()  # Force output to display immediately
    
    start_time = time.time()
    try:
        # Don't capture output - let it stream to console in real-time
        result = subprocess.run(cmd, check=True)
        elapsed = time.time() - start_time
        print(f"\n✓ Completed in {elapsed:.2f}s")
        sys.stdout.flush()
        return True, elapsed, None
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\n✗ Failed after {elapsed:.2f}s")
        print(f"Error: {e}")
        sys.stdout.flush()
        return False, elapsed, str(e)

def main():
    parser = argparse.ArgumentParser(description='Run experiments on text-named datasets')
    parser.add_argument('--datasets', nargs='+', default=None,
                        help='Specific datasets to run (default: all text datasets)')
    parser.add_argument('--methods', nargs='+', default=ALL_METHODS,
                        choices=ALL_METHODS,
                        help='Methods to run (default: all methods)')
    parser.add_argument('--model', default='log', choices=['log', 'two'],
                        help='Model to use (default: log)')
    parser.add_argument('--train_seed', type=int, default=1,
                        help='Training seed (default: 1)')
    parser.add_argument('--split_seed', type=int, default=1,
                        help='Data split seed (default: 1)')
    parser.add_argument('--data_dir', default='data',
                        help='Data directory (default: data)')
    parser.add_argument('--result_dir', default='result',
                        help='Result directory (default: result)')
    parser.add_argument('--skip_existing', action='store_true',
                        help='Skip experiments if result directory already exists')
    parser.add_argument('--list', action='store_true',
                        help='List all available text-named datasets and exit')
    
    args = parser.parse_args()
    
    # List datasets if requested
    if args.list:
        print("Available text-named datasets:")
        for i, ds in enumerate(ALL_TEXT_DATASETS, 1):
            exists = "✓" if os.path.exists(os.path.join(args.data_dir, ds)) else "✗"
            print(f"  {i:2d}. {exists} {ds}")
        return
    
    # Determine which datasets to run
    if args.datasets:
        datasets = args.datasets
    else:
        datasets = ALL_TEXT_DATASETS
    
    # Create log file
    log_file = f"experiment_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    total_experiments = len(datasets) * len(args.methods)
    completed = 0
    failed = 0
    skipped = 0
    
    print(f"\n{'='*80}")
    print(f"Starting experiments")
    print(f"Datasets: {len(datasets)}")
    print(f"Methods: {args.methods}")
    print(f"Total experiments: {total_experiments}")
    print(f"Log file: {log_file}")
    print(f"{'='*80}\n")
    
    with open(log_file, 'w') as log:
        log.write(f"Experiment started at: {datetime.now()}\n")
        log.write(f"Datasets: {datasets}\n")
        log.write(f"Methods: {args.methods}\n")
        log.write(f"Model: {args.model}\n")
        log.write(f"Seeds - Train: {args.train_seed}, Split: {args.split_seed}\n")
        log.write("="*80 + "\n\n")
        
        for dataset in datasets:
            # Check if dataset exists
            dataset_path = os.path.join(args.data_dir, dataset)
            if not os.path.exists(dataset_path):
                print(f"⚠ Warning: Dataset '{dataset}' not found at {dataset_path}, skipping...")
                log.write(f"SKIPPED: {dataset} - not found\n")
                skipped += 1
                continue
            
            print(f"\n{'#'*80}")
            print(f"# Dataset: {dataset}")
            print(f"# Progress: {completed} succeeded, {failed} failed, {skipped} skipped")
            print(f"{'#'*80}\n")
            
            for method in args.methods:
                # Check if we should skip existing results
                if args.skip_existing:
                    result_path = os.path.join(args.result_dir, method, dataset)
                    if os.path.exists(result_path):
                        print(f"⏭ Skipping {dataset} - {method} (result already exists)")
                        log.write(f"[{datetime.now()}] {dataset} - {method}: SKIPPED (exists)\n")
                        skipped += 1
                        continue
                
                success, elapsed, error = run_experiment(
                    dataset, method, args.model,
                    args.train_seed, args.split_seed,
                    args.data_dir, args.result_dir
                )
                
                if success:
                    completed += 1
                    status = "SUCCESS"
                else:
                    failed += 1
                    status = "FAILED"
                
                log_entry = f"[{datetime.now()}] {dataset} - {method}: {status} ({elapsed:.2f}s)\n"
                if error:
                    log_entry += f"  Error: {error}\n"
                log.write(log_entry)
                log.flush()
        
        log.write("\n" + "="*80 + "\n")
        log.write(f"Experiment completed at: {datetime.now()}\n")
        log.write(f"Results: {completed} succeeded, {failed} failed, {skipped} skipped\n")
    
    print(f"\n{'='*80}")
    print(f"All experiments completed!")
    print(f"Success: {completed}")
    print(f"Failed: {failed}")
    print(f"Skipped: {skipped}")
    print(f"Total: {total_experiments}")
    print(f"Log saved to: {log_file}")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
