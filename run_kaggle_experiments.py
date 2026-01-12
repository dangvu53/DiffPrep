"""
Kaggle-optimized script for running text dataset experiments
Includes progress tracking and automatic result saving
"""
import os
import subprocess
import time
from datetime import datetime
import sys

# Text-named datasets prioritized for Kaggle (smaller/faster first)
PRIORITY_DATASETS = [
    'abalone',
    'obesity', 
    'page-blocks',
    'pbcseq',
    'eeg',
    'google',
    'house_prices',
    'avila',
    'ada_prior',
    'microaggregation2',
    'pol',
    'Run_or_walk_information',
    'shuttle',
    'USCensus',
    'wall-robot-navigation',
    'connect-4',
    'mozilla4',
    'jungle_chess_2pcs_raw_endgame_complete'
]

# Methods to run (you can reduce this to save time)
METHODS = ['diffprep_fix', 'default']  # Reduced for faster execution

def run_experiment(dataset, method, model='log', train_seed=1, split_seed=1):
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
    print(f"{'='*80}\n")
    sys.stdout.flush()
    
    start_time = time.time()
    try:
        # Don't capture output - let it stream in real-time
        result = subprocess.run(cmd, check=True, timeout=1800)  # 30 min timeout
        elapsed = time.time() - start_time
        print(f"\n✓ Completed in {elapsed:.2f}s ({elapsed/60:.1f} min)")
        sys.stdout.flush()
        return True, elapsed, None
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        print(f"\n⏱ Timeout after {elapsed:.2f}s")
        sys.stdout.flush()
        return False, elapsed, "Timeout (30 min)"
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\n✗ Failed after {elapsed:.2f}s")
        print(f"Error: {e}")
        sys.stdout.flush()
        return False, elapsed, str(e)
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n✗ Exception after {elapsed:.2f}s: {str(e)}")
        sys.stdout.flush()
        return False, elapsed, str(e)

def save_checkpoint(log_file, completed, failed, current_dataset, current_method):
    """Save checkpoint for resuming"""
    with open('checkpoint.txt', 'w') as f:
        f.write(f"Last update: {datetime.now()}\n")
        f.write(f"Completed: {completed}\n")
        f.write(f"Failed: {failed}\n")
        f.write(f"Current: {current_dataset} - {current_method}\n")
        f.write(f"Log file: {log_file}\n")

def main():
    # Detect if running on Kaggle
    is_kaggle = os.path.exists('/kaggle/working')
    if is_kaggle:
        print("🔵 Running on Kaggle environment")
        # Change to working directory for outputs
        os.chdir('/kaggle/working')
    
    # Check if data exists
    if not os.path.exists('data'):
        print("❌ Error: 'data' directory not found!")
        print("Current directory:", os.getcwd())
        print("Files:", os.listdir('.'))
        sys.exit(1)
    
    # Filter datasets that actually exist
    available_datasets = [d for d in PRIORITY_DATASETS if os.path.exists(os.path.join('data', d))]
    
    if not available_datasets:
        print("❌ Error: No text-named datasets found in data/ directory")
        sys.exit(1)
    
    print(f"\n{'='*80}")
    print(f"Kaggle Experiment Runner")
    print(f"{'='*80}")
    print(f"Available datasets: {len(available_datasets)}")
    print(f"Methods: {METHODS}")
    print(f"Total experiments: {len(available_datasets) * len(METHODS)}")
    print(f"Estimated time: {len(available_datasets) * len(METHODS) * 10} minutes (rough estimate)")
    print(f"{'='*80}\n")
    
    # Create log file
    log_file = f"experiment_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    total_experiments = len(available_datasets) * len(METHODS)
    completed = 0
    failed = 0
    start_experiment_time = time.time()
    
    with open(log_file, 'w') as log:
        log.write(f"Kaggle Experiment Log\n")
        log.write(f"Started at: {datetime.now()}\n")
        log.write(f"Datasets: {available_datasets}\n")
        log.write(f"Methods: {METHODS}\n")
        log.write(f"Total experiments: {total_experiments}\n")
        log.write("="*80 + "\n\n")
        
        for dataset in available_datasets:
            for method in METHODS:
                # Check if result already exists
                result_path = os.path.join('result', method, dataset)
                if os.path.exists(result_path):
                    print(f"⏭ Skipping {dataset} - {method} (already exists)")
                    log.write(f"[{datetime.now()}] {dataset} - {method}: SKIPPED (exists)\n")
                    log.flush()
                    completed += 1
                    continue
                
                # Update progress
                progress = (completed + failed) / total_experiments * 100
                elapsed_total = time.time() - start_experiment_time
                avg_time = elapsed_total / max(1, completed + failed)
                remaining = (total_experiments - completed - failed) * avg_time
                
                print(f"\n{'#'*80}")
                print(f"# Progress: {completed}/{total_experiments} completed, {failed} failed ({progress:.1f}%)")
                print(f"# Elapsed: {elapsed_total/60:.1f} min | Est. remaining: {remaining/60:.1f} min")
                print(f"# Dataset: {dataset} | Method: {method}")
                print(f"{'#'*80}\n")
                
                # Save checkpoint
                save_checkpoint(log_file, completed, failed, dataset, method)
                
                # Run experiment
                success, elapsed, error = run_experiment(dataset, method)
                
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
        
        total_time = time.time() - start_experiment_time
        log.write("\n" + "="*80 + "\n")
        log.write(f"Completed at: {datetime.now()}\n")
        log.write(f"Total time: {total_time/60:.1f} minutes\n")
        log.write(f"Results: {completed} succeeded, {failed} failed\n")
    
    print(f"\n{'='*80}")
    print(f"✅ All experiments completed!")
    print(f"Success: {completed}/{total_experiments}")
    print(f"Failed: {failed}/{total_experiments}")
    print(f"Total time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    print(f"Log saved to: {log_file}")
    print(f"{'='*80}\n")
    
    if is_kaggle:
        print("\n📦 Results are in /kaggle/working/ and will be saved automatically")
        print("💾 Download the output data after notebook completes")

if __name__ == "__main__":
    main()
