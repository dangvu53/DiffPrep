"""
Simple Kaggle cell script: Run DiffPrep + AutoGluon for all text datasets
Copy and paste this entire script into a Kaggle notebook cell and run
"""

import subprocess
import json
from datetime import datetime
import time

# All text-named datasets
datasets = [
    'abalone', 'ada_prior', 'avila', 'connect-4', 'eeg', 'google',
    'house_prices', 'jungle_chess_2pcs_raw_endgame_complete',
    'microaggregation2', 'mozilla4', 'obesity', 'page-blocks',
    'pbcseq', 'pol', 'Run_or_walk_information', 'shuttle',
    'USCensus', 'wall-robot-navigation'
]

# Configuration
METHOD = 'diffprep_fix'  # Change to: 'diffprep_flex', 'default', 'random'
SPLIT_SEED = 42
TRAIN_SEED = 1
AUTOGLUON_TIME_LIMIT = 300  # seconds

results = {}
start_time = time.time()

print(f"{'='*80}")
print(f"Starting Full Pipeline Experiments")
print(f"Method: {METHOD}")
print(f"Split seed: {SPLIT_SEED}")
print(f"Train seed: {TRAIN_SEED}")
print(f"Total datasets: {len(datasets)}")
print(f"{'='*80}\n")

for idx, dataset in enumerate(datasets, 1):
    print(f"\n{'#'*80}")
    print(f"# [{idx}/{len(datasets)}] Processing: {dataset}")
    print(f"# Completed: {len([r for r in results.values() if 'SUCCESS' in r])}")
    print(f"{'#'*80}\n")
    
    dataset_start = time.time()
    
    try:
        # Step 1: Run DiffPrep experiment
        print(f"[1/3] Running {METHOD} experiment...")
        subprocess.run([
            'python', 'main.py',
            '--dataset', dataset,
            '--method', METHOD,
            '--model', 'log',
            '--split_seed', str(SPLIT_SEED),
            '--train_seed', str(TRAIN_SEED),
        ], check=True)
        
        # Step 2: Extract pipeline
        print(f"\n[2/3] Extracting pipeline...")
        subprocess.run([
            'python', 'extract_and_save_pipeline.py',
            '--dataset', dataset,
            '--method', METHOD,
        ], check=True)
        
        # Step 3: Run AutoGluon evaluation
        print(f"\n[3/3] Running AutoGluon evaluation...")
        subprocess.run([
            'python', 'evaluate_with_autogluon_v2.py',
            '--dataset', dataset,
            '--method', METHOD,
            '--time_limit', str(AUTOGLUON_TIME_LIMIT)
        ], check=True)
        
        elapsed = time.time() - dataset_start
        results[dataset] = f"SUCCESS ({elapsed:.1f}s)"
        print(f"\n✅ {dataset} completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")
        
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - dataset_start
        error_msg = f"FAILED at command: {e.cmd}"
        results[dataset] = error_msg
        print(f"\n❌ {dataset} failed after {elapsed:.1f}s")
        print(f"Error: {error_msg}")
        
    except Exception as e:
        elapsed = time.time() - dataset_start
        error_msg = f"FAILED: {str(e)}"
        results[dataset] = error_msg
        print(f"\n❌ {dataset} failed after {elapsed:.1f}s")
        print(f"Error: {error_msg}")

# Save results summary
total_time = time.time() - start_time
summary = {
    'timestamp': datetime.now().isoformat(),
    'method': METHOD,
    'split_seed': SPLIT_SEED,
    'train_seed': TRAIN_SEED,
    'total_time_minutes': total_time / 60,
    'results': results
}

with open('kaggle_results_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

# Print final summary
print(f"\n{'='*80}")
print(f"FINAL SUMMARY")
print(f"{'='*80}")
print(f"Total time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
print(f"Method: {METHOD}")
print(f"Split seed: {SPLIT_SEED}\n")

successful = [d for d, s in results.items() if 'SUCCESS' in s]
failed = [d for d, s in results.items() if 'FAILED' in s]

print(f"✅ Successful: {len(successful)}/{len(datasets)}")
print(f"❌ Failed: {len(failed)}/{len(datasets)}\n")

if successful:
    print("Successful datasets:")
    for dataset in successful:
        print(f"  ✓ {dataset}")

if failed:
    print(f"\nFailed datasets:")
    for dataset in failed:
        print(f"  ✗ {dataset}: {results[dataset]}")

print(f"\n{'='*80}")
print(f"Results saved to: kaggle_results_summary.json")
print(f"{'='*80}")
