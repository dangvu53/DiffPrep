"""
Complete Kaggle pipeline: Run experiments + Extract pipelines + Evaluate with AutoGluon
Optimized for Kaggle environment with split_seed=42
"""
import os
import subprocess
import time
from datetime import datetime
import sys
import argparse

# Text-named datasets prioritized for Kaggle
TEXT_DATASETS = [
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

ALL_METHODS = ['diffprep_fix', 'diffprep_flex', 'default', 'random']


def run_experiment(dataset, method, split_seed=42, train_seed=1):
    """Run DiffPrep experiment"""
    cmd = [
        'python', 'main.py',
        '--dataset', dataset,
        '--data_dir', 'data',
        '--result_dir', 'result',
        '--model', 'log',
        '--method', method,
        '--train_seed', str(train_seed),
        '--split_seed', str(split_seed)
    ]
    
    print(f"\n{'='*80}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 1/3 Running experiment: {dataset} - {method}")
    print(f"{'='*80}\n")
    sys.stdout.flush()
    
    start_time = time.time()
    try:
        subprocess.run(cmd, check=True, timeout=1800)
        elapsed = time.time() - start_time
        print(f"\n✓ Experiment completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")
        sys.stdout.flush()
        return True, elapsed, None
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n✗ Experiment failed after {elapsed:.1f}s: {e}")
        sys.stdout.flush()
        return False, elapsed, str(e)


def extract_pipeline(dataset, method):
    """Extract and save the best pipeline"""
    cmd = [
        'python', 'extract_and_save_pipeline.py',
        '--dataset', dataset,
        '--method', method
    ]
    
    print(f"\n{'='*80}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 2/3 Extracting pipeline: {dataset} - {method}")
    print(f"{'='*80}\n")
    sys.stdout.flush()
    
    start_time = time.time()
    try:
        subprocess.run(cmd, check=True, timeout=300)
        elapsed = time.time() - start_time
        print(f"\n✓ Pipeline extracted in {elapsed:.1f}s")
        sys.stdout.flush()
        return True, elapsed, None
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n✗ Pipeline extraction failed after {elapsed:.1f}s: {e}")
        sys.stdout.flush()
        return False, elapsed, str(e)


def evaluate_with_autogluon(dataset, method, time_limit=300):
    """Evaluate with AutoGluon"""
    cmd = [
        'python', 'evaluate_with_autogluon_v2.py',
        '--dataset', dataset,
        '--method', method,
        '--time_limit', str(time_limit)
    ]
    
    print(f"\n{'='*80}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 3/3 Evaluating with AutoGluon: {dataset} - {method}")
    print(f"{'='*80}\n")
    sys.stdout.flush()
    
    start_time = time.time()
    try:
        subprocess.run(cmd, check=True, timeout=time_limit + 300)
        elapsed = time.time() - start_time
        print(f"\n✓ AutoGluon evaluation completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")
        sys.stdout.flush()
        return True, elapsed, None
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n✗ AutoGluon evaluation failed after {elapsed:.1f}s: {e}")
        sys.stdout.flush()
        return False, elapsed, str(e)


def process_dataset(dataset, method, split_seed, train_seed, autogluon_time):
    """Run full pipeline for one dataset"""
    results = {
        'dataset': dataset,
        'method': method,
        'experiment': {'success': False, 'time': 0},
        'extract': {'success': False, 'time': 0},
        'autogluon': {'success': False, 'time': 0}
    }
    
    # Step 1: Run experiment
    success, elapsed, error = run_experiment(dataset, method, split_seed, train_seed)
    results['experiment'] = {'success': success, 'time': elapsed, 'error': error}
    if not success:
        return results
    
    # Step 2: Extract pipeline
    success, elapsed, error = extract_pipeline(dataset, method)
    results['extract'] = {'success': success, 'time': elapsed, 'error': error}
    if not success:
        return results
    
    # Step 3: Evaluate with AutoGluon
    success, elapsed, error = evaluate_with_autogluon(dataset, method, autogluon_time)
    results['autogluon'] = {'success': success, 'time': elapsed, 'error': error}
    
    return results


def save_checkpoint(log_file, all_results):
    """Save checkpoint"""
    with open('checkpoint_full.txt', 'w') as f:
        completed = sum(1 for r in all_results if r['autogluon']['success'])
        failed = len(all_results) - completed
        f.write(f"Last update: {datetime.now()}\n")
        f.write(f"Completed: {completed}\n")
        f.write(f"Failed: {failed}\n")
        f.write(f"Log file: {log_file}\n")


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run full pipeline with DiffPrep and AutoGluon evaluation')
    parser.add_argument('--methods', nargs='+', default=['diffprep_fix'],
                        choices=ALL_METHODS,
                        help='Methods to run (default: diffprep_fix)')
    parser.add_argument('--datasets', nargs='+', default=None,
                        help='Specific datasets to run (default: all text datasets)')
    parser.add_argument('--split_seed', type=int, default=42,
                        help='Split seed for train/val/test split (default: 42)')
    parser.add_argument('--train_seed', type=int, default=1,
                        help='Training seed (default: 1)')
    parser.add_argument('--autogluon_time', type=int, default=300,
                        help='AutoGluon time limit in seconds (default: 300)')
    parser.add_argument('--skip_existing', action='store_true',
                        help='Skip datasets that already have AutoGluon results')
    args = parser.parse_args()
    
    # Detect Kaggle environment
    is_kaggle = os.path.exists('/kaggle/working')
    if is_kaggle:
        print("🔵 Running on Kaggle environment")
        os.chdir('/kaggle/working')
    
    # Check data directory
    if not os.path.exists('data'):
        print("❌ Error: 'data' directory not found!")
        sys.exit(1)
    
    # Determine which datasets to run
    if args.datasets:
        available_datasets = [d for d in args.datasets if os.path.exists(os.path.join('data', d))]
    else:
        available_datasets = [d for d in TEXT_DATASETS if os.path.exists(os.path.join('data', d))]
    
    if not available_datasets:
        print("❌ Error: No text-named datasets found")
        sys.exit(1)
    
    print(f"\n{'='*80}")
    print(f"Kaggle Full Pipeline: Experiments + AutoGluon Evaluation")
    print(f"{'='*80}")
    print(f"Available datasets: {len(available_datasets)}")
    print(f"Methods: {args.methods}")
    print(f"Split seed: {args.split_seed}")
    print(f"Train seed: {args.train_seed}")
    print(f"AutoGluon time limit: {args.autogluon_time}s per dataset")
    print(f"Skip existing: {args.skip_existing}")
    print(f"Total pipelines: {len(available_datasets) * len(args.methods)}")
    print(f"{'='*80}\n")
    
    # Create log file
    log_file = f"full_pipeline_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    all_results = []
    start_time = time.time()
    
    with open(log_file, 'w') as log:
        log.write(f"Full Pipeline Experiment Log\n")
        log.write(f"Started at: {datetime.now()}\n")
        log.write(f"Split seed: {args.split_seed}\n")
        log.write(f"Train seed: {args.train_seed}\n")
        log.write(f"Datasets: {available_datasets}\n")
        log.write(f"Methods: {args.methods}\n")
        log.write("="*80 + "\n\n")
        
        for idx, dataset in enumerate(available_datasets, 1):
            for method in args.methods:
                print(f"\n{'#'*80}")
                print(f"# Dataset {idx}/{len(available_datasets)}: {dataset}")
                print(f"# Method: {method}")
                print(f"# Progress: {len([r for r in all_results if r['autogluon']['success']])} completed")
                print(f"{'#'*80}\n")
                
                # Check if already completed
                result_path = os.path.join('autogluon_results', method, dataset, 'autogluon_results.json')
                if args.skip_existing and os.path.exists(result_path):
                    print(f"⏭ Skipping {dataset} - {method} (AutoGluon results already exist)")
                    log.write(f"[{datetime.now()}] {dataset} - {method}: SKIPPED (exists)\n")
                    log.flush()
                    continue
                
                # Process dataset
                results = process_dataset(dataset, method, args.split_seed, args.train_seed, args.autogluon_time)
                all_results.append(results)
                
                # Log results
                total_time = results['experiment']['time'] + results['extract']['time'] + results['autogluon']['time']
                if results['autogluon']['success']:
                    status = "SUCCESS"
                elif results['experiment']['success']:
                    status = "PARTIAL (experiment done, autogluon failed)"
                else:
                    status = "FAILED"
                
                log_entry = f"[{datetime.now()}] {dataset} - {method}: {status} ({total_time:.1f}s)\n"
                log_entry += f"  Experiment: {results['experiment']['time']:.1f}s\n"
                log_entry += f"  Extract: {results['extract']['time']:.1f}s\n"
                log_entry += f"  AutoGluon: {results['autogluon']['time']:.1f}s\n"
                log.write(log_entry)
                log.flush()
                
                # Save checkpoint
                save_checkpoint(log_file, all_results)
        
        total_elapsed = time.time() - start_time
        completed = sum(1 for r in all_results if r['autogluon']['success'])
        
        log.write("\n" + "="*80 + "\n")
        log.write(f"Completed at: {datetime.now()}\n")
        log.write(f"Total time: {total_elapsed/60:.1f} minutes ({total_elapsed/3600:.2f} hours)\n")
        log.write(f"Successfully completed: {completed}/{len(all_results)}\n")
    
    print(f"\n{'='*80}")
    print(f"✅ Full pipeline completed!")
    print(f"Successfully completed: {completed}/{len(all_results)}")
    print(f"Total time: {total_elapsed/60:.1f} minutes ({total_elapsed/3600:.2f} hours)")
    print(f"Log saved to: {log_file}")
    print(f"Results in: autogluon_results/")
    print(f"{'='*80}\n")
    
    if is_kaggle:
        print("\n📦 All results saved to /kaggle/working/")
        print("💾 AutoGluon results: autogluon_results/")
        print("📊 Experiment results: result/")
        print("🔧 Pipelines: saved_pipelines/")


if __name__ == "__main__":
    main()
