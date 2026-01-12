"""
Run DiffPrep training and AutoGluon evaluation on multiple datasets
"""

import subprocess
import sys
import os
import json
import time
from pathlib import Path

# List of dataset IDs to process
DATASET_IDS = [
    40975, 1233, 1115, 1466, 248, 279, 40740, 803, 942, 373, 
    1518, 737, 1396, 1399, 823, 253, 922, 7, 1066, 1164, 932,
    974, 1047, 991, 244, 1400, 862, 40520, 2, 40663, 1054,
    1387, 1397, 1401, 1393, 728, 876, 1358, 75, 18
]

# Configuration
METHOD = "diffprep_fix"
SPLIT_SEED = 42
AUTOGLUON_TIME_LIMIT = 300  # 5 minutes per dataset
MAX_FEATURES = 200  # Skip datasets with more than this many features



def run_command(cmd, description):
    """Run a command and handle errors"""
    print("\n" + "="*80)
    print(f"{description}")
    print("="*80)
    print(f"Command: {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ ERROR: {description} failed!")
        print(f"Return code: {e.returncode}")
        return False
    except Exception as e:
        print(f"\n❌ ERROR: {description} failed with exception: {e}")
        return False


def check_dataset_exists(dataset_id, data_dir="data"):
    """Check if dataset exists in data folder"""
    dataset_path = os.path.join(data_dir, str(dataset_id))
    csv_path = os.path.join(dataset_path, "data.csv")
    info_path = os.path.join(dataset_path, "info.json")
    
    if not os.path.exists(csv_path):
        print(f"⚠️  Warning: Dataset {dataset_id} not found at {csv_path}")
        return False
    if not os.path.exists(info_path):
        print(f"⚠️  Warning: Dataset {dataset_id} missing info.json at {info_path}")
        return False
    return True


def check_dataset_size(dataset_id, data_dir="data", max_features=1000):
    """Check if dataset has too many features"""
    info_path = os.path.join(data_dir, str(dataset_id), "info.json")
    
    try:
        with open(info_path, 'r') as f:
            info = json.load(f)
        
        num_features = info.get('num_features', 0)
        
        if num_features > max_features:
            print(f"⚠️  Dataset {dataset_id} has {num_features} features (max: {max_features})")
            return False, num_features
        
        return True, num_features
    except Exception as e:
        print(f"⚠️  Warning: Could not read info.json for dataset {dataset_id}: {e}")
        return True, None  # Continue if we can't read the file


def check_training_completed(dataset_id, method="diffprep_fix", result_dir="result"):
    """Check if DiffPrep training already completed for this dataset"""
    result_path = os.path.join(result_dir, method, str(dataset_id), "result.json")
    pipeline_path = os.path.join(result_dir, method, str(dataset_id), "prep_pipeline.pth")
    
    if os.path.exists(result_path) and os.path.exists(pipeline_path):
        return True
    return False


def check_autogluon_completed(dataset_id, method="diffprep_fix", output_dir="autogluon_results"):
    """Check if AutoGluon evaluation already completed for this dataset"""
    comparison_path = os.path.join(output_dir, method, str(dataset_id), "comparison.json")
    return os.path.exists(comparison_path)


def main():
    print("="*80)
    print("BATCH EXPERIMENT: DiffPrep Training + AutoGluon Evaluation")
    print("="*80)
    print(f"Total datasets: {len(DATASET_IDS)}")
    print(f"Method: {METHOD}")
    print(f"AutoGluon time limit: {AUTOGLUON_TIME_LIMIT}s per dataset")
    print()
    
    # Track results
    results = {
        'datasets_found': [],
        'datasets_missing': [],
        'datasets_too_large': [],
        'training_completed': [],
        'training_skipped': [],
        'training_failed': [],
        'autogluon_completed': [],
        'autogluon_skipped': [],
        'autogluon_failed': []
    }
    
    start_time = time.time()
    
    for i, dataset_id in enumerate(DATASET_IDS, 1):
        print("\n" + "🔹"*80)
        print(f"DATASET {i}/{len(DATASET_IDS)}: {dataset_id}")
        print("🔹"*80)
        
        # Check if dataset exists
        if not check_dataset_exists(dataset_id):
            results['datasets_missing'].append(dataset_id)
            print(f"❌ Skipping dataset {dataset_id} - not found")
            continue
        
        results['datasets_found'].append(dataset_id)
        
        # Check if dataset has too many features
        is_ok, num_features = check_dataset_size(dataset_id, max_features=MAX_FEATURES)
        if not is_ok:
            results['datasets_too_large'].append({'id': dataset_id, 'num_features': num_features})
            print(f"❌ Skipping dataset {dataset_id} - too many features ({num_features})")
            continue
        
        # Step 1: Train DiffPrep (if not already done)
        if check_training_completed(dataset_id, METHOD):
            print(f"✓ Training already completed for dataset {dataset_id}, skipping...")
            results['training_skipped'].append(dataset_id)
        else:
            print(f"\n📊 Step 1/2: Training DiffPrep on dataset {dataset_id}...")
            train_cmd = [
                sys.executable, "main.py",
                "--dataset", str(dataset_id),
                "--method", METHOD,
                "--split_seed", str(SPLIT_SEED)
            ]
            
            if run_command(train_cmd, f"Training DiffPrep on dataset {dataset_id}"):
                results['training_completed'].append(dataset_id)
                print(f"✅ Training completed for dataset {dataset_id}")
            else:
                results['training_failed'].append(dataset_id)
                print(f"❌ Training failed for dataset {dataset_id}, skipping AutoGluon evaluation")
                continue
        
        # Step 2: Evaluate with AutoGluon (if not already done)
        if check_autogluon_completed(dataset_id, METHOD):
            print(f"✓ AutoGluon evaluation already completed for dataset {dataset_id}, skipping...")
            results['autogluon_skipped'].append(dataset_id)
        else:
            print(f"\n🤖 Step 2/2: Evaluating with AutoGluon on dataset {dataset_id}...")
            eval_cmd = [
                sys.executable, "evaluate_with_autogluon.py",
                "--dataset", str(dataset_id),
                "--method", METHOD,
                "--split_seed", str(SPLIT_SEED),
                "--time_limit", str(AUTOGLUON_TIME_LIMIT)
            ]
            
            if run_command(eval_cmd, f"AutoGluon evaluation on dataset {dataset_id}"):
                results['autogluon_completed'].append(dataset_id)
                print(f"✅ AutoGluon evaluation completed for dataset {dataset_id}")
            else:
                results['autogluon_failed'].append(dataset_id)
                print(f"❌ AutoGluon evaluation failed for dataset {dataset_id}")
    
    # Print summary
    elapsed_time = time.time() - start_time
    print("\n\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    print(f"Total time: {elapsed_time/3600:.2f} hours ({elapsed_time/60:.1f} minutes)")
    print(f"\nDatasets:")
    print(f"  Found: {len(results['datasets_found'])}")
    print(f"  Missing: {len(results['datasets_missing'])}")
    print(f"  Too large (>{MAX_FEATURES} features): {len(results['datasets_too_large'])}")
    
    print(f"\nDiffPrep Training:")
    print(f"  Completed: {len(results['training_completed'])}")
    print(f"  Skipped (already done): {len(results['training_skipped'])}")
    print(f"  Failed: {len(results['training_failed'])}")
    
    print(f"\nAutoGluon Evaluation:")
    print(f"  Completed: {len(results['autogluon_completed'])}")
    print(f"  Skipped (already done): {len(results['autogluon_skipped'])}")
    print(f"  Failed: {len(results['autogluon_failed'])}")
    
    if results['datasets_missing']:
        print(f"\n⚠️  Missing datasets: {results['datasets_missing']}")
    
    if results['datasets_too_large']:
        print(f"\n⚠️  Datasets with too many features (skipped): {[d['id'] for d in results['datasets_too_large']]}")
    
    if results['training_failed']:
        print(f"\n❌ Training failed for: {results['training_failed']}")
    
    if results['autogluon_failed']:
        print(f"\n❌ AutoGluon failed for: {results['autogluon_failed']}")
    
    # Save summary to file
    summary_path = os.path.join("autogluon_results", METHOD, "experiment_summary.json")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    
    results['elapsed_time_seconds'] = elapsed_time
    results['total_datasets'] = len(DATASET_IDS)
    results['method'] = METHOD
    results['split_seed'] = SPLIT_SEED
    results['autogluon_time_limit'] = AUTOGLUON_TIME_LIMIT
    results['max_features'] = MAX_FEATURES
    
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\n📄 Summary saved to: {summary_path}")
    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETED!")
    print("="*80)


if __name__ == "__main__":
    main()
