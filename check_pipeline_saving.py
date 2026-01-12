"""
Verification script to check if pipelines are being saved correctly.

This demonstrates exactly what gets saved when you run main.py with save_model=True.
"""

import os
import json

def check_experiment_output(result_dir, method, dataset):
    """Check what files were saved by an experiment"""
    
    exp_dir = os.path.join(result_dir, method, dataset)
    
    print("="*70)
    print(f"Checking experiment output: {method} on {dataset}")
    print(f"Directory: {exp_dir}")
    print("="*70)
    
    if not os.path.exists(exp_dir):
        print(f"❌ Directory does not exist: {exp_dir}")
        print(f"\nYou need to run the experiment first:")
        print(f"  python main.py --dataset {dataset} --method {method}")
        return False
    
    # Check required files
    required_files = {
        "params.json": "Hyperparameters",
        "result.json": "Training results (val_acc, test_acc, best_epoch)"
    }
    
    optional_files = {
        "prep_pipeline.pth": "Saved preprocessing pipeline (if save_model=True)",
        "end_model.pth": "Saved end model (if save_model=True)",
        "pipeline_config.json": "Human-readable pipeline configuration"
    }
    
    print("\n📁 Required Files:")
    for filename, description in required_files.items():
        filepath = os.path.join(exp_dir, filename)
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            print(f"  ✅ {filename:25s} ({size:,} bytes) - {description}")
        else:
            print(f"  ❌ {filename:25s} MISSING - {description}")
    
    print("\n📁 Optional Files (require save_model=True):")
    has_models = True
    for filename, description in optional_files.items():
        filepath = os.path.join(exp_dir, filename)
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            print(f"  ✅ {filename:25s} ({size:,} bytes) - {description}")
        else:
            print(f"  ❌ {filename:25s} MISSING - {description}")
            if filename.endswith('.pth'):
                has_models = False
    
    # Check params.json for save_model setting
    params_path = os.path.join(exp_dir, "params.json")
    if os.path.exists(params_path):
        with open(params_path, 'r') as f:
            params = json.load(f)
        
        print(f"\n⚙️  Configuration:")
        print(f"  save_model setting: {params.get('save_model', 'NOT SET (defaults to False)')}")
        
        if not has_models and params.get('save_model', False):
            print(f"\n⚠️  WARNING: save_model=True in params but .pth files are missing!")
            print(f"  This experiment was run with OLD CODE that ignored save_model.")
            print(f"  Re-run the experiment to save models:")
            print(f"    python main.py --dataset {dataset} --method {method}")
        elif not has_models:
            print(f"\n⚠️  Models were not saved (save_model=False or not set)")
            print(f"  To save models for testing with AutoGluon:")
            print(f"    1. Set save_model=True in main.py (line 32)")
            print(f"    2. Re-run: python main.py --dataset {dataset} --method {method}")
    
    # Check result.json
    result_path = os.path.join(exp_dir, "result.json")
    if os.path.exists(result_path):
        with open(result_path, 'r') as f:
            result = json.load(f)
        
        print(f"\n📊 Training Results:")
        print(f"  Best Epoch:    {result.get('best_epoch', 'N/A')}")
        print(f"  Val Accuracy:  {result.get('best_val_acc', 'N/A'):.4f}")
        print(f"  Test Accuracy: {result.get('best_test_acc', 'N/A'):.4f}")
    
    print("\n" + "="*70)
    
    if has_models:
        print("✅ SUCCESS: Pipeline models are saved!")
        print("\nNext steps:")
        print(f"  1. Extract pipeline: python extract_and_save_pipeline.py --dataset {dataset} --method {method}")
        print(f"  2. Test with AutoGluon: python evaluate_with_autogluon_v2.py --dataset {dataset} --method {method}")
    else:
        print("⚠️  Pipeline models NOT saved - need to re-run experiment")
        print("\nNext steps:")
        print(f"  1. Ensure main.py has save_model=True (line 32) ✅ Already set!")
        print(f"  2. Re-run experiment: python main.py --dataset {dataset} --method {method}")
    
    print("="*70)
    
    return has_models


if __name__ == "__main__":
    import sys
    
    # Check default experiment
    print("\n" * 2)
    
    # Check common experiment directories
    result_dir = "result"
    
    experiments_to_check = []
    
    # Find all existing experiments
    for root, dirs, files in os.walk(result_dir):
        if "params.json" in files:
            # Extract method and dataset from path
            path_parts = root.replace(result_dir, "").strip(os.sep).split(os.sep)
            if len(path_parts) >= 2:
                method = path_parts[0] if path_parts[0] != "result" else path_parts[1]
                dataset = path_parts[-1]
                experiments_to_check.append((method, dataset))
    
    if not experiments_to_check:
        print("="*70)
        print("No experiments found in result/ directory")
        print("="*70)
        print("\nRun an experiment first:")
        print("  python main.py --dataset abalone --method diffprep_fix")
        sys.exit(1)
    
    # Check each experiment
    all_good = True
    for method, dataset in experiments_to_check:
        has_models = check_experiment_output(result_dir, method, dataset)
        if not has_models:
            all_good = False
        print("\n")
    
    if all_good:
        print("🎉 All experiments have saved pipeline models!")
    else:
        print("⚠️  Some experiments need to be re-run to save pipeline models")
