"""
Extract and display the actual preprocessing configuration from a saved pipeline.

Usage:
    python extract_pipeline_config.py --dataset abalone --method diffprep_fix
"""

import argparse
import os
import pickle
import torch
import numpy as np
from torch.distributions.utils import logits_to_probs
import sys


def get_transformation_name(tf):
    """Get a readable name for the transformation."""
    return tf.method if hasattr(tf, 'method') else str(type(tf).__name__)


def extract_pipeline_config(pipeline_path):
    """Extract the preprocessing configuration from a saved pipeline."""
    
    # Set up numpy._core compatibility module if needed
    if not hasattr(np, '_core'):
        import types
        np._core = types.ModuleType('numpy._core')
        np._core.multiarray = np.core.multiarray
        sys.modules['numpy._core'] = np._core
        sys.modules['numpy._core.multiarray'] = np.core.multiarray
    
    # Create stub classes for NumPy 2.x random generators
    if not hasattr(sys.modules.get('numpy.random', {}), '_mt19937'):
        import types
        mt19937_module = types.ModuleType('numpy.random._mt19937')
        
        # Create a stub MT19937 class that can be unpickled
        class MT19937Stub:
            def __init__(self, *args, **kwargs):
                pass
            def __setstate__(self, state):
                pass
        
        mt19937_module.MT19937 = MT19937Stub
        sys.modules['numpy.random._mt19937'] = mt19937_module
    
    # Custom unpickler to handle numpy version compatibility
    class NumpyUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            # Handle NumPy 2.x to 1.x compatibility
            if module == 'numpy._core.multiarray':
                module = 'numpy.core.multiarray'
            elif module.startswith('numpy._core'):
                module = module.replace('numpy._core', 'numpy.core')
            return super().find_class(module, name)
    
    # Load the pipeline with compatibility settings
    try:
        with open(pipeline_path, 'rb') as f:
            pipeline = NumpyUnpickler(f).load()
    except Exception as e1:
        print(f"[WARNING] First attempt failed: {e1}")
        print("[INFO] Trying alternative loading method...")
        try:
            with open(pipeline_path, 'rb') as f:
                pipeline = pickle.load(f, encoding='latin1')
        except Exception as e2:
            print(f"[ERROR] Alternative method also failed: {e2}")
            raise
    
    print("\n" + "="*80)
    print("LEARNED PREPROCESSING PIPELINE")
    print("="*80)
    
    config = {
        'transformers': []
    }
    
    # Iterate through each transformer in the pipeline
    for idx, transformer in enumerate(pipeline.pipeline):
        transformer_info = {
            'name': transformer.name,
            'features': []
        }
        
        print(f"\n{'─'*80}")
        print(f"STEP {idx + 1}: {transformer.name.upper()}")
        print(f"{'─'*80}")
        
        # Check if this is the FirstTransformer (missing value imputation)
        if hasattr(transformer, 'num_tf_options') and hasattr(transformer, 'cat_tf_options'):
            # This is FirstTransformer - handles numerical and categorical separately
            
            # Get the selected transformations using argmax
            if transformer.num_tf_prob_logits is not None:
                print("\n📊 NUMERICAL FEATURES:")
                num_probs = logits_to_probs(transformer.num_tf_prob_logits.data, is_binary=False)
                num_selected = torch.argmax(num_probs, dim=1)
                
                for feat_idx, selected_idx in enumerate(num_selected):
                    selected_idx = selected_idx.item()
                    feature_name = transformer.num_columns[feat_idx] if feat_idx < len(transformer.num_columns) else f"num_feature_{feat_idx}"
                    selected_tf = transformer.num_tf_options[selected_idx]
                    tf_name = get_transformation_name(selected_tf)
                    prob = num_probs[feat_idx, selected_idx].item()
                    
                    print(f"  • {feature_name:30s} → {tf_name:30s} (prob: {prob:.3f})")
                    
                    transformer_info['features'].append({
                        'feature_name': feature_name,
                        'transformation': tf_name,
                        'probability': float(prob),
                        'type': 'numerical'
                    })
            
            if transformer.cat_tf_prob_logits is not None:
                print("\n📊 CATEGORICAL FEATURES (then one-hot encoded):")
                cat_probs = logits_to_probs(transformer.cat_tf_prob_logits.data, is_binary=False)
                cat_selected = torch.argmax(cat_probs, dim=1)
                
                # Get the one-hot encoded feature names
                if hasattr(transformer, 'one_hot_encoder'):
                    cat_feature_names = transformer.one_hot_encoder.get_feature_names_out(transformer.cat_columns)
                else:
                    cat_feature_names = [f"cat_feature_{i}" for i in range(len(cat_selected))]
                
                for feat_idx, selected_idx in enumerate(cat_selected):
                    selected_idx = selected_idx.item()
                    feature_name = cat_feature_names[feat_idx] if feat_idx < len(cat_feature_names) else f"cat_feature_{feat_idx}"
                    selected_tf = transformer.cat_tf_options[selected_idx]
                    tf_name = get_transformation_name(selected_tf)
                    prob = cat_probs[feat_idx, selected_idx].item()
                    
                    print(f"  • {feature_name:30s} → {tf_name:30s} (prob: {prob:.3f})")
                    
                    transformer_info['features'].append({
                        'feature_name': feature_name,
                        'transformation': tf_name,
                        'probability': float(prob),
                        'type': 'categorical'
                    })
        
        else:
            # Regular transformer - applies to all features
            if hasattr(transformer, 'tf_prob_logits'):
                probs = logits_to_probs(transformer.tf_prob_logits.data, is_binary=False)
                selected = torch.argmax(probs, dim=1)
                
                print(f"\n📊 TRANSFORMATIONS FOR {transformer.in_features} FEATURES:")
                
                # Count which transformations are selected
                tf_counts = {}
                for feat_idx, selected_idx in enumerate(selected):
                    selected_idx = selected_idx.item()
                    selected_tf = transformer.tf_options[selected_idx]
                    tf_name = get_transformation_name(selected_tf)
                    
                    if tf_name not in tf_counts:
                        tf_counts[tf_name] = 0
                    tf_counts[tf_name] += 1
                
                # Display summary
                for tf_name, count in sorted(tf_counts.items(), key=lambda x: -x[1]):
                    percentage = (count / transformer.in_features) * 100
                    print(f"  • {tf_name:30s} applied to {count:4d} features ({percentage:5.1f}%)")
                    
                    transformer_info['features'].append({
                        'transformation': tf_name,
                        'count': int(count),
                        'percentage': float(percentage)
                    })
                
                # Show individual feature details if there are not too many
                if transformer.in_features <= 20:
                    print(f"\n  Detailed feature-level transformations:")
                    for feat_idx, selected_idx in enumerate(selected):
                        selected_idx = selected_idx.item()
                        selected_tf = transformer.tf_options[selected_idx]
                        tf_name = get_transformation_name(selected_tf)
                        prob = probs[feat_idx, selected_idx].item()
                        
                        feature_name = transformer.feature_names[feat_idx] if hasattr(transformer, 'feature_names') and feat_idx < len(transformer.feature_names) else f"feature_{feat_idx}"
                        print(f"    [{feat_idx:2d}] {feature_name:25s} → {tf_name:25s} (prob: {prob:.3f})")
        
        config['transformers'].append(transformer_info)
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total pipeline steps: {len(pipeline.pipeline)}")
    print(f"Final output features: {pipeline.out_features}")
    print("="*80 + "\n")
    
    return config


def main():
    parser = argparse.ArgumentParser(description='Extract preprocessing configuration from saved pipeline')
    parser.add_argument('--dataset', required=True, help='Dataset name (e.g., abalone)')
    parser.add_argument('--method', default='diffprep_fix', help='Method name')
    parser.add_argument('--pipeline-dir', default='saved_pipelines', help='Directory containing saved pipelines')
    args = parser.parse_args()
    
    # Construct path to pipeline
    pipeline_path = os.path.join(args.pipeline_dir, args.method, args.dataset, 'pipeline.pkl')
    
    if not os.path.exists(pipeline_path):
        print(f"[ERROR] Pipeline not found at: {pipeline_path}")
        return
    
    print(f"\n[INFO] Loading pipeline from: {pipeline_path}")
    
    # Extract and display configuration
    config = extract_pipeline_config(pipeline_path)
    
    # Optionally save to JSON
    import json
    config_path = os.path.join(args.pipeline_dir, args.method, args.dataset, 'pipeline_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"[SUCCESS] Saved configuration to: {config_path}")


if __name__ == '__main__':
    main()
