"""
Script to create a summary CSV from all numeric dataset info.json files
"""

import os
import json
import pandas as pd

def collect_dataset_info(data_dir='data'):
    """Collect information from all numeric dataset folders"""
    
    summary_data = []
    
    # Get all numeric folders
    for folder_name in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder_name)
        
        # Skip non-directories and non-numeric folders
        if not os.path.isdir(folder_path):
            continue
        if not folder_name.isdigit():
            continue
            
        # Check for info.json
        info_path = os.path.join(folder_path, 'info.json')
        if not os.path.exists(info_path):
            print(f"Warning: {folder_name} missing info.json")
            continue
            
        # Load info.json
        try:
            with open(info_path, 'r') as f:
                info = json.load(f)
            
            # Extract relevant fields
            dataset_summary = {
                'dataset_id': folder_name,
                'dataset_name': info.get('dataset_name', 'N/A'),
                'task_type': info.get('task_type', 'N/A'),
                'num_samples': info.get('num_samples', 0),
                'num_features': info.get('num_features', 0),
                'num_classes': info.get('num_classes', 'N/A'),
                'num_missing_values': info.get('num_missing_values', 0),
                'missing_percentage': info.get('missing_percentage', 0.0),
                'num_outlier_cells': info.get('num_outlier_cells', info.get('num_outlier_samples', 0)),
                'outlier_percentage': info.get('outlier_percentage', 0.0),
                'openml_id': info.get('openml_id', folder_name)
            }
            
            summary_data.append(dataset_summary)
            print(f"Processed: {folder_name} - {info.get('dataset_name', 'N/A')}")
            
        except Exception as e:
            print(f"Error processing {folder_name}: {e}")
            continue
    
    return summary_data


if __name__ == "__main__":
    print("Collecting dataset information...")
    print("=" * 60)
    
    summary_data = collect_dataset_info('data')
    
    if not summary_data:
        print("\nNo datasets found!")
    else:
        # Create DataFrame
        df = pd.DataFrame(summary_data)
        
        # Sort by dataset_id
        df = df.sort_values('dataset_id')
        
        # Save to CSV
        output_file = 'dataset_summary.csv'
        df.to_csv(output_file, index=False)
        
        print("\n" + "=" * 60)
        print(f"Summary saved to: {output_file}")
        print(f"Total datasets: {len(df)}")
        print("=" * 60)
        
        # Print preview
        print("\nPreview of summary:")
        print(df.to_string(index=False))
