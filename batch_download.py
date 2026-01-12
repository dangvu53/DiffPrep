"""
Batch download multiple datasets from OpenML
"""

import sys
from download_openml_dataset import load_openml_dataset, save_dataset

# List of dataset IDs to download
DATASET_IDS = [
    40975, 1233, 1115, 1466, 248, 279, 40740, 803, 942, 373, 
    1518, 737, 1396, 1399, 823, 253, 922, 7, 1066, 1164, 932,
    974, 1047, 991, 244, 1400, 862, 40520, 2, 40663, 1054,
    1387, 1397, 1401, 1393, 728, 876, 1358, 75, 18
]

def main():
    # Check if --full flag is present
    full_dataset = '--full' in sys.argv
    max_samples = None if full_dataset else 5000
    
    print("=" * 60)
    print(f"Batch downloading {len(DATASET_IDS)} datasets from OpenML")
    print(f"Mode: {'Full dataset' if full_dataset else f'Limited to {max_samples} samples'}")
    print("=" * 60)
    print()
    
    successful = []
    failed = []
    
    for i, dataset_id in enumerate(DATASET_IDS, 1):
        print(f"\n[{i}/{len(DATASET_IDS)}] Processing dataset {dataset_id}")
        print("-" * 60)
        
        try:
            # Load dataset from OpenML
            dataset_data = load_openml_dataset(dataset_id, max_samples=max_samples)
            
            if dataset_data:
                # Use dataset ID as folder name
                folder_name = str(dataset_id)
                print(f"Dataset name: {dataset_data['name']}")
                
                # Save to data folder
                success = save_dataset(dataset_data, folder_name)
                
                if success:
                    successful.append(dataset_id)
                    print(f"✓ Successfully saved dataset {dataset_id}")
                else:
                    failed.append(dataset_id)
                    print(f"✗ Failed to save dataset {dataset_id}")
            else:
                failed.append(dataset_id)
                print(f"✗ Failed to load dataset {dataset_id}")
                
        except Exception as e:
            failed.append(dataset_id)
            print(f"✗ Error processing dataset {dataset_id}: {e}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("BATCH DOWNLOAD SUMMARY")
    print("=" * 60)
    print(f"Total datasets: {len(DATASET_IDS)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if failed:
        print(f"\nFailed dataset IDs: {failed}")
    
    print("\nAll downloads complete!")

if __name__ == "__main__":
    main()
