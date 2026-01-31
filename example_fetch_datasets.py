"""
Example script demonstrating how to fetch and save OpenML datasets
"""

from fetch_and_save_openml_datasets import fetch_and_save_datasets

# Define the dataset IDs you want to fetch from OpenML
dataset_ids = [
167,
1485,
23512,
300,
188,
28,
    # Add more dataset IDs here as needed
]

# Optional: Define datasets that should have larger sample limit (100k instead of 5k)
test_dataset_ids = [
167,
1485,
23512,
300,
188,
28,
    # 23512,  # higgs (will allow up to 100k samples)
]

# Fetch and save the datasets
print("Starting dataset download from OpenML...")
successful, failed = fetch_and_save_datasets(
    dataset_ids=dataset_ids,
    save_dir='data',  # Save to the 'data' folder
    test_dataset_ids=test_dataset_ids
)

print("\n" + "="*60)
print("DONE!")
print("="*60)
print(f"Check the 'data' folder for the downloaded datasets.")
print(f"Each dataset will be in a folder named with its OpenML ID.")
