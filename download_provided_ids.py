import json
import os

from download_openml_dataset import load_openml_dataset, save_dataset

IDS = [
    248, 1066, 1164, 1047, 862, 2, 40663, 1054, 1387, 876,
    18, 1520, 1548, 184, 378, 381, 382, 993, 1485, 14,
]
MAX_SAMPLES = 100000


def main():
    successful = []
    failed = []

    print("=" * 72)
    print(f"Downloading {len(IDS)} OpenML datasets with max_samples={MAX_SAMPLES}")
    print("=" * 72)

    for i, dataset_id in enumerate(IDS, 1):
        print(f"\n[{i}/{len(IDS)}] Dataset {dataset_id}")
        print("-" * 72)
        try:
            dataset_data = load_openml_dataset(dataset_id, max_samples=MAX_SAMPLES)
            if dataset_data is None:
                failed.append(dataset_id)
                print(f"✗ Failed to load dataset {dataset_id}")
                continue

            ok = save_dataset(dataset_data, str(dataset_id), data_folder="data")
            if ok:
                successful.append(dataset_id)
                print(f"✓ Saved dataset {dataset_id} to data/{dataset_id}")
            else:
                failed.append(dataset_id)
                print(f"✗ Failed to save dataset {dataset_id}")
        except Exception as exc:
            failed.append(dataset_id)
            print(f"✗ Error for dataset {dataset_id}: {exc}")

    summary = {
        "requested_ids": IDS,
        "max_samples": MAX_SAMPLES,
        "successful": successful,
        "failed": failed,
    }

    output_path = os.path.join("data", "provided_ids_download_summary.json")
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("\n" + "=" * 72)
    print(f"Done. Successful: {len(successful)} | Failed: {len(failed)}")
    print(f"Summary: {output_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
