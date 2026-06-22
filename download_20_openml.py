import json
import os
import re
from sklearn.datasets import fetch_openml
from download_openml_dataset import load_openml_dataset, save_dataset

IDS = [248, 1066, 1164, 1047, 862, 2, 40663, 1054, 1387, 876, 18, 1520, 1548, 184, 378, 381, 382, 993, 1485, 14]
MAX_SAMPLES = 100000

KEYWORD_DOMAINS = [
    ("Healthcare/Biomedical", ["medical", "health", "disease", "patient", "hospital", "clinical", "diagnosis", "obesity", "eeg", "cancer", "survival", "pbc"]),
    ("Finance/Economics", ["credit", "bank", "loan", "finance", "income", "census", "price", "housing", "house", "insurance"]),
    ("Web/Internet", ["web", "mozilla", "google", "page", "click", "ad", "search", "browser"]),
    ("Games/Strategy", ["chess", "game", "endgame", "connect-4", "board"]),
    ("Biology/Ecology", ["bio", "species", "ecology", "animal", "abalone", "yeast", "genome"]),
    ("Robotics/Sensors/Signal", ["robot", "sensor", "signal", "walk", "navigation", "activity", "motion", "time series"]),
    ("Computer Vision/Pattern Recognition", ["image", "vision", "digit", "handwritten", "pixel", "classification"]),
]


def infer_domain(name, descr):
    text = f"{name} {descr}".lower()
    text = re.sub(r"\s+", " ", text)
    for domain, keywords in KEYWORD_DOMAINS:
        if any(keyword in text for keyword in keywords):
            return domain
    return "General/Unspecified"


def main():
    results = []
    successful = []
    failed = []

    print("=" * 72)
    print(f"Downloading {len(IDS)} OpenML datasets with max_samples={MAX_SAMPLES}")
    print("=" * 72)

    for i, dataset_id in enumerate(IDS, 1):
        print(f"\n[{i}/{len(IDS)}] Dataset {dataset_id}")
        print("-" * 72)
        try:
            ds = load_openml_dataset(dataset_id, max_samples=MAX_SAMPLES)
            if ds is None:
                failed.append(dataset_id)
                results.append({"id": dataset_id, "status": "failed", "reason": "load_failed"})
                continue

            ok = save_dataset(ds, str(dataset_id), data_folder="data")
            if not ok:
                failed.append(dataset_id)
                results.append({"id": dataset_id, "status": "failed", "reason": "save_failed"})
                continue

            domain = "General/Unspecified"
            try:
                meta = fetch_openml(data_id=dataset_id, as_frame=True, parser="auto")
                descr = getattr(meta, "DESCR", "") or ""
                domain = infer_domain(ds["name"], descr)
            except Exception:
                pass

            successful.append(dataset_id)
            results.append(
                {
                    "id": dataset_id,
                    "status": "success",
                    "dataset_name": ds["name"],
                    "task_type": ds["task_type"],
                    "num_samples_saved": int(len(ds["X"])),
                    "domain": domain,
                    "path": os.path.join("data", str(dataset_id)),
                }
            )
            print(f"Saved -> data/{dataset_id} | domain: {domain}")

        except Exception as exc:
            failed.append(dataset_id)
            results.append({"id": dataset_id, "status": "failed", "reason": str(exc)})
            print(f"Error: {exc}")

    summary = {
        "requested_ids": IDS,
        "max_samples": MAX_SAMPLES,
        "successful": successful,
        "failed": failed,
        "results": results,
    }

    out_path = os.path.join("data", "openml_20_download_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 72)
    print(f"Done. Successful: {len(successful)} | Failed: {len(failed)}")
    print(f"Summary file: {out_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
