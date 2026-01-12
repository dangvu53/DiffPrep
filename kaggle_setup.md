# Hướng dẫn chạy DiffPrep trên Kaggle

## Bước 1: Tạo Kaggle Notebook mới

1. Truy cập https://www.kaggle.com/code
2. Click "New Notebook"
3. Chọn "Notebook" (Python)

## Bước 2: Setup Environment

### 2.1. Upload code lên Kaggle

Có 2 cách:

**Cách 1: Upload trực tiếp (khuyến nghị cho lần đầu)**

```python
# Tạo cell đầu tiên trong notebook
!pip install kaggle
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()

# Hoặc upload thủ công bằng cách:
# 1. Zip toàn bộ folder DiffPrep
# 2. Add Dataset mới trên Kaggle
# 3. Import dataset vào notebook
```

**Cách 2: Clone từ GitHub**

```bash
# Cell 1: Clone repository
!git clone https://github.com/YOUR_USERNAME/DiffPrep.git
%cd DiffPrep
```

### 2.2. Install dependencies

```bash
# Cell 2: Cài đặt packages
!pip install -r requirements.txt
```

### 2.3. Verify installation

```python
# Cell 3: Kiểm tra imports
import torch
import pandas as pd
import numpy as np
from sklearn import __version__ as sklearn_version
import autogluon
from autogluon.tabular import TabularPredictor

print(f"PyTorch: {torch.__version__}")
print(f"Pandas: {pd.__version__}")
print(f"NumPy: {np.__version__}")
print(f"Scikit-learn: {sklearn_version}")
print(f"AutoGluon: {autogluon.__version__}")
```

## Bước 3: Upload dữ liệu

### 3.1. Tạo Kaggle Dataset cho dữ liệu

```bash
# Trên máy local, zip folder data
# Windows PowerShell:
Compress-Archive -Path "data\*" -DestinationPath "diffprep_data.zip"

# Hoặc tải từng dataset lên Kaggle Datasets
```

### 3.2. Import dataset vào notebook

```python
# Cell 4: Setup data path
import os
# Nếu upload dataset với tên "diffprep-data"
data_dir = '/kaggle/input/diffprep-data'
# Hoặc sử dụng data có sẵn trong code
data_dir = './data'
```

## Bước 4: Chạy thực nghiệm

### 4.1. Test trên 1 dataset nhỏ

```python
# Cell 5: Test trên dataset nhỏ (18)
!python main.py --dataset 18 --method diffprep_fix --split_seed 1 --time_limit 300
```

### 4.2. Chạy trên tất cả text datasets

```python
# Cell 6: Chạy toàn bộ text datasets
!python run_text_datasets.py
```

### 4.3. Hoặc chạy từng dataset riêng

```python
# Cell 7: Chạy từng dataset với vòng lặp
import subprocess
import json

datasets = [
    'abalone', 'ada_prior', 'avila', 'connect-4', 'eeg', 'google',
    'house_prices', 'jungle_chess_2pcs_raw_endgame_complete',
    'microaggregation2', 'mozilla4', 'obesity', 'page-blocks',
    'pbcseq', 'pol', 'Run_or_walk_information', 'shuttle',
    'USCensus', 'wall-robot-navigation'
]

results = {}

for dataset in datasets:
    print(f"\n{'='*60}")
    print(f"Processing dataset: {dataset}")
    print(f"{'='*60}\n")

    try:
        # Chạy DiffPrep
        subprocess.run([
            'python', 'main.py',
            '--dataset', dataset,
            '--method', 'diffprep_fix',
            '--split_seed', '1',
            '--time_limit', '300'
        ], check=True)

        # Chạy AutoGluon evaluation
        subprocess.run([
            'python', 'evaluate_with_autogluon.py',
            '--dataset', dataset,
            '--method', 'diffprep_fix',
            '--split_seed', '1',
            '--time_limit', '300'
        ], check=True)

        results[dataset] = "SUCCESS"

    except Exception as e:
        print(f"Error processing {dataset}: {str(e)}")
        results[dataset] = f"FAILED: {str(e)}"

# Lưu kết quả
with open('kaggle_results_summary.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n" + "="*60)
print("Summary:")
print("="*60)
for dataset, status in results.items():
    print(f"{dataset}: {status}")
```

## Bước 5: Thu thập kết quả

### 5.1. Xem kết quả AutoGluon

```python
# Cell 8: Load và hiển thị kết quả
import json
import pandas as pd

# Load experiment summary
with open('autogluon_results/diffprep_fix/experiment_summary.json', 'r') as f:
    summary = json.load(f)

# Convert to DataFrame
df_results = pd.DataFrame(summary).T
df_results = df_results.sort_values('test_accuracy', ascending=False)

print("\nTop 10 Best Results:")
print(df_results[['test_accuracy', 'best_pipeline']].head(10))

print(f"\nAverage Test Accuracy: {df_results['test_accuracy'].mean():.4f}")
print(f"Median Test Accuracy: {df_results['test_accuracy'].median():.4f}")
```

### 5.2. Download kết quả

```python
# Cell 9: Zip kết quả để download
!zip -r results.zip result/ autogluon_results/

# Kaggle sẽ tự động lưu file output, bạn có thể download sau khi notebook chạy xong
```

## Bước 6: Tối ưu hóa cho Kaggle

### 6.1. Settings cho Kaggle Notebook

- **Accelerator**: None (không cần GPU cho preprocessing)
- **Persistence**: Files only (lưu kết quả)
- **Internet**: On (để cài packages)

### 6.2. Tăng tốc độ

```python
# Giảm time_limit nếu muốn chạy nhanh hơn
!python main.py --dataset 18 --method diffprep_fix --time_limit 60  # 1 phút thay vì 5 phút

# Hoặc chạy parallel cho nhiều datasets (cẩn thận với RAM)
from concurrent.futures import ProcessPoolExecutor
import subprocess

def run_experiment(dataset):
    subprocess.run([
        'python', 'main.py',
        '--dataset', dataset,
        '--method', 'diffprep_fix',
        '--time_limit', '180'
    ])
    return dataset

with ProcessPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(run_experiment, ds) for ds in datasets[:4]]
    for future in futures:
        print(f"Completed: {future.result()}")
```

## Lưu ý quan trọng

1. **RAM limit**: Kaggle free tier có 13GB RAM, cẩn thận với datasets lớn
2. **Time limit**: Kaggle notebook có timeout 9 giờ, plan accordingly
3. **Persistent storage**: Kết quả sẽ mất sau khi session kết thúc, nhớ download
4. **AutoGluon**: Có thể tốn nhiều thời gian, cân nhắc giảm time_limit

## Commands tóm tắt cho Kaggle

```bash
# Setup
!git clone YOUR_REPO_URL || echo "Upload code manually"
%cd DiffPrep
!pip install -r requirements.txt

# Test
!python main.py --dataset 18 --method diffprep_fix --time_limit 300

# Run all
!python run_text_datasets.py

# Collect results
!zip -r results.zip result/ autogluon_results/
```
