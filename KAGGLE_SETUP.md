# Running DiffPrep Experiments on Kaggle

This guide will help you run text-named dataset experiments on Kaggle notebooks.

## Setup Instructions

### Step 1: Create a New Kaggle Notebook

1. Go to [kaggle.com/code](https://www.kaggle.com/code)
2. Click "New Notebook"
3. Enable GPU/TPU if needed (Settings → Accelerator)
4. Set Internet to "On" (Settings → Internet)

### Step 2: Upload Your Code

**Option A: Upload as a Dataset**

1. Zip your DiffPrep folder (excluding large files like results)
2. Upload to Kaggle Datasets
3. Add the dataset to your notebook

**Option B: Clone from GitHub (if available)**

```python
!git clone https://github.com/yourusername/DiffPrep.git
%cd DiffPrep
```

**Option C: Direct Upload**
Upload individual files using the "Add Data" → "Upload" button

### Step 3: Install Dependencies

```python
# Install required packages
!pip install -q impyute matplotlib numpy pandas scikit-learn scipy torch tqdm

# Or use requirements.txt if uploaded
!pip install -q -r requirements.txt
```

### Step 4: Verify Setup

```python
# Check if data directory exists
import os
print("Data folders:", os.listdir('data'))

# Check Python version
!python --version

# Check GPU availability
import torch
print(f"GPU available: {torch.cuda.is_available()}")
```

## Running Experiments

### Option 1: Run All Text Datasets (Recommended for Kaggle)

```python
# Run all experiments
!python run_text_datasets_experiment.py
```

### Option 2: Run Specific Datasets (Faster Testing)

```python
# Test with a few datasets first
!python run_text_datasets_flexible.py --datasets abalone eeg google --methods diffprep_fix
```

### Option 3: Run with Kaggle-Optimized Settings

```python
# Run selected datasets with fewer methods to save time
!python run_text_datasets_flexible.py \
    --datasets abalone avila eeg google house_prices obesity page-blocks pbcseq \
    --methods diffprep_fix default \
    --skip_existing
```

### Option 4: Run Single Dataset for Testing

```python
# Quick test with one dataset
!python main.py --dataset abalone --method diffprep_fix --model log
```

## Complete Kaggle Notebook Example

```python
# Cell 1: Setup and Installation
!pip install -q impyute scipy tqdm

# Cell 2: Navigate to code directory (if uploaded as dataset)
import os
os.chdir('/kaggle/input/diffprep-code')  # Adjust path as needed

# Cell 3: List available datasets
!python run_text_datasets_flexible.py --list

# Cell 4: Run experiments (choose one)
# Option A: Run all text datasets
!python run_text_datasets_experiment.py

# Option B: Run subset for faster execution
# !python run_text_datasets_flexible.py --datasets abalone eeg google --methods diffprep_fix default

# Cell 5: Check results
!ls -la result/

# Cell 6: View log file
!cat experiment_log_*.txt

# Cell 7: Download results (if needed)
from IPython.display import FileLink
import glob

log_files = glob.glob('experiment_log_*.txt')
for log_file in log_files:
    display(FileLink(log_file))
```

## Kaggle-Specific Tips

### 1. Time Limits

Kaggle notebooks have execution time limits:

- Free tier: ~9-12 hours
- GPU sessions: ~12 hours

**Strategy:** Run subsets of datasets or use `--skip_existing` to resume

### 2. Save Intermediate Results

```python
# Add to notebook cells periodically
from kaggle_datasets import KaggleDatasets
import shutil

# Save results
shutil.make_archive('results_backup', 'zip', 'result/')
shutil.make_archive('logs_backup', 'zip', '.', base_dir='experiment_log_*.txt')
```

### 3. Monitor Progress

```python
# In a separate cell, monitor while running
import time
import glob

while True:
    logs = glob.glob('experiment_log_*.txt')
    if logs:
        with open(logs[-1], 'r') as f:
            print(f.read()[-1000:])  # Print last 1000 chars
    time.sleep(60)  # Check every minute
```

### 4. Optimize for Kaggle

Run fewer experiments by selecting key datasets:

```python
# Priority datasets (smaller, faster)
PRIORITY_DATASETS = [
    'abalone',
    'eeg',
    'obesity',
    'page-blocks',
    'pbcseq'
]

!python run_text_datasets_flexible.py --datasets {' '.join(PRIORITY_DATASETS)} --methods diffprep_fix default
```

## Downloading Results

### Method 1: Direct Download from Notebook

```python
from IPython.display import FileLink
import shutil

# Create zip of results
shutil.make_archive('experiment_results', 'zip', 'result/')
display(FileLink('experiment_results.zip'))
```

### Method 2: Save as Kaggle Dataset

```python
# This will appear in your Kaggle datasets
# Save output version in notebook settings
```

### Method 3: Copy to Kaggle Output

```python
import shutil
import os

# Kaggle automatically saves /kaggle/working/ directory
if os.path.exists('/kaggle/working/'):
    shutil.copytree('result/', '/kaggle/working/result/')
    for log in glob.glob('experiment_log_*.txt'):
        shutil.copy(log, '/kaggle/working/')
```

## Troubleshooting

### Import Errors

```python
# Ensure all packages are installed
!pip install impyute matplotlib numpy pandas scikit-learn scipy torch tqdm
```

### Path Issues

```python
# Check current directory
import os
print("Current dir:", os.getcwd())
print("Files:", os.listdir('.'))

# Adjust paths if needed
os.chdir('/kaggle/working')  # or wherever your code is
```

### Memory Issues

- Run fewer datasets at a time
- Use `--methods diffprep_fix` only (skip other methods)
- Clear outputs between runs: `!rm -rf result/*/`

### Session Timeout

- Use `--skip_existing` to resume interrupted runs
- Save results periodically
- Split into multiple notebook runs

## Quick Start Commands

```bash
# Minimal install
!pip install -q impyute scipy tqdm

# Quick test (1 dataset, 1 method, ~5-10 min)
!python run_text_datasets_flexible.py --datasets abalone --methods diffprep_fix

# Medium run (5 datasets, 2 methods, ~1-2 hours)
!python run_text_datasets_flexible.py --datasets abalone eeg google obesity page-blocks --methods diffprep_fix default

# Full run (18 datasets, 4 methods, ~6-8 hours)
!python run_text_datasets_experiment.py
```
