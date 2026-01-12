"""
Run DiffPrep training and AutoGluon evaluation on all text-named datasets
Chạy diffprep_fix và đánh giá với AutoGluon trên các dataset có tên bằng chữ
"""

import subprocess
import sys
import os
import json
import time
from pathlib import Path

# Danh sách các dataset có tên bằng chữ (text names)
TEXT_DATASETS = [
    'abalone',
    'ada_prior',
    'avila',
    'connect-4',
    'eeg',
    'google',
    'house_prices',
    'jungle_chess_2pcs_raw_endgame_complete',
    'microaggregation2',
    'mozilla4',
    'obesity',
    'page-blocks',
    'pbcseq',
    'pol',
    'Run_or_walk_information',
    'shuttle',
    'USCensus',
    'wall-robot-navigation'
]

# Cấu hình
METHOD = "diffprep_fix"
SPLIT_SEED = 42
TRAIN_SEED = 1
AUTOGLUON_TIME_LIMIT = 300  # 10 phút cho mỗi dataset
DATA_DIR = "data"
RESULT_DIR = "result"
AUTOGLUON_OUTPUT_DIR = "autogluon_results"


def run_command(cmd, description):
    """Chạy lệnh và xử lý lỗi"""
    print("\n" + "="*80)
    print(f"{description}")
    print("="*80)
    print(f"Command: {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ LỖI: {description} thất bại!")
        print(f"Return code: {e.returncode}")
        return False
    except Exception as e:
        print(f"\n❌ LỖI: {description} thất bại với exception: {e}")
        return False


def check_dataset_exists(dataset_name, data_dir=DATA_DIR):
    """Kiểm tra dataset có tồn tại không"""
    dataset_path = os.path.join(data_dir, dataset_name)
    data_file = os.path.join(dataset_path, "data.csv")
    info_file = os.path.join(dataset_path, "info.json")
    
    if not os.path.exists(dataset_path):
        print(f"⚠️  Bỏ qua {dataset_name}: Thư mục không tồn tại")
        return False
    
    if not os.path.exists(data_file):
        print(f"⚠️  Bỏ qua {dataset_name}: Không tìm thấy data.csv")
        return False
        
    if not os.path.exists(info_file):
        print(f"⚠️  Bỏ qua {dataset_name}: Không tìm thấy info.json")
        return False
    
    return True


def run_diffprep_training(dataset_name):
    """Chạy DiffPrep training"""
    cmd = [
        sys.executable,
        "main.py",
        "--dataset", dataset_name,
        "--method", METHOD,
        "--train_seed", str(TRAIN_SEED),
        "--split_seed", str(SPLIT_SEED),
        "--data_dir", DATA_DIR
    ]
    
    return run_command(cmd, f"Đang train DiffPrep cho {dataset_name}")


def run_autogluon_evaluation(dataset_name):
    """Chạy AutoGluon evaluation"""
    cmd = [
        sys.executable,
        "evaluate_with_autogluon.py",
        "--dataset", dataset_name,
        "--method", METHOD,
        "--split_seed", str(SPLIT_SEED),
        "--time_limit", str(AUTOGLUON_TIME_LIMIT),
        "--data_dir", DATA_DIR,
        "--result_dir", RESULT_DIR,
        "--output_dir", AUTOGLUON_OUTPUT_DIR
    ]
    
    return run_command(cmd, f"Đang đánh giá AutoGluon cho {dataset_name}")


def save_summary(results, output_file="text_datasets_summary.json"):
    """Lưu tổng kết kết quả"""
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n📊 Đã lưu tổng kết vào {output_file}")


def print_summary(results):
    """In tổng kết kết quả"""
    print("\n" + "="*80)
    print("📊 TỔNG KẾT KẾT QUẢ")
    print("="*80)
    
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] == 'failed']
    skipped = [r for r in results if r['status'] == 'skipped']
    
    print(f"\n✅ Thành công: {len(successful)}/{len(results)}")
    print(f"❌ Thất bại: {len(failed)}/{len(results)}")
    print(f"⏭️  Bỏ qua: {len(skipped)}/{len(results)}")
    
    if successful:
        print("\n✅ Các dataset thành công:")
        for r in successful:
            print(f"  - {r['dataset']}")
            if 'autogluon_acc' in r:
                print(f"    AutoGluon Test Acc: {r['autogluon_acc']:.4f}")
    
    if failed:
        print("\n❌ Các dataset thất bại:")
        for r in failed:
            print(f"  - {r['dataset']}: {r.get('error', 'Unknown error')}")
    
    if skipped:
        print("\n⏭️  Các dataset bỏ qua:")
        for r in skipped:
            print(f"  - {r['dataset']}: {r.get('reason', 'Unknown reason')}")


def main():
    """Hàm chính"""
    print("="*80)
    print("🚀 BẮT ĐẦU CHẠY DIFFPREP VÀ AUTOGLUON CHO CÁC DATASET VĂN BẢN")
    print("="*80)
    print(f"Phương pháp: {METHOD}")
    print(f"Split seed: {SPLIT_SEED}")
    print(f"Train seed: {TRAIN_SEED}")
    print(f"AutoGluon time limit: {AUTOGLUON_TIME_LIMIT}s")
    print(f"Tổng số dataset: {len(TEXT_DATASETS)}")
    print()
    
    results = []
    start_time = time.time()
    
    for i, dataset in enumerate(TEXT_DATASETS, 1):
        print(f"\n{'='*80}")
        print(f"📦 DATASET {i}/{len(TEXT_DATASETS)}: {dataset}")
        print(f"{'='*80}")
        
        dataset_start_time = time.time()
        result = {
            'dataset': dataset,
            'status': 'unknown',
            'diffprep_time': 0,
            'autogluon_time': 0
        }
        
        # Kiểm tra dataset tồn tại
        if not check_dataset_exists(dataset):
            result['status'] = 'skipped'
            result['reason'] = 'Dataset không tồn tại hoặc thiếu file'
            results.append(result)
            continue
        
        # Bước 1: Train DiffPrep
        print(f"\n🔧 Bước 1/2: Train DiffPrep...")
        diffprep_start = time.time()
        if not run_diffprep_training(dataset):
            result['status'] = 'failed'
            result['error'] = 'DiffPrep training failed'
            results.append(result)
            print(f"\n⏭️  Bỏ qua {dataset} do lỗi training")
            continue
        result['diffprep_time'] = time.time() - diffprep_start
        
        # Bước 2: Evaluate với AutoGluon
        print(f"\n📈 Bước 2/2: Đánh giá với AutoGluon...")
        autogluon_start = time.time()
        if not run_autogluon_evaluation(dataset):
            result['status'] = 'failed'
            result['error'] = 'AutoGluon evaluation failed'
            results.append(result)
            print(f"\n⚠️  {dataset}: Training thành công nhưng evaluation thất bại")
            continue
        result['autogluon_time'] = time.time() - autogluon_start
        
        # Đọc kết quả AutoGluon
        try:
            ag_result_file = os.path.join(
                AUTOGLUON_OUTPUT_DIR,
                METHOD,
                dataset,
                "result.json"
            )
            if os.path.exists(ag_result_file):
                with open(ag_result_file, 'r') as f:
                    ag_result = json.load(f)
                    result['autogluon_acc'] = ag_result.get('test_acc', 0)
        except Exception as e:
            print(f"⚠️  Không đọc được kết quả AutoGluon: {e}")
        
        result['status'] = 'success'
        result['total_time'] = time.time() - dataset_start_time
        results.append(result)
        
        print(f"\n✅ Hoàn thành {dataset} trong {result['total_time']:.1f}s")
        print(f"   - DiffPrep: {result['diffprep_time']:.1f}s")
        print(f"   - AutoGluon: {result['autogluon_time']:.1f}s")
        
        # Lưu tổng kết sau mỗi dataset
        save_summary(results)
    
    # In tổng kết cuối cùng
    total_time = time.time() - start_time
    print(f"\n⏱️  Tổng thời gian: {total_time/60:.1f} phút")
    
    print_summary(results)
    save_summary(results)
    
    print("\n" + "="*80)
    print("🎉 HOÀN THÀNH!")
    print("="*80)


if __name__ == "__main__":
    main()
