import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

RESULT_PATH = 'Data Curation - Result.csv'
PIPE_PATH = 'Data Curation - Pipeline Comparison.csv'
OUT_DIR = os.path.join('result', 'dataset_selection_visuals')
os.makedirs(OUT_DIR, exist_ok=True)


# ---- Load result file with two-row header ----
res = pd.read_csv(RESULT_PATH, header=[0, 1])
flat_cols = []
for a, b in res.columns:
    a = '' if str(a).startswith('Unnamed') else str(a).strip()
    b = '' if str(b).startswith('Unnamed') else str(b).strip()
    flat_cols.append((b if b else a).strip())
res.columns = flat_cols
res = res.loc[:, [c for c in res.columns if c != '']]

rename_map = {
    '#Ex.': 'n_samples',
    '#Feat.': 'n_features',
    '#Classes': 'n_classes',
    '#MVs': 'n_missing',
    '#Out.': 'outlier_rate',
    'Dataset': 'dataset_raw',
    'BASELINE (AUTOGLUON ONLY)': 'baseline_autogluon',
    'CTXpipe (CTXpipe space)': 'ctxpipe_ctx',
    'CTXpipe (Our space)': 'ctxpipe_ourspace',
    'DiffFix (DiffPrep space)': 'difffix',
    'FULL AUTOGLUON': 'full_autogluon',
    'Ours(our search\nspace)': 'ours',
}
for old, new in rename_map.items():
    if old in res.columns:
        res = res.rename(columns={old: new})

res['dataset_id'] = res['dataset_raw'].astype(str).str.extract(r'\((\d+)\)', expand=False)
res['dataset_id'] = res['dataset_id'].fillna(res['dataset_raw'].astype(str).str.extract(r'^(\d+)$', expand=False))

for col in ['n_samples', 'n_features', 'n_classes', 'n_missing']:
    if col in res.columns:
        res[col] = pd.to_numeric(res[col], errors='coerce')
if 'outlier_rate' in res.columns:
    res['outlier_rate'] = pd.to_numeric(res['outlier_rate'].astype(str).str.replace('%', '', regex=False), errors='coerce') / 100.0

score_cols = [
    c
    for c in ['baseline_autogluon', 'ctxpipe_ctx', 'ctxpipe_ourspace', 'difffix', 'full_autogluon', 'ours']
    if c in res.columns
]
for c in score_cols:
    res[c] = pd.to_numeric(res[c], errors='coerce')


# ---- Load pipeline file with two-row header ----
pipe_raw = pd.read_csv(PIPE_PATH, header=[0, 1])
pipe_cols = []
for a, b in pipe_raw.columns:
    a = '' if str(a).startswith('Unnamed') else str(a).strip()
    b = '' if str(b).startswith('Unnamed') else str(b).strip()
    pipe_cols.append((b if b else a).strip())
pipe_raw.columns = pipe_cols
pipe_raw = pipe_raw.loc[:, [c for c in pipe_raw.columns if c != '']]

if 'Dataset' not in pipe_raw.columns:
    first_col = pipe_raw.columns[0]
    pipe_raw = pipe_raw.rename(columns={first_col: 'Dataset'})

pipe_raw['dataset_id'] = pipe_raw['Dataset'].astype(str).str.extract(r'(\d+)', expand=False)

pipeline_cols = [c for c in pipe_raw.columns if ('CtxPipe' in c or 'Ours' in c)]
pipe_slim = pipe_raw[['dataset_id'] + pipeline_cols].drop_duplicates(subset=['dataset_id'])


# ---- Merge and filter ----
merged = res.merge(pipe_slim, on='dataset_id', how='left')
merged = merged.dropna(subset=['dataset_id'])

feat_thr = max(50, merged['n_features'].quantile(0.75))
out_thr = max(0.05, merged['outlier_rate'].quantile(0.75))

merged['is_hard'] = (merged['n_features'] >= feat_thr) & (merged['outlier_rate'] >= out_thr)
merged['best_score'] = merged[score_cols].max(axis=1)
merged['ours_gap_to_best'] = merged['best_score'] - merged['ours']
merged['ours_good'] = merged['ours_gap_to_best'] <= 0.01

selected = merged[merged['is_hard'] & merged['ours_good']].copy()
selected = selected.sort_values(['ours_gap_to_best', 'n_features', 'outlier_rate'], ascending=[True, False, False])

merged.to_csv(os.path.join(OUT_DIR, 'merged_analysis_table.csv'), index=False)
selected.to_csv(os.path.join(OUT_DIR, 'selected_datasets.csv'), index=False)


# ---- Visualizations ----
sns.set_theme(style='whitegrid')

# 1) Difficulty map
plt.figure(figsize=(9, 6))
plt.scatter(merged['n_features'], merged['outlier_rate'] * 100, alpha=0.35, label='All datasets')
if len(selected) > 0:
    plt.scatter(selected['n_features'], selected['outlier_rate'] * 100, color='crimson', label='Selected (hard + ours good)')
    for _, row in selected.head(12).iterrows():
        plt.annotate(str(row['dataset_id']), (row['n_features'], row['outlier_rate'] * 100), fontsize=8, alpha=0.9)
plt.axvline(feat_thr, color='gray', linestyle='--', linewidth=1)
plt.axhline(out_thr * 100, color='gray', linestyle='--', linewidth=1)
plt.xlabel('Number of features')
plt.ylabel('Outlier cells rate (%)')
plt.title('Dataset difficulty map and selected datasets')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '01_difficulty_map.png'), dpi=220)
plt.close()

# 2) Ours vs best on selected
if len(selected) > 0:
    show = selected.head(20).copy()
    plt.figure(figsize=(11, max(4, 0.35 * len(show))))
    y = np.arange(len(show))
    plt.barh(y, show['best_score'], color='#bdbdbd', label='Best method score')
    plt.barh(y, show['ours'], color='#1f77b4', alpha=0.9, label='Ours score')
    plt.yticks(y, show['dataset_id'])
    plt.xlabel('Test accuracy')
    plt.title('Selected hard datasets: Ours vs best achieved score')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, '02_ours_vs_best_selected.png'), dpi=220)
    plt.close()

# 3) Heatmap selected methods
methods_for_heatmap = [c for c in ['baseline_autogluon', 'ctxpipe_ctx', 'ctxpipe_ourspace', 'difffix', 'full_autogluon', 'ours'] if c in selected.columns]
if len(selected) > 0 and len(methods_for_heatmap) > 1:
    hm = selected[['dataset_id'] + methods_for_heatmap].set_index('dataset_id')
    plt.figure(figsize=(10, max(4, 0.35 * len(hm))))
    sns.heatmap(hm, annot=True, fmt='.3f', cmap='YlGnBu', cbar_kws={'label': 'Test accuracy'})
    plt.title('Method performance on selected hard datasets')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, '03_selected_methods_heatmap.png'), dpi=220)
    plt.close()

# 4) Export selected with pipelines
export_cols = ['dataset_id', 'dataset_raw', 'n_samples', 'n_features', 'outlier_rate', 'ours', 'best_score', 'ours_gap_to_best'] + [c for c in pipeline_cols if c in selected.columns]
export_cols = [c for c in export_cols if c in selected.columns]
selected[export_cols].to_csv(os.path.join(OUT_DIR, '04_selected_with_pipelines.csv'), index=False)

# 5) Summary markdown
with open(os.path.join(OUT_DIR, 'SUMMARY.md'), 'w', encoding='utf-8') as f:
    f.write('# Selected datasets for visualization\n\n')
    f.write(f'- Total datasets analyzed: {len(merged)}\n')
    f.write(f'- Difficulty threshold: n_features >= {feat_thr:.1f}, outlier_rate >= {out_thr * 100:.2f}%\n')
    f.write(f'- Selected datasets count: {len(selected)}\n\n')
    if len(selected) == 0:
        f.write('No dataset met both criteria with current thresholds.\n')
    else:
        f.write('## Selected dataset IDs\n\n')
        f.write(', '.join(selected['dataset_id'].astype(str).tolist()) + '\n\n')
        f.write('## Top rows\n\n')
        preview_cols = [c for c in ['dataset_id', 'dataset_raw', 'n_samples', 'n_features', 'outlier_rate', 'ours', 'best_score', 'ours_gap_to_best'] if c in selected.columns]
        f.write(selected[preview_cols].head(30).to_markdown(index=False))

print('DONE')
print('OUT_DIR=', OUT_DIR)
print('ANALYZED=', len(merged), 'SELECTED=', len(selected))
print('FEATURE_THR=', feat_thr, 'OUTLIER_THR=', out_thr)
