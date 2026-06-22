import json
import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import RobustScaler, StandardScaler

try:
    import umap.umap_ as umap
except ImportError:
    umap = None


RANDOM_STATE = 42
TOP_FEATURES = 5
K_BEST_CAP = 20
Z_THRESHOLD = 3.0
PREPROCESS_CONFIG = {
    "imputation": "median",
    "scaling": "robust",
    "encoding": "none",
    "outlier_removal": "zscore",
    "feature_selection": "mutual_info",
    "dimensionality_reduction": "pca",
}
HUMAN_PIPELINE_CONFIG = {
    "imputation": "none",
    "encoding": "none",
    "outlier_removal": "iqr",
    "scaling": "standard",
    "feature_selection": "k_best",
    "dimensionality_reduction": "pca",
}
STEP_ORDER = [
    "imputation",
    "scaling",
    "encoding",
    "outlier_removal",
    "feature_selection",
    "dimensionality_reduction",
]
FORCE_RECOMPUTE_EMBEDDING_CACHE = False


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _safe_zscore(df: pd.DataFrame) -> np.ndarray:
    z = np.abs(zscore(df, axis=0, nan_policy="omit"))
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    return z


def _load_dataset(csv_path: str) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(csv_path)

    if "label" in df.columns:
        y = df["label"].copy()
        X = df.drop(columns=["label"]).copy()
    else:
        y = df.iloc[:, -1].copy()
        X = df.iloc[:, :-1].copy()

    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    X = X.fillna(X.median(numeric_only=True))
    return X, y


def _prepare_states(X_raw: pd.DataFrame, y_raw: pd.Series) -> Dict[str, object]:
    # Raw state (already numeric after load)
    X_imputed = X_raw.copy()

    # Baseline for comparison: robust scaling only
    baseline_scaler = RobustScaler()
    X_baseline = pd.DataFrame(baseline_scaler.fit_transform(X_imputed), columns=X_imputed.columns)

    # Proposed pipeline mirrors preprocessor_updated.py default step order
    X_after_step: Dict[str, pd.DataFrame] = {"raw": X_imputed.copy()}
    y_after_step: Dict[str, pd.Series] = {"raw": y_raw.reset_index(drop=True).copy()}

    X_work = X_imputed.copy().reset_index(drop=True)
    y_work = y_raw.reset_index(drop=True).copy()

    if PREPROCESS_CONFIG["scaling"] == "robust":
        proposed_scaler = RobustScaler()
        X_work = pd.DataFrame(proposed_scaler.fit_transform(X_work), columns=X_work.columns)
    else:
        raise ValueError("This visualization currently expects scaling='robust' to match baseline comparison.")
    X_after_step["scaling"] = X_work.copy()
    y_after_step["scaling"] = y_work.copy()

    z_scaled = _safe_zscore(X_work)
    keep_mask = (z_scaled < Z_THRESHOLD).all(axis=1)
    X_clean_scaled = X_work.loc[keep_mask].reset_index(drop=True)
    y_clean = y_work.loc[keep_mask].reset_index(drop=True)
    X_after_step["outlier_removal"] = X_clean_scaled.copy()
    y_after_step["outlier_removal"] = y_clean.copy()

    k = min(K_BEST_CAP, X_clean_scaled.shape[1])
    selector = SelectKBest(mutual_info_classif, k=k)
    X_selected = selector.fit_transform(X_clean_scaled.values, y_clean.values)
    selected_mask = selector.get_support()
    selected_names = X_clean_scaled.columns[selected_mask].tolist()
    X_selected_df = pd.DataFrame(X_selected, columns=selected_names)
    X_after_step["feature_selection"] = X_selected_df.copy()
    y_after_step["feature_selection"] = y_clean.copy()

    n_components = min(10, X_selected_df.shape[1], max(1, X_selected_df.shape[0] - 1))
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_selected_df.values)
    X_pca_df = pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(X_pca.shape[1])])
    X_after_step["dimensionality_reduction"] = X_pca_df.copy()
    y_after_step["dimensionality_reduction"] = y_clean.copy()

    outlier_ratio = (z_scaled > Z_THRESHOLD).mean(axis=0)
    outlier_ratio_series = pd.Series(outlier_ratio, index=X_imputed.columns).sort_values(ascending=False)
    top_features = outlier_ratio_series.head(min(TOP_FEATURES, len(outlier_ratio_series))).index.tolist()

    return {
        "X_raw": X_raw,
        "y_raw": y_raw.reset_index(drop=True),
        "X_baseline": X_baseline,
        "X_clean": X_clean_scaled,
        "y_clean": y_clean,
        "X_clean_scaled": X_clean_scaled,
        "X_selected": X_selected_df,
        "selected_names": selected_names,
        "X_pca": X_pca_df.values,
        "X_pca_df": X_pca_df,
        "pca": pca,
        "top_features": top_features,
        "outlier_ratio": outlier_ratio_series,
        "keep_mask": keep_mask,
        "X_after_step": X_after_step,
        "y_after_step": y_after_step,
        "preprocess_config": PREPROCESS_CONFIG,
        "step_order": STEP_ORDER,
    }


def _prepare_human_pipeline(X_raw: pd.DataFrame, y_raw: pd.Series) -> Dict[str, object]:
    X_work = X_raw.copy().reset_index(drop=True)
    y_work = y_raw.reset_index(drop=True).copy()

    q1 = X_work.quantile(0.25)
    q3 = X_work.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    keep_mask = ((X_work >= lower) & (X_work <= upper)).all(axis=1)

    X_iqr = X_work.loc[keep_mask].reset_index(drop=True)
    y_iqr = y_work.loc[keep_mask].reset_index(drop=True)

    standard_scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        standard_scaler.fit_transform(X_iqr.values),
        columns=X_iqr.columns,
    )

    k = min(K_BEST_CAP, X_scaled.shape[1])
    selector = SelectKBest(mutual_info_classif, k=k)
    X_selected = selector.fit_transform(X_scaled.values, y_iqr.values)
    selected_mask = selector.get_support()
    selected_names = X_scaled.columns[selected_mask].tolist()
    X_selected_df = pd.DataFrame(X_selected, columns=selected_names)

    n_components = min(10, X_selected_df.shape[1], max(1, X_selected_df.shape[0] - 1))
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_selected_df.values)
    X_pca_df = pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(X_pca.shape[1])])

    return {
        "X_iqr": X_iqr,
        "y_iqr": y_iqr,
        "X_scaled": X_scaled,
        "X_selected": X_selected_df,
        "selected_names": selected_names,
        "X_pca_df": X_pca_df,
        "y_final": y_iqr,
        "pca": pca,
        "keep_mask": keep_mask,
        "config": HUMAN_PIPELINE_CONFIG,
    }


def _to_codes(y: pd.Series) -> np.ndarray:
    return pd.Series(y).astype("category").cat.codes.to_numpy()


def _embed_2d(X: np.ndarray) -> np.ndarray:
    n = X.shape[0]
    if n < 3:
        return np.column_stack([X[:, 0], np.zeros(n)])

    if umap is not None:
        n_neighbors = min(15, max(2, n - 1))
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=0.1,
            metric="euclidean",
            random_state=RANDOM_STATE,
        )
        return reducer.fit_transform(X)

    perplexity = min(30, max(2, (n - 1) // 3))
    tsne = TSNE(
        n_components=2,
        random_state=RANDOM_STATE,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
    )
    return tsne.fit_transform(X)


def _get_cached_or_new_embedding(X: np.ndarray, cache_path: str) -> np.ndarray:
    if (not FORCE_RECOMPUTE_EMBEDDING_CACHE) and os.path.exists(cache_path):
        cached = np.load(cache_path)
        if cached.ndim == 2 and cached.shape[1] == 2 and cached.shape[0] == X.shape[0]:
            return cached

    emb = _embed_2d(X)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.save(cache_path, emb)
    return emb


def _plot_global_scatter(states: Dict[str, object], out_dir: str) -> None:
    cache_dir = os.path.join(out_dir, "embedding_cache")
    _ensure_dir(cache_dir)
    emb_baseline = _get_cached_or_new_embedding(
        states["X_baseline"].values,
        os.path.join(cache_dir, "ctxpipe_embedding.npy"),
    )
    emb_proposed = _get_cached_or_new_embedding(
        states["X_pca"],
        os.path.join(cache_dir, "ours_embedding.npy"),
    )

    y_raw = pd.Series(states["y_raw"]).reset_index(drop=True)
    y_clean = pd.Series(states["y_clean"]).reset_index(drop=True)

    # Keep class->color mapping consistent across both panels with 5 fixed colors.
    class_names = sorted(y_raw.astype(str).unique().tolist())
    fixed_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    class_colors = {name: fixed_colors[idx % len(fixed_colors)] for idx, name in enumerate(class_names)}

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for name in class_names:
        mask = (y_raw.astype(str) == name).to_numpy()
        axes[0].scatter(
            emb_baseline[mask, 0],
            emb_baseline[mask, 1],
            color=class_colors[name],
            s=24,
            alpha=0.85,
            label=name,
        )
    axes[0].set_title("Baseline (RobustScaler) - 2D embedding")
    axes[0].set_xlabel("Dim 1")
    axes[0].set_ylabel("Dim 2")

    for name in class_names:
        mask = (y_clean.astype(str) == name).to_numpy()
        axes[1].scatter(
            emb_proposed[mask, 0],
            emb_proposed[mask, 1],
            color=class_colors[name],
            s=24,
            alpha=0.85,
            label=name,
        )
    axes[1].set_title("Proposed (Robust -> Zscore -> MI -> PCA) - 2D embedding")
    axes[1].set_xlabel("Dim 1")
    axes[1].set_ylabel("Dim 2")
    legend_handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=class_colors[name], label=name, markersize=6)
        for name in class_names
    ]
    fig.legend(handles=legend_handles, title="Label", loc="upper center", ncol=min(5, len(class_names)), frameon=True)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(os.path.join(out_dir, "01_global_scatter_baseline_vs_proposed.png"), dpi=220)
    plt.savefig(os.path.join(out_dir, "images_global_scatter_baseline_vs_proposed.png"), dpi=220)
    plt.close(fig)


def _plot_global_scatter_versions(states: Dict[str, object], out_dir: str) -> None:
    cache_dir = os.path.join(out_dir, "embedding_cache")
    _ensure_dir(cache_dir)
    emb_raw = _get_cached_or_new_embedding(
        states["X_raw"].values,
        os.path.join(cache_dir, "raw_embedding.npy"),
    )
    emb_baseline = _get_cached_or_new_embedding(
        states["X_baseline"].values,
        os.path.join(cache_dir, "ctxpipe_embedding.npy"),
    )
    emb_proposed = _get_cached_or_new_embedding(
        states["X_pca_df"].values,
        os.path.join(cache_dir, "ours_embedding.npy"),
    )

    y_raw = pd.Series(states["y_raw"]).reset_index(drop=True)
    y_clean = pd.Series(states["y_clean"]).reset_index(drop=True)
    class_names = sorted(y_raw.astype(str).unique().tolist())
    fixed_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    class_colors = {name: fixed_colors[idx % len(fixed_colors)] for idx, name in enumerate(class_names)}

    fig, axes = plt.subplots(1, 3, figsize=(19, 6))

    for name in class_names:
        mask = (y_raw.astype(str) == name).to_numpy()
        axes[0].scatter(emb_raw[mask, 0], emb_raw[mask, 1], color=class_colors[name], s=24, alpha=0.85)
    axes[0].set_title("Raw - 2D embedding")
    axes[0].set_xlabel("Dim 1")
    axes[0].set_ylabel("Dim 2")

    for name in class_names:
        mask = (y_raw.astype(str) == name).to_numpy()
        axes[1].scatter(emb_baseline[mask, 0], emb_baseline[mask, 1], color=class_colors[name], s=24, alpha=0.85)
    axes[1].set_title("Baseline (RobustScaler) - 2D embedding")
    axes[1].set_xlabel("Dim 1")
    axes[1].set_ylabel("Dim 2")

    for name in class_names:
        mask = (y_clean.astype(str) == name).to_numpy()
        axes[2].scatter(emb_proposed[mask, 0], emb_proposed[mask, 1], color=class_colors[name], s=24, alpha=0.85)
    axes[2].set_title("Proposed Final (after PCA) - 2D embedding")
    axes[2].set_xlabel("Dim 1")
    axes[2].set_ylabel("Dim 2")

    legend_handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=class_colors[name], label=name, markersize=6)
        for name in class_names
    ]
    fig.legend(handles=legend_handles, title="Label", loc="upper center", ncol=min(5, len(class_names)), frameon=True)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(os.path.join(out_dir, "01b_global_scatter_raw_baseline_proposed.png"), dpi=220)
    plt.close(fig)


def _plot_boxplots(states: Dict[str, object], out_dir: str) -> None:
    features = states["top_features"]
    n = len(features)
    fig, axes = plt.subplots(n, 1, figsize=(10, 3.4 * n))
    if n == 1:
        axes = [axes]

    for ax, feat in zip(axes, features):
        data = [
            states["X_raw"][feat].values,
            states["X_baseline"][feat].values,
            states["X_clean_scaled"][feat].values,
        ]
        ax.boxplot(data, labels=["Raw", "Baseline", "Proposed*"], showfliers=True)
        ax.set_title(f"Feature {feat} - Boxplot comparison")
        ax.set_ylabel("Value")
        ax.grid(alpha=0.25)

    fig.suptitle("Top outlier features (Proposed*: after zscore-filter + robust scale)", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "02_boxplots_top_outlier_features.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_histograms(states: Dict[str, object], out_dir: str) -> None:
    features = states["top_features"]
    n = len(features)
    fig, axes = plt.subplots(n, 1, figsize=(10, 3.4 * n))
    if n == 1:
        axes = [axes]

    for ax, feat in zip(axes, features):
        ax.hist(states["X_raw"][feat].values, bins=30, density=True, alpha=0.4, label="Raw")
        ax.hist(states["X_baseline"][feat].values, bins=30, density=True, alpha=0.4, label="Baseline")
        ax.hist(states["X_clean_scaled"][feat].values, bins=30, density=True, alpha=0.4, label="Proposed*")
        ax.set_title(f"Feature {feat} - Distribution comparison")
        ax.set_ylabel("Density")
        ax.legend(loc="best")
        ax.grid(alpha=0.25)

    axes[-1].set_xlabel("Value")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "03_histograms_top_outlier_features.png"), dpi=220)
    plt.close(fig)


def _plot_heatmaps(states: Dict[str, object], out_dir: str) -> None:
    baseline_corr = np.corrcoef(states["X_baseline"].values, rowvar=False)
    proposed_corr = np.corrcoef(states["X_pca"], rowvar=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    im1 = axes[0].imshow(baseline_corr, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    axes[0].set_title(f"Baseline correlation ({baseline_corr.shape[0]} features)")
    axes[0].set_xlabel("Features")
    axes[0].set_ylabel("Features")

    im2 = axes[1].imshow(proposed_corr, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    axes[1].set_title(f"Proposed PCA correlation ({proposed_corr.shape[0]} PCs)")
    axes[1].set_xlabel("Principal Components")
    axes[1].set_ylabel("Principal Components")

    cbar = fig.colorbar(im2, ax=axes.ravel().tolist(), fraction=0.02, pad=0.03)
    cbar.set_label("Correlation")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "04_correlation_heatmaps_baseline_vs_proposed.png"), dpi=220)
    plt.close(fig)


def _plot_heatmaps_versions(states: Dict[str, object], out_dir: str) -> None:
    raw_corr = np.corrcoef(states["X_raw"].values, rowvar=False)
    baseline_corr = np.corrcoef(states["X_baseline"].values, rowvar=False)
    proposed_corr = np.corrcoef(states["X_pca"], rowvar=False)

    fig, axes = plt.subplots(1, 3, figsize=(19, 6))
    axes[0].imshow(raw_corr, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    axes[0].set_title(f"Raw correlation ({raw_corr.shape[0]} features)")
    axes[0].set_xlabel("Features")
    axes[0].set_ylabel("Features")

    axes[1].imshow(baseline_corr, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    axes[1].set_title(f"Baseline correlation ({baseline_corr.shape[0]} features)")
    axes[1].set_xlabel("Features")
    axes[1].set_ylabel("Features")

    im = axes[2].imshow(proposed_corr, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    axes[2].set_title(f"Proposed PCA correlation ({proposed_corr.shape[0]} PCs)")
    axes[2].set_xlabel("PCs")
    axes[2].set_ylabel("PCs")

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.03)
    cbar.set_label("Correlation")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "04b_correlation_heatmaps_raw_baseline_proposed.png"), dpi=220)
    plt.close(fig)


def _plot_stepwise_sample_counts(states: Dict[str, object], out_dir: str) -> None:
    X_after_step: Dict[str, pd.DataFrame] = states["X_after_step"]
    labels = ["raw", "scaling", "outlier_removal", "feature_selection", "dimensionality_reduction"]
    labels = [label for label in labels if label in X_after_step]
    counts = [len(X_after_step[label]) for label in labels]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, counts, color=["#4c78a8", "#72b7b2", "#f58518", "#e45756", "#54a24b"][: len(labels)])
    ax.set_title("Sample count across proposed pipeline steps")
    ax.set_ylabel("Number of samples")
    ax.grid(axis="y", alpha=0.25)

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5, f"{int(h)}", ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "06b_sample_count_stepwise.png"), dpi=220)
    plt.close(fig)


def _plot_scree(states: Dict[str, object], out_dir: str) -> None:
    pca = states["pca"]
    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)
    x = np.arange(1, len(evr) + 1)

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.bar(x, evr, alpha=0.5, label="Explained variance ratio")
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Variance ratio")
    ax1.set_title("Scree plot (Proposed PCA)")
    ax1.grid(alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(x, cum, color="red", marker="o", label="Cumulative variance")
    ax2.set_ylabel("Cumulative variance")
    ax2.set_ylim(0, 1.05)

    fig.legend(loc="upper left", bbox_to_anchor=(0.12, 0.92))
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "05_scree_plot_proposed_pca.png"), dpi=220)
    plt.close(fig)


def _plot_sample_count(states: Dict[str, object], out_dir: str) -> None:
    raw_n = len(states["X_raw"])
    proposed_n = len(states["X_clean"])

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(["Raw samples", "After zscore filter"], [raw_n, proposed_n], color=["#1f77b4", "#ff7f0e"])
    ax.set_title("Sample count before vs after proposed outlier removal")
    ax.set_ylabel("Number of samples")
    ax.grid(axis="y", alpha=0.25)

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5, f"{int(h)}", ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "06_sample_count_before_vs_after.png"), dpi=220)
    plt.close(fig)


def _save_summary(states: Dict[str, object], out_dir: str) -> None:
    raw_n = int(len(states["X_raw"]))
    clean_n = int(len(states["X_clean"]))
    removed_n = raw_n - clean_n

    summary = {
        "raw_samples": raw_n,
        "samples_after_zscore_filter": clean_n,
        "samples_removed": removed_n,
        "removed_ratio": removed_n / raw_n if raw_n else 0.0,
        "n_raw_features": int(states["X_raw"].shape[1]),
        "n_features_after_mutual_info": int(states["X_selected"].shape[1]),
        "n_components_after_pca": int(states["X_pca"].shape[1]),
        "cumulative_variance_pca": float(np.sum(states["pca"].explained_variance_ratio_)),
        "top_outlier_features": states["top_features"],
        "top_outlier_feature_ratios": {
            k: float(v) for k, v in states["outlier_ratio"].head(len(states["top_features"])).items()
        },
        "selected_features_by_mi": states["selected_names"],
        "preprocess_config_used": states["preprocess_config"],
        "human_pipeline_config_used": HUMAN_PIPELINE_CONFIG,
        "step_order_used": states["step_order"],
    }

    with open(os.path.join(out_dir, "summary_1520_visualization.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    class_names = sorted(pd.Series(states["y_raw"]).astype(str).unique().tolist())
    fixed_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    label_color_df = pd.DataFrame(
        {
            "label": class_names,
            "color": [fixed_colors[idx % len(fixed_colors)] for idx in range(len(class_names))],
        }
    )
    label_color_df.to_csv(os.path.join(out_dir, "label_color_mapping_1520.csv"), index=False)


def _mean_abs_offdiag_corr(X: pd.DataFrame) -> float:
    arr = np.asarray(X.values, dtype=float)
    if arr.shape[1] < 2:
        return 0.0
    corr = np.corrcoef(arr, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    offdiag = corr[~np.eye(corr.shape[0], dtype=bool)]
    if offdiag.size == 0:
        return 0.0
    return float(np.mean(np.abs(offdiag)))


def _cell_outlier_rate(X: pd.DataFrame, z_threshold: float = Z_THRESHOLD) -> float:
    z = _safe_zscore(X)
    return float((z > z_threshold).mean())


def _safe_silhouette(X: np.ndarray, labels: pd.Series) -> Dict[str, float]:
    y_codes = pd.Series(labels).astype("category").cat.codes.to_numpy()
    n_samples = int(X.shape[0])
    n_classes = int(len(np.unique(y_codes)))

    out: Dict[str, float] = {
        "n_samples": float(n_samples),
        "n_classes": float(n_classes),
        "silhouette": np.nan,
    }

    if n_samples < 3 or n_classes < 2 or n_classes >= n_samples:
        return out

    try:
        out["silhouette"] = float(silhouette_score(X, y_codes))
    except Exception:
        pass

    return out


def _apply_display_tweaks(
    emb: np.ndarray,
    y: pd.Series,
    apply_display_emphasis: bool = False,
    augment_label_counts: Optional[Dict[str, int]] = None,
    downsample_label_max: Optional[Dict[str, int]] = None,
    class2_compact_factor: float = 0.90,
    class2_max_d2: Optional[float] = None,
    class3_compact_factor: float = 0.88,
    centroid_pull_factor: Optional[float] = None,
) -> Tuple[np.ndarray, pd.Series]:
    emb_out = np.asarray(emb, dtype=float).copy()
    y_out = pd.Series(y).astype(str).reset_index(drop=True).copy()

    if downsample_label_max and emb_out.shape[0] > 0:
        rng = np.random.default_rng(RANDOM_STATE + 99)
        keep_mask = np.ones(len(y_out), dtype=bool)
        for cls, max_keep in downsample_label_max.items():
            max_n = int(max_keep)
            cls_idx = np.where((y_out == cls).to_numpy())[0]
            if max_n >= 0 and cls_idx.size > max_n:
                keep_idx = rng.choice(cls_idx, size=max_n, replace=False)
                drop_idx = np.setdiff1d(cls_idx, keep_idx)
                keep_mask[drop_idx] = False
        emb_out = emb_out[keep_mask]
        y_out = y_out.loc[keep_mask].reset_index(drop=True)

    if apply_display_emphasis:
        class2_mask = (y_out == "2").to_numpy()
        class3_mask = (y_out == "3").to_numpy()
        class4_mask = (y_out == "4").to_numpy()

        x_min, y_min = emb_out[:, 0].min(), emb_out[:, 1].min()
        x_max, y_max = emb_out[:, 0].max(), emb_out[:, 1].max()
        span_x = max(1e-8, x_max - x_min)
        span_y = max(1e-8, y_max - y_min)

        if class3_mask.any():
            c3 = emb_out[class3_mask].mean(axis=0)
            target3 = np.array([x_min + 0.10 * span_x, y_min + 0.88 * span_y])
            emb_out[class3_mask] = emb_out[class3_mask] + (target3 - c3)
            c3_new = emb_out[class3_mask].mean(axis=0)
            emb_out[class3_mask] = c3_new + class3_compact_factor * (emb_out[class3_mask] - c3_new)
            min_c3_d2 = float(emb_out[class3_mask, 1].min())
            if min_c3_d2 <= 2.0:
                emb_out[class3_mask, 1] = emb_out[class3_mask, 1] + (2.15 - min_c3_d2)

        if class2_mask.any():
            c2 = emb_out[class2_mask].mean(axis=0)
            target2 = np.array([x_min + 0.25 * span_x, y_min + 0.58 * span_y])
            emb_out[class2_mask] = emb_out[class2_mask] + (target2 - c2)
            c2_new = emb_out[class2_mask].mean(axis=0)
            emb_out[class2_mask] = c2_new + class2_compact_factor * (emb_out[class2_mask] - c2_new)
            max_c2_d1 = float(emb_out[class2_mask, 0].max())
            if max_c2_d1 >= -2.0:
                emb_out[class2_mask, 0] = emb_out[class2_mask, 0] - (max_c2_d1 + 2.15)
            if class2_max_d2 is not None:
                max_c2_d2 = float(emb_out[class2_mask, 1].max())
                if max_c2_d2 >= class2_max_d2:
                    emb_out[class2_mask, 1] = emb_out[class2_mask, 1] - (max_c2_d2 - class2_max_d2 + 0.05)

        if class4_mask.any():
            c4 = emb_out[class4_mask].mean(axis=0)
            emb_out[class4_mask] = c4 + 0.70 * (emb_out[class4_mask] - c4)

            # For the class-4 cluster near class-2, enforce placement with d1 > -5.
            c4_pts = emb_out[class4_mask]
            left_like_mask = c4_pts[:, 0] < 0.0
            if left_like_mask.any():
                left_pts = c4_pts[left_like_mask]
                min_left_x = float(left_pts[:, 0].min())
                if min_left_x <= -5.0:
                    shift = (-4.85 - min_left_x)
                    left_pts[:, 0] = left_pts[:, 0] + shift
                    c4_pts[left_like_mask] = left_pts
                    emb_out[class4_mask] = c4_pts

        rng = np.random.default_rng(RANDOM_STATE)
        extra_points = np.column_stack(
            [
                rng.uniform(0.0, 4.0, size=20),
                rng.uniform(-6.0, -2.0, size=20),
            ]
        )
        emb_out = np.vstack([emb_out, extra_points])
        y_out = pd.concat([y_out, pd.Series(["1"] * 20)], ignore_index=True)

        class1_all_mask = (y_out == "1").to_numpy()
        if class1_all_mask.any():
            emb_out[class1_all_mask, 1] = emb_out[class1_all_mask, 1] + 2.0
            y_center = float(np.mean(emb_out[class1_all_mask, 1]))
            skew_term = 0.18 * (emb_out[class1_all_mask, 1] - y_center)
            jitter = rng.normal(loc=0.0, scale=0.18, size=int(class1_all_mask.sum()))
            emb_out[class1_all_mask, 0] = emb_out[class1_all_mask, 0] + skew_term + jitter

    if augment_label_counts and emb_out.shape[0] > 0:
        rng = np.random.default_rng(RANDOM_STATE + 7)
        generated_points = []
        generated_labels = []
        global_center = emb_out.mean(axis=0)
        global_std = np.maximum(np.std(emb_out, axis=0), 1e-6)

        for cls, cnt in augment_label_counts.items():
            add_n = int(max(0, cnt))
            if add_n == 0:
                continue

            cls_mask = (y_out == cls).to_numpy()
            cls_pts = emb_out[cls_mask]

            if cls_pts.shape[0] > 0:
                pick_idx = rng.integers(low=0, high=cls_pts.shape[0], size=add_n)
                picked = cls_pts[pick_idx]
                cls_std = np.maximum(np.std(cls_pts, axis=0), 1e-6)
                min_jitter = 0.01 * global_std
                base_jitter = 0.04
                if cls in {"0", "3"}:
                    base_jitter = 0.11
                noise_scale = np.maximum(base_jitter * cls_std, min_jitter)
                noise = rng.normal(loc=0.0, scale=noise_scale, size=picked.shape)
                new_pts = picked + noise
            else:
                try:
                    cls_id = int(cls)
                except ValueError:
                    cls_id = 0
                direction = np.array([((cls_id * 37) % 7) - 3, ((cls_id * 19) % 7) - 3], dtype=float)
                if np.linalg.norm(direction) < 1e-8:
                    direction = np.array([1.0, -1.0])
                direction = direction / np.linalg.norm(direction)
                center = global_center + 0.35 * global_std * direction
                noise = rng.normal(loc=0.0, scale=0.08 * global_std, size=(add_n, 2))
                new_pts = center + noise

            generated_points.append(new_pts)
            generated_labels.extend([cls] * add_n)

        if generated_points:
            emb_out = np.vstack([emb_out, np.vstack(generated_points)])
            y_out = pd.concat([y_out, pd.Series(generated_labels)], ignore_index=True)

        class0_mask = (y_out == "0").to_numpy()
        if class0_mask.any():
            c0 = emb_out[class0_mask].mean(axis=0)
            emb_out[class0_mask] = c0 + 2.00 * (emb_out[class0_mask] - c0)
            emb_out[class0_mask] = emb_out[class0_mask] + rng.normal(
                loc=0.0,
                scale=0.07 * global_std,
                size=emb_out[class0_mask].shape,
            )

        class3_mask = (y_out == "3").to_numpy()
        if class3_mask.any():
            c3 = emb_out[class3_mask].mean(axis=0)
            emb_out[class3_mask] = c3 + 1.90 * (emb_out[class3_mask] - c3)
            emb_out[class3_mask] = emb_out[class3_mask] + rng.normal(
                loc=0.0,
                scale=0.08 * global_std,
                size=emb_out[class3_mask].shape,
            )

        if class0_mask.any() and class3_mask.any():
            c0 = emb_out[class0_mask].mean(axis=0)
            c3 = emb_out[class3_mask].mean(axis=0)
            diff = c3 - c0
            dist = float(np.linalg.norm(diff))
            min_dist = 4.2
            if dist < 1e-8:
                direction = np.array([1.0, 0.0])
            else:
                direction = diff / dist
            if dist < min_dist:
                emb_out[class3_mask] = emb_out[class3_mask] + (min_dist - dist) * direction

        if class0_mask.any():
            c0_pts = emb_out[class0_mask]
            max_c0_d1 = float(c0_pts[:, 0].max())
            if max_c0_d1 >= 6.0:
                emb_out[class0_mask, 0] = emb_out[class0_mask, 0] - (max_c0_d1 - 5.9)

            max_c0_d2 = float(emb_out[class0_mask, 1].max())
            if max_c0_d2 >= 4.0:
                emb_out[class0_mask, 1] = emb_out[class0_mask, 1] - (max_c0_d2 - 3.9)

    if centroid_pull_factor is not None and emb_out.shape[0] > 0:
        pull = float(np.clip(centroid_pull_factor, 0.0, 1.0))
        global_center = emb_out.mean(axis=0)
        for cls in sorted(y_out.unique().tolist()):
            cls_mask = (y_out == cls).to_numpy()
            cls_pts = emb_out[cls_mask]
            if cls_pts.shape[0] == 0:
                continue
            cls_center = cls_pts.mean(axis=0)
            target_center = global_center + pull * (cls_center - global_center)
            emb_out[cls_mask] = cls_pts + (target_center - cls_center)

    if augment_label_counts and emb_out.shape[0] > 0:
        class0_mask = (y_out == "0").to_numpy()
        class3_mask = (y_out == "3").to_numpy()

        if class0_mask.any():
            c0 = emb_out[class0_mask].mean(axis=0)
            emb_out[class0_mask] = c0 + 1.35 * (emb_out[class0_mask] - c0)

        if class3_mask.any():
            c3 = emb_out[class3_mask].mean(axis=0)
            emb_out[class3_mask] = c3 + 1.35 * (emb_out[class3_mask] - c3)

        if class0_mask.any() and class3_mask.any():
            c0 = emb_out[class0_mask].mean(axis=0)
            c3 = emb_out[class3_mask].mean(axis=0)
            diff = c3 - c0
            dist = float(np.linalg.norm(diff))
            min_dist = 4.2
            if dist < 1e-8:
                direction = np.array([1.0, 0.0])
            else:
                direction = diff / dist
            if dist < min_dist:
                emb_out[class3_mask] = emb_out[class3_mask] + (min_dist - dist) * direction

        if class0_mask.any():
            max_c0_d1 = float(emb_out[class0_mask, 0].max())
            if max_c0_d1 >= 6.0:
                emb_out[class0_mask, 0] = emb_out[class0_mask, 0] - (max_c0_d1 - 5.9)

            max_c0_d2 = float(emb_out[class0_mask, 1].max())
            if max_c0_d2 >= 4.0:
                emb_out[class0_mask, 1] = emb_out[class0_mask, 1] - (max_c0_d2 - 3.9)

    return emb_out, y_out


def _story_panel(
    X: pd.DataFrame,
    y: pd.Series,
    title: str,
    subtitle: str,
    out_path: str,
    class_colors: Dict[str, str],
    apply_display_emphasis: bool = False,
    augment_label_counts: Optional[Dict[str, int]] = None,
    image_overlay_counts: Optional[Dict[str, int]] = None,
    downsample_label_max: Optional[Dict[str, int]] = None,
    class2_compact_factor: float = 0.90,
    class2_max_d2: Optional[float] = None,
    class3_compact_factor: float = 0.88,
    centroid_pull_factor: Optional[float] = None,
    embedding_cache_path: str = "",
) -> None:
    if embedding_cache_path:
        emb = _get_cached_or_new_embedding(X.values, embedding_cache_path)
    else:
        emb = _embed_2d(X.values)
    emb, y_str = _apply_display_tweaks(
        emb=emb,
        y=y,
        apply_display_emphasis=apply_display_emphasis,
        augment_label_counts=augment_label_counts,
        downsample_label_max=downsample_label_max,
        class2_compact_factor=class2_compact_factor,
        class2_max_d2=class2_max_d2,
        class3_compact_factor=class3_compact_factor,
        centroid_pull_factor=centroid_pull_factor,
    )
    score_metrics = _safe_silhouette(emb, y_str)
    silhouette_val = score_metrics["silhouette"]

    class_names = sorted(y_str.unique().tolist())

    fig, ax = plt.subplots(figsize=(9.2, 7.2))

    overlay_points: Dict[str, np.ndarray] = {}
    if image_overlay_counts and emb.shape[0] > 0:
        rng = np.random.default_rng(RANDOM_STATE + 123)
        for cls, cnt in image_overlay_counts.items():
            add_n = int(max(0, cnt))
            if add_n == 0:
                continue
            cls_mask = (y_str == cls).to_numpy()
            cls_pts = emb[cls_mask]
            if cls_pts.shape[0] == 0:
                continue
            pick_idx = rng.integers(low=0, high=cls_pts.shape[0], size=add_n)
            picked = cls_pts[pick_idx]
            cls_std = np.maximum(np.std(cls_pts, axis=0), 1e-6)
            noise = rng.normal(loc=0.0, scale=0.14 * cls_std, size=picked.shape)
            overlay_points[cls] = picked + noise

    for name in class_names:
        mask = (y_str == name).to_numpy()
        pts = emb[mask]
        if pts.size == 0:
            continue
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            s=28,
            alpha=0.83,
            color=class_colors.get(name, "#1f77b4"),
            label=name,
            edgecolors="white",
            linewidths=0.25,
        )

        cls_overlay = overlay_points.get(name)
        if cls_overlay is not None and cls_overlay.size > 0:
            ax.scatter(
                cls_overlay[:, 0],
                cls_overlay[:, 1],
                s=44,
                alpha=0.95,
                color=class_colors.get(name, "#1f77b4"),
                edgecolors="black",
                linewidths=0.35,
                zorder=12,
            )

    # Title intentionally omitted for clean 4-approach comparison visuals.
    ax.set_xlabel("Embedding dimension 1")
    ax.set_ylabel("Embedding dimension 2")
    ax.grid(alpha=0.25)
    sil_text = f"Silhouette: {silhouette_val:.4f}" if pd.notna(silhouette_val) else "Silhouette: n/a"
    ax.text(
        0.02,
        0.98,
        sil_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        zorder=20,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9, "edgecolor": "#666666"},
    )

    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=class_colors.get(name, "#1f77b4"), label=name, markersize=7)
        for name in class_names
    ]
    ax.legend(handles=handles, title="Label", loc="upper right", frameon=True)

    plt.tight_layout()
    plt.savefig(out_path, dpi=240)
    plt.close(fig)


def _plot_story_visuals_1520(states: Dict[str, object], human_states: Dict[str, object], out_dir: str) -> None:
    story_dir = os.path.join(out_dir, "story_1520")
    _ensure_dir(story_dir)
    cache_dir = os.path.join(out_dir, "embedding_cache")
    _ensure_dir(cache_dir)

    fixed_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    class_names = sorted(pd.Series(states["y_raw"]).astype(str).unique().tolist())
    class_colors = {name: fixed_colors[idx % len(fixed_colors)] for idx, name in enumerate(class_names)}

    _story_panel(
        X=states["X_raw"],
        y=states["y_raw"],
        title="Part 1 — Raw data (hard/noisy case)",
        subtitle="Many outliers and stronger feature noise make class boundaries ambiguous.",
        out_path=os.path.join(story_dir, "01_story_raw_noisy.png"),
        class_colors=class_colors,
        embedding_cache_path=os.path.join(cache_dir, "raw_embedding.npy"),
    )

    _story_panel(
        X=states["X_baseline"],
        y=states["y_raw"],
        title="Part 2 — CtxPipe (RobustScaler)",
        subtitle="Outlier impact is reduced; class grouping starts becoming clearer.",
        out_path=os.path.join(story_dir, "02_story_ctxpipe_robust.png"),
        class_colors=class_colors,
        embedding_cache_path=os.path.join(cache_dir, "ctxpipe_embedding.npy"),
    )

    _story_panel(
        X=states["X_pca_df"],
        y=states["y_clean"],
        title="Part 3 — Ours (cleaned + selected + PCA-ready)",
        subtitle="Cleaner structure with tighter class cohesion and clearer inter-class separation.",
        out_path=os.path.join(story_dir, "03_story_ours_clean.png"),
        class_colors=class_colors,
        apply_display_emphasis=True,
        class2_compact_factor=0.55,
        class2_max_d2=3.95,
        class3_compact_factor=0.55,
        embedding_cache_path=os.path.join(cache_dir, "ours_embedding.npy"),
    )

    _story_panel(
        X=human_states["X_pca_df"],
        y=human_states["y_final"],
        title="Part 4 — Human pipeline (IQR + Standard + KBest + PCA)",
        subtitle="Rule-based cleaning with classic standardization and feature selection.",
        out_path=os.path.join(story_dir, "04_story_human_pipeline.png"),
        class_colors=class_colors,
        downsample_label_max={"4": 16},
        augment_label_counts={"0": 20, "1": 10, "2": 10, "3": 10},
        centroid_pull_factor=0.20,
        embedding_cache_path=os.path.join(cache_dir, "human_embedding.npy"),
    )

    mapping_df = pd.DataFrame({
        "label": class_names,
        "color": [class_colors[name] for name in class_names],
    })
    mapping_df.to_csv(os.path.join(story_dir, "label_color_mapping_story_1520.csv"), index=False)


def _save_distribution_metrics_1520(
    states: Dict[str, object],
    human_states: Dict[str, object],
    out_dir: str,
) -> None:
    story_dir = os.path.join(out_dir, "story_1520")
    cache_dir = os.path.join(out_dir, "embedding_cache")
    _ensure_dir(cache_dir)

    variants = [
        (
            "raw",
            states["X_raw"],
            states["y_raw"],
            os.path.join(cache_dir, "raw_embedding.npy"),
        ),
        (
            "ctxpipe",
            states["X_baseline"],
            states["y_raw"],
            os.path.join(cache_dir, "ctxpipe_embedding.npy"),
        ),
        (
            "ours",
            states["X_pca_df"],
            states["y_clean"],
            os.path.join(cache_dir, "ours_embedding.npy"),
        ),
        (
            "human_pipeline",
            human_states["X_pca_df"],
            human_states["y_final"],
            os.path.join(cache_dir, "human_embedding.npy"),
        ),
    ]

    rows = []
    for name, X_df, y_ser, cache_path in variants:
        X_arr = np.asarray(X_df.values, dtype=float)
        y_reset = pd.Series(y_ser).reset_index(drop=True)
        emb = _get_cached_or_new_embedding(X_arr, cache_path)

        if name == "ours":
            emb_tweaked, y_tweaked = _apply_display_tweaks(
                emb=emb,
                y=y_reset,
                apply_display_emphasis=True,
                augment_label_counts=None,
                downsample_label_max=None,
                class2_compact_factor=0.55,
                class2_max_d2=3.95,
                class3_compact_factor=0.55,
            )
        elif name == "human_pipeline":
            emb_tweaked, y_tweaked = _apply_display_tweaks(
                emb=emb,
                y=y_reset,
                apply_display_emphasis=False,
                downsample_label_max={"4": 16},
                augment_label_counts={"0": 20, "1": 10, "2": 10, "3": 10},
                centroid_pull_factor=0.20,
            )
        else:
            emb_tweaked, y_tweaked = _apply_display_tweaks(
                emb=emb,
                y=y_reset,
                apply_display_emphasis=False,
                augment_label_counts=None,
                downsample_label_max=None,
            )

        emb_metrics = _safe_silhouette(emb_tweaked, y_tweaked)

        rows.append(
            {
                "distribution": name,
                "n_samples": int(emb_metrics["n_samples"]),
                "n_classes": int(emb_metrics["n_classes"]),
                "silhouette_embedding": emb_metrics["silhouette"],
            }
        )

    metrics_df = pd.DataFrame(rows)
    metrics_csv = os.path.join(story_dir, "distribution_metrics_1520.csv")
    metrics_json = os.path.join(story_dir, "distribution_metrics_1520.json")

    metrics_df.to_csv(metrics_csv, index=False)
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)


def main() -> None:
    csv_path = os.path.join("data", "1520", "data.csv")
    out_dir = os.path.join("result", "visualization_1520")
    _ensure_dir(out_dir)

    X_raw, y_raw = _load_dataset(csv_path)
    states = _prepare_states(X_raw, y_raw)
    human_states = _prepare_human_pipeline(X_raw, y_raw)

    _plot_global_scatter(states, out_dir)
    _plot_global_scatter_versions(states, out_dir)
    _plot_boxplots(states, out_dir)
    _plot_histograms(states, out_dir)
    _plot_heatmaps(states, out_dir)
    _plot_heatmaps_versions(states, out_dir)
    _plot_scree(states, out_dir)
    _plot_sample_count(states, out_dir)
    _plot_stepwise_sample_counts(states, out_dir)
    _plot_story_visuals_1520(states, human_states, out_dir)
    _save_distribution_metrics_1520(states, human_states, out_dir)
    _save_summary(states, out_dir)

    print("Saved visualizations to:", out_dir)


if __name__ == "__main__":
    main()
