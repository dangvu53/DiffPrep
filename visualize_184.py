import json
import os
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.manifold import TSNE
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import MaxAbsScaler, RobustScaler, StandardScaler

try:
    import umap.umap_ as umap
except ImportError:
    umap = None


RANDOM_STATE = 42
TOP_FEATURES = 5
Z_THRESHOLD = 3.0
SVD_COMPONENT_CAP = 10
K_BEST_CAP = 20
FORCE_RECOMPUTE_EMBEDDING_CACHE = False

CTXPIPE_CONFIG = {
    "scaling": "robust",
}

OURS_CONFIG = {
    "imputation": "constant",
    "scaling": "maxabs",
    "encoding": "onehot",
    "outlier_removal": "zscore",
    "feature_selection": "none",
    "dimensionality_reduction": "svd",
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


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _safe_zscore(df: pd.DataFrame) -> np.ndarray:
    numeric = pd.DataFrame(df).apply(pd.to_numeric, errors="coerce").fillna(0.0)
    z = np.abs(zscore(numeric.to_numpy(dtype=float), axis=0, nan_policy="omit"))
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
    return X, y


def _split_columns(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    return numeric_cols, categorical_cols


def _constant_impute(X: pd.DataFrame, numeric_cols: List[str], categorical_cols: List[str]) -> pd.DataFrame:
    X_imp = X.copy()
    if numeric_cols:
        X_imp[numeric_cols] = X_imp[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    if categorical_cols:
        X_imp[categorical_cols] = X_imp[categorical_cols].astype("object").fillna("missing")
    return X_imp


def _ctxpipe_baseline(X: pd.DataFrame) -> pd.DataFrame:
    numeric = X.select_dtypes(include=[np.number]).copy()
    if numeric.shape[1] == 0:
        numeric = pd.get_dummies(X.astype("object").fillna("missing"), drop_first=False)
    numeric = numeric.apply(pd.to_numeric, errors="coerce")
    numeric = numeric.fillna(numeric.median(numeric_only=True)).fillna(0.0)

    scaler = RobustScaler()
    arr = scaler.fit_transform(numeric.values)
    return pd.DataFrame(arr, columns=numeric.columns)


def _embed_2d(X: np.ndarray) -> np.ndarray:
    n = X.shape[0]
    if n < 3:
        return np.column_stack([X[:, 0], np.zeros(n)])

    if n > 2000:
        fast_proj = TruncatedSVD(n_components=2, random_state=RANDOM_STATE)
        return fast_proj.fit_transform(X)

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


def _to_codes(y: pd.Series) -> np.ndarray:
    return pd.Series(y).astype("category").cat.codes.to_numpy()


def _prepare_states(X_raw: pd.DataFrame, y_raw: pd.Series) -> Dict[str, object]:
    numeric_cols, categorical_cols = _split_columns(X_raw)

    # Raw (numeric view for comparability plots)
    X_raw_num = X_raw.select_dtypes(include=[np.number]).copy()
    if X_raw_num.shape[1] == 0:
        X_raw_num = pd.DataFrame(index=X_raw.index)
    X_raw_num = X_raw_num.apply(pd.to_numeric, errors="coerce")
    X_raw_num = X_raw_num.fillna(X_raw_num.median(numeric_only=True)).fillna(0.0)

    # Baseline: CtxPipe = robust scaling
    X_baseline = _ctxpipe_baseline(X_raw)

    # Ours: impute -> scale -> encode -> outlier -> feature_selection(none) -> svd
    X_step = {}
    y_step = {}

    X_imp = _constant_impute(X_raw, numeric_cols, categorical_cols)
    X_step["imputation"] = X_imp.copy()
    y_step["imputation"] = y_raw.reset_index(drop=True).copy()

    X_scaled = X_imp.copy()
    if numeric_cols:
        maxabs = MaxAbsScaler()
        X_scaled[numeric_cols] = maxabs.fit_transform(X_scaled[numeric_cols].values)
    X_step["scaling"] = X_scaled.copy()
    y_step["scaling"] = y_raw.reset_index(drop=True).copy()

    if categorical_cols:
        X_encoded = pd.get_dummies(X_scaled, columns=categorical_cols, drop_first=False)
    else:
        X_encoded = X_scaled.copy()
    X_encoded = X_encoded.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    X_step["encoding"] = X_encoded.copy()
    y_step["encoding"] = y_raw.reset_index(drop=True).copy()

    z_encoded = _safe_zscore(X_encoded)
    keep_mask = (z_encoded < Z_THRESHOLD).all(axis=1)
    X_clean = X_encoded.loc[keep_mask].reset_index(drop=True)
    y_clean = y_raw.reset_index(drop=True).loc[keep_mask].reset_index(drop=True)
    X_step["outlier_removal"] = X_clean.copy()
    y_step["outlier_removal"] = y_clean.copy()

    # feature_selection = none
    X_fs = X_clean.copy()
    X_step["feature_selection"] = X_fs.copy()
    y_step["feature_selection"] = y_clean.copy()

    n_components = min(SVD_COMPONENT_CAP, X_fs.shape[1])
    n_components = max(1, n_components)
    svd = TruncatedSVD(n_components=n_components, random_state=RANDOM_STATE)
    X_svd = svd.fit_transform(X_fs.values)
    X_svd_df = pd.DataFrame(X_svd, columns=[f"SVD{i+1}" for i in range(X_svd.shape[1])])
    X_step["dimensionality_reduction"] = X_svd_df.copy()
    y_step["dimensionality_reduction"] = y_clean.copy()

    # Top outlier features from common numeric columns only (for side-by-side comparability)
    common_numeric = [c for c in numeric_cols if c in X_clean.columns and c in X_baseline.columns and c in X_raw_num.columns]
    if common_numeric:
        z_common = _safe_zscore(X_encoded[common_numeric])
        outlier_ratio = (z_common > Z_THRESHOLD).mean(axis=0)
        outlier_ratio_series = pd.Series(outlier_ratio, index=common_numeric).sort_values(ascending=False)
    else:
        outlier_ratio_series = pd.Series(dtype=float)

    top_features = outlier_ratio_series.head(min(TOP_FEATURES, len(outlier_ratio_series))).index.tolist()

    return {
        "X_raw_num": X_raw_num,
        "y_raw": y_raw.reset_index(drop=True),
        "X_baseline": X_baseline,
        "X_imputed": X_imp,
        "X_scaled": X_scaled,
        "X_encoded": X_encoded,
        "X_clean": X_clean,
        "y_clean": y_clean,
        "X_svd_df": X_svd_df,
        "X_svd": X_svd_df.values,
        "svd": svd,
        "top_features": top_features,
        "outlier_ratio": outlier_ratio_series,
        "keep_mask": keep_mask,
        "X_step": X_step,
        "y_step": y_step,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
    }


def _prepare_human_pipeline(X_raw_num: pd.DataFrame, y_raw: pd.Series) -> Dict[str, object]:
    X_work = X_raw_num.copy().reset_index(drop=True)
    y_work = y_raw.reset_index(drop=True).copy()

    q1 = X_work.quantile(0.25)
    q3 = X_work.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    keep_mask = ((X_work >= lower) & (X_work <= upper)).all(axis=1)

    X_iqr = X_work.loc[keep_mask].reset_index(drop=True)
    y_iqr = y_work.loc[keep_mask].reset_index(drop=True)

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_iqr.values), columns=X_iqr.columns)

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
        "keep_mask": keep_mask,
        "pca": pca,
        "config": HUMAN_PIPELINE_CONFIG,
    }


def _plot_global_scatter_versions(states: Dict[str, object], out_dir: str) -> None:
    emb_raw = _embed_2d(states["X_raw_num"].values)
    emb_baseline = _embed_2d(states["X_baseline"].values)
    emb_proposed = _embed_2d(states["X_svd_df"].values)

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
    axes[1].set_title("CtxPipe (RobustScaler) - 2D embedding")
    axes[1].set_xlabel("Dim 1")
    axes[1].set_ylabel("Dim 2")

    for name in class_names:
        mask = (y_clean.astype(str) == name).to_numpy()
        axes[2].scatter(emb_proposed[mask, 0], emb_proposed[mask, 1], color=class_colors[name], s=24, alpha=0.85)
    axes[2].set_title("Ours Final (SVD) - 2D embedding")
    axes[2].set_xlabel("Dim 1")
    axes[2].set_ylabel("Dim 2")

    legend_handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=class_colors[name], label=name, markersize=6)
        for name in class_names
    ]
    fig.legend(handles=legend_handles, title="Label", loc="upper center", ncol=min(5, len(class_names)), frameon=True)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(os.path.join(out_dir, "01_global_scatter_raw_ctxpipe_ours.png"), dpi=220)
    plt.close(fig)


def _separation_score(embedding: np.ndarray, labels: pd.Series) -> float:
    y = pd.Series(labels).astype("category").cat.codes.to_numpy()
    classes = np.unique(y)
    if embedding.shape[0] < 3 or len(classes) < 2:
        return 0.0

    centroids = []
    spreads = []
    for c in classes:
        pts = embedding[y == c]
        if pts.shape[0] == 0:
            continue
        centroid = pts.mean(axis=0)
        centroids.append(centroid)
        spreads.append(np.mean(np.linalg.norm(pts - centroid, axis=1)) + 1e-8)

    if len(centroids) < 2:
        return 0.0
    centroids = np.vstack(centroids)
    dists = np.linalg.norm(centroids[:, None, :] - centroids[None, :, :], axis=2)
    inter = np.mean(dists[~np.eye(dists.shape[0], dtype=bool)])
    intra = float(np.mean(spreads))
    return float(inter / max(intra, 1e-8))


def _safe_cluster_metrics(X: np.ndarray, labels: pd.Series) -> Dict[str, float]:
    y_codes = pd.Series(labels).astype("category").cat.codes.to_numpy()
    n_samples = int(X.shape[0])
    n_classes = int(len(np.unique(y_codes)))

    out: Dict[str, float] = {
        "n_samples": float(n_samples),
        "n_classes": float(n_classes),
        "silhouette": np.nan,
        "calinski_harabasz": np.nan,
        "davies_bouldin": np.nan,
        "separation_score": np.nan,
    }

    if n_samples < 3 or n_classes < 2 or n_classes >= n_samples:
        return out

    try:
        out["silhouette"] = float(silhouette_score(X, y_codes))
    except Exception:
        pass
    try:
        out["calinski_harabasz"] = float(calinski_harabasz_score(X, y_codes))
    except Exception:
        pass
    try:
        out["davies_bouldin"] = float(davies_bouldin_score(X, y_codes))
    except Exception:
        pass
    try:
        out["separation_score"] = float(_separation_score(X, labels))
    except Exception:
        pass
    return out


def _story_panel(
    X: pd.DataFrame,
    y: pd.Series,
    title: str,
    subtitle: str,
    out_path: str,
    class_colors: Dict[str, str],
    embedding_cache_path: str,
) -> None:
    emb = _get_cached_or_new_embedding(X.values, embedding_cache_path)
    y_str = pd.Series(y).astype(str).reset_index(drop=True)
    class_names = sorted(y_str.unique().tolist())

    fig, ax = plt.subplots(figsize=(9.2, 7.2))
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

    ax.set_title(f"{title}\n{subtitle}", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Embedding dimension 1")
    ax.set_ylabel("Embedding dimension 2")
    ax.grid(alpha=0.25)

    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=class_colors.get(name, "#1f77b4"), label=name, markersize=7)
        for name in class_names
    ]
    ax.legend(handles=handles, title="Label", loc="upper right", frameon=True)

    plt.tight_layout()
    plt.savefig(out_path, dpi=240)
    plt.close(fig)


def _plot_story_visuals_184(states: Dict[str, object], human_states: Dict[str, object], out_dir: str) -> None:
    story_dir = os.path.join(out_dir, "story_184")
    _ensure_dir(story_dir)
    cache_dir = os.path.join(story_dir, "embedding_cache")
    _ensure_dir(cache_dir)

    fixed_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    class_names = sorted(pd.Series(states["y_raw"]).astype(str).unique().tolist())
    class_colors = {name: fixed_colors[idx % len(fixed_colors)] for idx, name in enumerate(class_names)}

    _story_panel(
        X=states["X_raw_num"],
        y=states["y_raw"],
        title="Part 1 — Raw data",
        subtitle="Original distribution before preprocessing.",
        out_path=os.path.join(story_dir, "01_story_raw.png"),
        class_colors=class_colors,
        embedding_cache_path=os.path.join(cache_dir, "raw_embedding.npy"),
    )

    _story_panel(
        X=states["X_baseline"],
        y=states["y_raw"],
        title="Part 2 — CtxPipe (RobustScaler)",
        subtitle="Baseline robust scaling view.",
        out_path=os.path.join(story_dir, "02_story_ctxpipe.png"),
        class_colors=class_colors,
        embedding_cache_path=os.path.join(cache_dir, "ctxpipe_embedding.npy"),
    )

    _story_panel(
        X=states["X_svd_df"],
        y=states["y_clean"],
        title="Part 3 — Ours",
        subtitle="Constant impute + MaxAbs + OneHot + Z-score + SVD.",
        out_path=os.path.join(story_dir, "03_story_ours.png"),
        class_colors=class_colors,
        embedding_cache_path=os.path.join(cache_dir, "ours_embedding.npy"),
    )

    _story_panel(
        X=human_states["X_pca_df"],
        y=human_states["y_final"],
        title="Part 4 — Human pipeline",
        subtitle="None impute/encode + IQR + Standard + KBest + PCA.",
        out_path=os.path.join(story_dir, "04_story_human_pipeline.png"),
        class_colors=class_colors,
        embedding_cache_path=os.path.join(cache_dir, "human_embedding.npy"),
    )


def _save_distribution_metrics_184(states: Dict[str, object], human_states: Dict[str, object], out_dir: str) -> None:
    story_dir = os.path.join(out_dir, "story_184")
    cache_dir = os.path.join(story_dir, "embedding_cache")
    _ensure_dir(cache_dir)

    variants = [
        ("raw", states["X_raw_num"], states["y_raw"], os.path.join(cache_dir, "raw_embedding.npy")),
        ("ctxpipe", states["X_baseline"], states["y_raw"], os.path.join(cache_dir, "ctxpipe_embedding.npy")),
        ("ours", states["X_svd_df"], states["y_clean"], os.path.join(cache_dir, "ours_embedding.npy")),
        ("human_pipeline", human_states["X_pca_df"], human_states["y_final"], os.path.join(cache_dir, "human_embedding.npy")),
    ]

    rows = []
    for name, X_df, y_ser, cache_path in variants:
        X_arr = np.asarray(X_df.values, dtype=float)
        y_reset = pd.Series(y_ser).reset_index(drop=True)
        emb = _get_cached_or_new_embedding(X_arr, cache_path)

        emb_metrics = _safe_cluster_metrics(emb, y_reset)
        feat_metrics = _safe_cluster_metrics(X_arr, y_reset)

        rows.append(
            {
                "distribution": name,
                "n_samples": int(emb_metrics["n_samples"]),
                "n_classes": int(emb_metrics["n_classes"]),
                "silhouette_embedding": emb_metrics["silhouette"],
                "calinski_harabasz_embedding": emb_metrics["calinski_harabasz"],
                "davies_bouldin_embedding": emb_metrics["davies_bouldin"],
                "separation_embedding": emb_metrics["separation_score"],
                "silhouette_feature_space": feat_metrics["silhouette"],
                "calinski_harabasz_feature_space": feat_metrics["calinski_harabasz"],
                "davies_bouldin_feature_space": feat_metrics["davies_bouldin"],
                "separation_feature_space": feat_metrics["separation_score"],
            }
        )

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(os.path.join(story_dir, "distribution_metrics_184.csv"), index=False)
    with open(os.path.join(story_dir, "distribution_metrics_184.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)


def _plot_boxplots(states: Dict[str, object], out_dir: str) -> None:
    features = states["top_features"]
    if not features:
        return

    n = len(features)
    fig, axes = plt.subplots(n, 1, figsize=(10, 3.4 * n))
    if n == 1:
        axes = [axes]

    for ax, feat in zip(axes, features):
        data = [
            states["X_raw_num"][feat].values,
            states["X_baseline"][feat].values,
            states["X_clean"][feat].values,
        ]
        ax.boxplot(data, labels=["Raw", "CtxPipe", "Ours*"], showfliers=True)
        ax.set_title(f"Feature {feat} - Boxplot comparison")
        ax.set_ylabel("Value")
        ax.grid(alpha=0.25)

    fig.suptitle("Top outlier features (Ours*: after onehot + zscore filter)", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "02_boxplots_top_outlier_features.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_histograms(states: Dict[str, object], out_dir: str) -> None:
    features = states["top_features"]
    if not features:
        return

    n = len(features)
    fig, axes = plt.subplots(n, 1, figsize=(10, 3.4 * n))
    if n == 1:
        axes = [axes]

    for ax, feat in zip(axes, features):
        ax.hist(states["X_raw_num"][feat].values, bins=30, density=True, alpha=0.4, label="Raw")
        ax.hist(states["X_baseline"][feat].values, bins=30, density=True, alpha=0.4, label="CtxPipe")
        ax.hist(states["X_clean"][feat].values, bins=30, density=True, alpha=0.4, label="Ours*")
        ax.set_title(f"Feature {feat} - Distribution comparison")
        ax.set_ylabel("Density")
        ax.legend(loc="best")
        ax.grid(alpha=0.25)

    axes[-1].set_xlabel("Value")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "03_histograms_top_outlier_features.png"), dpi=220)
    plt.close(fig)


def _plot_heatmaps_versions(states: Dict[str, object], out_dir: str) -> None:
    raw_corr = np.corrcoef(states["X_raw_num"].values, rowvar=False)
    baseline_corr = np.corrcoef(states["X_baseline"].values, rowvar=False)
    proposed_corr = np.corrcoef(states["X_svd"], rowvar=False)

    fig, axes = plt.subplots(1, 3, figsize=(19, 6))
    axes[0].imshow(raw_corr, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    axes[0].set_title(f"Raw correlation ({raw_corr.shape[0]} features)")
    axes[0].set_xlabel("Features")
    axes[0].set_ylabel("Features")

    axes[1].imshow(baseline_corr, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    axes[1].set_title(f"CtxPipe correlation ({baseline_corr.shape[0]} features)")
    axes[1].set_xlabel("Features")
    axes[1].set_ylabel("Features")

    im = axes[2].imshow(proposed_corr, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    axes[2].set_title(f"Ours SVD correlation ({proposed_corr.shape[0]} comps)")
    axes[2].set_xlabel("SVD components")
    axes[2].set_ylabel("SVD components")

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.03)
    cbar.set_label("Correlation")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "04_correlation_heatmaps_raw_ctxpipe_ours.png"), dpi=220)
    plt.close(fig)


def _plot_scree(states: Dict[str, object], out_dir: str) -> None:
    svd = states["svd"]
    evr = svd.explained_variance_ratio_
    cum = np.cumsum(evr)
    x = np.arange(1, len(evr) + 1)

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.bar(x, evr, alpha=0.5, label="Explained variance ratio")
    ax1.set_xlabel("SVD Component")
    ax1.set_ylabel("Variance ratio")
    ax1.set_title("Scree plot (Ours SVD)")
    ax1.grid(alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(x, cum, color="red", marker="o", label="Cumulative variance")
    ax2.set_ylabel("Cumulative variance")
    ax2.set_ylim(0, 1.05)

    fig.legend(loc="upper left", bbox_to_anchor=(0.12, 0.92))
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "05_scree_plot_ours_svd.png"), dpi=220)
    plt.close(fig)


def _plot_sample_counts(states: Dict[str, object], out_dir: str) -> None:
    raw_n = len(states["X_raw_num"])
    ours_after_outlier_n = len(states["X_clean"])

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(["Raw samples", "After Ours zscore"], [raw_n, ours_after_outlier_n], color=["#1f77b4", "#ff7f0e"])
    ax.set_title("Sample count before vs after Ours outlier removal")
    ax.set_ylabel("Number of samples")
    ax.grid(axis="y", alpha=0.25)

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5, f"{int(h)}", ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "06_sample_count_before_vs_after.png"), dpi=220)
    plt.close(fig)


def _plot_stepwise_sample_counts(states: Dict[str, object], out_dir: str) -> None:
    labels = ["imputation", "scaling", "encoding", "outlier_removal", "feature_selection", "dimensionality_reduction"]
    counts = [len(states["X_step"][label]) for label in labels]

    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(labels, counts, color=["#4c78a8", "#72b7b2", "#f58518", "#e45756", "#54a24b", "#b279a2"])
    ax.set_title("Sample count across Ours pipeline steps")
    ax.set_ylabel("Number of samples")
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=15)

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5, f"{int(h)}", ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "06b_sample_count_stepwise_ours.png"), dpi=220)
    plt.close(fig)


def _save_summary(states: Dict[str, object], out_dir: str) -> None:
    raw_n = int(len(states["X_raw_num"]))
    clean_n = int(len(states["X_clean"]))
    removed_n = raw_n - clean_n

    summary = {
        "dataset": "184",
        "ctxpipe_config_used": CTXPIPE_CONFIG,
        "ours_config_used": OURS_CONFIG,
        "human_pipeline_config_used": HUMAN_PIPELINE_CONFIG,
        "step_order_used": STEP_ORDER,
        "raw_samples": raw_n,
        "samples_after_ours_outlier": clean_n,
        "samples_removed": removed_n,
        "removed_ratio": removed_n / raw_n if raw_n else 0.0,
        "n_raw_numeric_features": int(states["X_raw_num"].shape[1]),
        "n_features_after_encoding": int(states["X_encoded"].shape[1]),
        "n_features_after_feature_selection": int(states["X_step"]["feature_selection"].shape[1]),
        "n_components_after_svd": int(states["X_svd"].shape[1]),
        "cumulative_variance_svd": float(np.sum(states["svd"].explained_variance_ratio_)),
        "top_outlier_features": states["top_features"],
        "top_outlier_feature_ratios": {k: float(v) for k, v in states["outlier_ratio"].head(len(states["top_features"])).items()},
        "n_categorical_columns": int(len(states["categorical_cols"])),
        "n_numeric_columns": int(len(states["numeric_cols"])),
    }

    with open(os.path.join(out_dir, "summary_184_visualization.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def main() -> None:
    csv_path = os.path.join("data", "184", "data.csv")
    out_dir = os.path.join("result", "visualization_184")
    _ensure_dir(out_dir)

    X_raw, y_raw = _load_dataset(csv_path)
    states = _prepare_states(X_raw, y_raw)
    human_states = _prepare_human_pipeline(states["X_raw_num"], y_raw)

    _plot_global_scatter_versions(states, out_dir)
    _plot_boxplots(states, out_dir)
    _plot_histograms(states, out_dir)
    _plot_heatmaps_versions(states, out_dir)
    _plot_scree(states, out_dir)
    _plot_sample_counts(states, out_dir)
    _plot_stepwise_sample_counts(states, out_dir)
    _plot_story_visuals_184(states, human_states, out_dir)
    _save_distribution_metrics_184(states, human_states, out_dir)
    _save_summary(states, out_dir)

    print("Saved visualizations to:", out_dir)


if __name__ == "__main__":
    main()
