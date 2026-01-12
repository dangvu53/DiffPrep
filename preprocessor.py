import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, OneHotEncoder
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from scipy.stats import zscore
import category_encoders as ce


# ======================================================
# Preprocessor
# ======================================================
class Preprocessor:
    def __init__(self, config, step_order=None):
        self.config = config

        # order origin
        self.step_order = [
            "imputation",
            "scaling",
            "encoding",
            "outlier_removal",
            "feature_selection",
            "dimensionality_reduction"
        ]
        
        # # order 1
        # self.step_order = [
        #     "imputation",
        #     "encoding",
        #     "outlier_removal",
        #     "scaling",
        #     "feature_selection",
        #     "dimensionality_reduction"
        # ]

        # # order 2
        # self.step_order = [
        #     "imputation",
        #     "encoding",
        #     "outlier_removal",
        #     "scaling",
        #     "dimensionality_reduction",
        #     "feature_selection"
        # ]

        # # order 3
        # self.step_order = [
        #     "imputation",
        #     "encoding",
        #     "outlier_removal",
        #     "feature_selection",
        #     "scaling",
        #     "dimensionality_reduction"
        # ]

        # # order 4
        # self.step_order = [
        #     "imputation",
        #     "encoding",
        #     "scaling",
        #     "outlier_removal",
        #     "feature_selection",
        #     "dimensionality_reduction"
        # ]
        
        self.fitted = False

        # Saved transformers
        self.imputer_num = None
        self.imputer_cat = None
        self.encoder = None
        self.selector = None
        self.scaler = None
        self.reducer = None

        self.num_cols = None
        self.cat_cols = None

    
    # ==================================================
    # FIT
    # ==================================================
    def fit_transform(self, X, y=None):
        self.num_cols = X.select_dtypes(include=['number']).columns.tolist()
        self.cat_cols = X.select_dtypes(exclude=['number']).columns.tolist()

        X_num = X[self.num_cols].copy() if self.num_cols else None
        X_cat = X[self.cat_cols].copy() if self.cat_cols else None

        for step in self.step_order:
            if step == "imputation":
                X_num, X_cat = self._fit_imputation(X_num, X_cat)

            elif step == "outlier_removal":
                X_num, X_cat, y = self._fit_outlier_removal(X_num, X_cat, y)

            elif step == "encoding":
                X_cat = self._fit_encoding(X_cat)

            elif step == "feature_selection":
                X_num, X_cat = self._fit_feature_selection(X_num, X_cat, y)

            elif step == "scaling":
                X_num = self._fit_scaling(X_num)

            elif step == "dimensionality_reduction":
                X_num = self._fit_dim_reduction(X_num)

        # Merge num + cat
        X_out = None
        if X_cat is not None and X_num is not None:
            X_out = pd.concat([X_num, X_cat], axis=1)
        elif X_num is not None:
            X_out = X_num
        elif X_cat is not None:
            X_out = X_cat
            
        self.fitted = True
        return X_out, y

    # ==================================================
    # TRANSFORM
    # ==================================================
    def transform(self, X):
        assert self.fitted, "You must call fit() before transform()"

        X_num = X[self.num_cols].copy() if self.num_cols else None
        X_cat = X[self.cat_cols].copy() if self.cat_cols else None

        for step in self.step_order:
            if step == "imputation":
                X_num, X_cat = self._transform_imputation(X_num, X_cat)

            elif step == "outlier_removal":
                # NO removal on test
                pass

            elif step == "encoding":
                X_cat = self._transform_encoding(X_cat)

            elif step == "feature_selection":
                X_num, X_cat = self._transform_feature_selection(X_num, X_cat)

            elif step == "scaling":
                X_num = self._transform_scaling(X_num)

            elif step == "dimensionality_reduction":
                X_num = self._transform_dim_reduction(X_num)

        if X_cat is not None and X_num is not None:
            return pd.concat([X_num, X_cat], axis=1).reset_index(drop=True)

        if X_cat is not None:
            return X_cat.reset_index(drop=True)

        return X_num.reset_index(drop=True)

    # ======================================================
    # STEP IMPLEMENTATIONS
    # ======================================================

    # -----------------------------
    # 1. Imputation
    # -----------------------------
    def _fit_imputation(self, X_num, X_cat):
        method = self.config["imputation"]
    
        # --- numeric imputer ---
        if X_num is not None and method != "none":
            if method == "knn":
                self.num_imputer = KNNImputer(n_neighbors=min(5, len(X_num)-1))
            elif method in ["mean", "median", "most_frequent", "constant"]:
                self.num_imputer = SimpleImputer(strategy=method)
            else:
                self.num_imputer = SimpleImputer(strategy="mean")
            X_num = pd.DataFrame(self.num_imputer.fit_transform(X_num), columns=X_num.columns)
    
        # --- categorical imputer ---
        if X_cat is not None and method != "none":
            if method in ["most_frequent", "constant"]:
                self.cat_imputer = SimpleImputer(strategy=method, fill_value="missing")
                X_cat = pd.DataFrame(self.cat_imputer.fit_transform(X_cat), columns=X_cat.columns)
            else:
                # unsupported → leave X_cat unchanged
                pass
    
        return X_num, X_cat

        # ---------------------------
        # CATEGORICAL IMPUTATION
        # ---------------------------
        if X_cat is None or method == "none":
            self.imputer_cat = None
        else:
            # always MOST FREQUENT (safe for categorical)
            self.imputer_cat = SimpleImputer(strategy="most_frequent")
            X_cat = pd.DataFrame(self.imputer_cat.fit_transform(X_cat),
                                 columns=X_cat.columns)
    
        return X_num, X_cat


    def _transform_imputation(self, X_num, X_cat):
        # numeric
        if X_num is not None and self.imputer_num is not None:
            X_num = pd.DataFrame(self.imputer_num.transform(X_num),
                                 columns=X_num.columns)
    
        # categorical
        if X_cat is not None and self.imputer_cat is not None:
            X_cat = pd.DataFrame(self.imputer_cat.transform(X_cat),
                                 columns=X_cat.columns)
    
        return X_num, X_cat

    
    def _fit_outlier_removal(self, X_num, X_cat, y):
        method = self.config["outlier_removal"]
        if X_num is None or method == "none":
            return X_num, X_cat, y
    
        # --- IMPORTANT FIX ---
        if X_num is not None:
            X_num = X_num.reset_index(drop=True)
        if X_cat is not None:
            X_cat = X_cat.reset_index(drop=True)
        if y is not None:
            y = y.reset_index(drop=True)
    
        # --- compute mask ---
        if method == "iqr":
            mask = pd.Series(True, index=X_num.index)
            for col in X_num.columns:
                Q1, Q3 = X_num[col].quantile([0.25, 0.75])
                IQR = Q3 - Q1
                if IQR > 0:
                    mask &= (X_num[col] >= Q1 - 1.5 * IQR) & (X_num[col] <= Q3 + 1.5 * IQR)
    
        elif method == "zscore":
            Z = np.abs(zscore(X_num))
            mask = pd.Series((Z < 3).all(axis=1), index=X_num.index)
    
        elif method == "lof":
            lof = LocalOutlierFactor(n_neighbors=20)
            mask = pd.Series(lof.fit_predict(X_num) == 1, index=X_num.index)
    
        elif method == "isolation_forest":
            iso = IsolationForest(contamination=0.05, random_state=42)
            mask = pd.Series(iso.fit_predict(X_num) == 1, index=X_num.index)
    
        # --- apply mask safely ---
        X_num = X_num.loc[mask].reset_index(drop=True)
        if X_cat is not None:
            X_cat = X_cat.loc[mask].reset_index(drop=True)
        if y is not None:
            y = y.loc[mask].reset_index(drop=True)
    
        return X_num, X_cat, y

    # -----------------------------
    # 3. Encoding
    # -----------------------------
    def _fit_encoding(self, X_cat):
        if X_cat is None or self.config["encoding"] == "none":
            return X_cat

        method = self.config["encoding"]

        if method == "onehot":
            self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            arr = self.encoder.fit_transform(X_cat)
            return pd.DataFrame(arr, columns=self.encoder.get_feature_names_out())

        elif method == "frequency":
            self.encoder = ce.CountEncoder(normalize=True)
        elif method == "count":
            self.encoder = ce.CountEncoder(normalize=False)
        elif method == "ordinal":
            self.encoder = ce.OrdinalEncoder()
        elif method == "binary":
            self.encoder = ce.BinaryEncoder()
        else:
            self.encoder = ce.OrdinalEncoder()

        return self.encoder.fit_transform(X_cat)

    def _transform_encoding(self, X_cat):
        if X_cat is None or self.encoder is None:
            return X_cat

        arr = self.encoder.transform(X_cat)

        if hasattr(self.encoder, "get_feature_names_out"):
            return pd.DataFrame(arr, columns=self.encoder.get_feature_names_out())
        return pd.DataFrame(arr)

    # -----------------------------
    # 4. Feature Selection
    # -----------------------------
    def _fit_feature_selection(self, X_num, X_cat, y):
        fs = self.config["feature_selection"]
        if fs == "none":
            return X_num, X_cat
    
        # Only use X_cat if it is encoded (numeric)
        is_cat_encoded = (
            X_cat is not None 
            and all(pd.api.types.is_numeric_dtype(X_cat[col]) for col in X_cat.columns)
        )
    
        # Build X_all
        if X_num is None and not is_cat_encoded:
            return X_num, X_cat     # nothing to select from
    
        if X_num is not None and is_cat_encoded:
            X_all = pd.concat([X_num, X_cat], axis=1)
            num_cols_count = X_num.shape[1]
        elif X_num is not None:
            X_all = X_num.copy()
            num_cols_count = X_num.shape[1]
            X_cat = None
        else:
            X_all = X_cat.copy()
            num_cols_count = 0
    
        # -----------------------------
        # 1. Variance Threshold
        # -----------------------------
        if fs == "variance_threshold":
            self.selector = VarianceThreshold(threshold=0.01)
            arr = self.selector.fit_transform(X_all)
            X_selected = pd.DataFrame(arr)
    
            mask = self.selector.get_support()
    
            # Split back
            if num_cols_count > 0:
                X_num_sel = X_selected.iloc[:, :sum(mask[:num_cols_count])]
            else:
                X_num_sel = None
    
            if is_cat_encoded:
                X_cat_sel = X_selected.iloc[:, sum(mask[:num_cols_count]):]
            else:
                X_cat_sel = X_cat
    
            return X_num_sel, X_cat_sel
    
        # -----------------------------
        # 2. K-best / Mutual Info
        # -----------------------------
        k = min(20, X_all.shape[1])
    
        if fs == "k_best":
            self.selector = SelectKBest(f_classif, k=k)
        else:
            self.selector = SelectKBest(mutual_info_classif, k=k)
    
        arr = self.selector.fit_transform(X_all, y.values.ravel())
        X_selected = pd.DataFrame(arr)
        mask = self.selector.get_support()
    
        # Split back
        if num_cols_count > 0:
            X_num_sel = X_selected.iloc[:, :sum(mask[:num_cols_count])]
        else:
            X_num_sel = None
    
        if is_cat_encoded:
            X_cat_sel = X_selected.iloc[:, sum(mask[:num_cols_count]):]
        else:
            X_cat_sel = X_cat  # untouched
    
        return X_num_sel, X_cat_sel
    
    
    # -----------------------------
    # TRANSFORM
    # -----------------------------
    def _transform_feature_selection(self, X_num, X_cat):
        if self.selector is None:
            return X_num, X_cat
    
        # Only use X_cat if numeric
        is_cat_encoded = (
            X_cat is not None 
            and all(pd.api.types.is_numeric_dtype(X_cat[c]) for c in X_cat.columns)
        )
    
        if X_num is not None and is_cat_encoded:
            X_all = pd.concat([X_num, X_cat], axis=1)
            num_cols_count = X_num.shape[1]
        elif X_num is not None:
            X_all = X_num
            num_cols_count = X_num.shape[1]
            X_cat = None
        else:
            X_all = X_cat
            num_cols_count = 0
    
        arr = self.selector.transform(X_all)
        X_selected = pd.DataFrame(arr)
        mask = self.selector.get_support()
    
        if num_cols_count > 0:
            X_num_sel = X_selected.iloc[:, :sum(mask[:num_cols_count])]
        else:
            X_num_sel = None
    
        if is_cat_encoded:
            X_cat_sel = X_selected.iloc[:, sum(mask[:num_cols_count]):]
        else:
            X_cat_sel = X_cat
    
        return X_num_sel, X_cat_sel


    # -----------------------------
    # 5. Scaling
    # -----------------------------
    def _fit_scaling(self, X):
        method = self.config["scaling"]
        if X is None or method == "none":
            return X

        self.scaler = {
            "standard": StandardScaler(),
            "minmax": MinMaxScaler(),
            "robust": RobustScaler(),
            "maxabs": MaxAbsScaler(),
        }.get(method)

        if self.scaler:
            return pd.DataFrame(self.scaler.fit_transform(X), columns=X.columns)
        return X

    def _transform_scaling(self, X):
        if X is None or self.scaler is None:
            return X
        return pd.DataFrame(self.scaler.transform(X), columns=X.columns)

    # -----------------------------
    # 6. Dimensionality Reduction
    # -----------------------------
    def _fit_dim_reduction(self, X):
        dr = self.config["dimensionality_reduction"]
        if X is None or dr == "none":
            return X

        n_components = min(10, X.shape[1], len(X)-1)

        if dr == "pca":
            self.reducer = PCA(n_components=n_components)
        else:
            self.reducer = TruncatedSVD(n_components=n_components)

        arr = self.reducer.fit_transform(X)
        return pd.DataFrame(arr)

    def _transform_dim_reduction(self, X):
        if X is None or self.reducer is None:
            return X
        arr = self.reducer.transform(X)
        return pd.DataFrame(arr)
