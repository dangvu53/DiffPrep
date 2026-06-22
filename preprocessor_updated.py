# Our new preprocesser with old opeartor space

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
from copy import deepcopy
from sklearn.base import BaseEstimator, TransformerMixin

class FrequencyEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.freq_maps_ = {}
        X = pd.DataFrame(X)
        for col in X.columns:
            freq = X[col].value_counts(normalize=True)
            self.freq_maps_[col] = freq
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for col, freq in self.freq_maps_.items():
            X[col] = X[col].map(freq).fillna(0.0)
        return X.values

    def get_feature_names_out(self, input_features=None):
        return input_features


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
            # "outlier_cleaning",
            "feature_selection",
            "dimensionality_reduction"
        ]
        
        self.fitted = False

        # Saved transformers
        self.num_imputer = None
        self.cat_imputer = None
        self.encoder = None
        self.selector = None
        self.scaler = None
        self.reducer = None

        self.selected_columns_ = None
        self.num_cols = None
        self.cat_cols = None
        self.num_columns_ = None
        self.cat_columns_ = None

        self.outlier_cleaner_num = None
        self.outlier_cleaner_cat = None

    
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

            elif step == "outlier_cleaning":
                X_num, X_cat = self._fit_outlier_cleaning(X_num, X_cat)
                
            elif step == "encoding":
                X_cat = self._fit_encoding(X_cat)

            elif step == "feature_selection":
                X_num, X_cat = self._fit_feature_selection(X_num, X_cat, y)

            elif step == "scaling":
                X_num = self._fit_scaling(X_num)

            elif step == "dimensionality_reduction":
                X_num = self._fit_dim_reduction(X_num)

        if X_num is not None:
            X_num.columns = X_num.columns.astype(str)  # 🔧
        if X_cat is not None:
            X_cat.columns = X_cat.columns.astype(str)  # 🔧

        
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

            elif step == "outlier_cleaning":
                X_num, X_cat = self._transform_outlier_cleaning(X_num, X_cat)
                
            elif step == "encoding":
                X_cat = self._transform_encoding(X_cat)

            elif step == "feature_selection":
                X_num, X_cat = self._transform_feature_selection(X_num, X_cat)

            elif step == "scaling":
                X_num = self._transform_scaling(X_num)

            elif step == "dimensionality_reduction":
                X_num = self._transform_dim_reduction(X_num)

        if X_num is not None:
            X_num.columns = X_num.columns.astype(str)  # 🔧
        if X_cat is not None:
            X_cat.columns = X_cat.columns.astype(str)  # 🔧

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
                self.num_imputer = KNNImputer(
                    n_neighbors=min(5, len(X_num) - 1)
                )
            elif method in ["mean", "median", "most_frequent", "constant"]:
                self.num_imputer = SimpleImputer(strategy=method)
            else:
                self.num_imputer = SimpleImputer(strategy="mean")
    
            X_num = pd.DataFrame(
                self.num_imputer.fit_transform(X_num),
                index=X_num.index,
                columns=X_num.columns
            )
    
        # --- categorical imputer ---
        if X_cat is not None and method != "none":
            self.cat_imputer = SimpleImputer(strategy="most_frequent")

            X_cat = pd.DataFrame(
                self.cat_imputer.fit_transform(X_cat),
                index=X_cat.index,
                columns=X_cat.columns
            )
            
        return X_num, X_cat
    

    def _transform_imputation(self, X_num, X_cat):
        # numeric
        if X_num is not None and self.num_imputer is not None:
            X_num = pd.DataFrame(
                self.num_imputer.transform(X_num),
                index=X_num.index,
                columns=X_num.columns
            )
                
        # categorical
        if X_cat is not None and self.cat_imputer is not None:

            X_cat = pd.DataFrame(
                self.cat_imputer.transform(X_cat),
                index=X_cat.index,
                columns=X_cat.columns
            )
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

    
    def _fit_outlier_cleaning(self, X_num, X_cat):
        method = self.config.get("outlier_cleaning", "none")
    
        # reset state
        self.outlier_cleaner_num = None
        self.outlier_cleaner_cat = None
    
        if method == "none":
            return X_num, X_cat
    
        # check encoded categorical
        is_cat_encoded = (
            X_cat is not None
            and all(pd.api.types.is_numeric_dtype(X_cat[c]) for c in X_cat.columns)
        )
    
        # ------------------------------------------------
        # internal helper
        # ------------------------------------------------
        def _fit_cleaner(X):
            X_array = X.values.astype(float)
            params = {}
    
            # ---------- CELL-WISE METHODS ----------
            if method.startswith("zscore"):
                nstd = float(method.split("-")[1]) if "-" in method else 3.0
                mean = X_array.mean(axis=0)
                std = X_array.std(axis=0)
                cut = std * nstd
                params["lower"] = (mean - cut).reshape(1, -1)
                params["upper"] = (mean + cut).reshape(1, -1)
                params["mode"] = "cell"
    
            elif method.startswith("iqr"):
                k = float(method.split("-")[1]) if "-" in method else 1.5
                q25 = np.percentile(X_array, 25, axis=0)
                q75 = np.percentile(X_array, 75, axis=0)
                iqr = q75 - q25
                cut = iqr * k
                params["lower"] = (q25 - cut).reshape(1, -1)
                params["upper"] = (q75 + cut).reshape(1, -1)
                params["mode"] = "cell"
    
            elif method.startswith("mad"):
                nmad = float(method.split("-")[1]) if "-" in method else 2.5
                median = np.median(X_array, axis=0, keepdims=True)
                mad = np.median(np.abs(X_array - median), axis=0, keepdims=True)
                params["lower"] = median - nmad * mad
                params["upper"] = median + nmad * mad
                params["mode"] = "cell"
    
            # ---------- ROW-WISE METHODS ----------
            elif method == "lof":
                lof = LocalOutlierFactor(n_neighbors=20, novelty=True)
                lof.fit(X_array)
                params["model"] = lof
                params["mode"] = "row"
    
            elif method == "isolation_forest":
                iso = IsolationForest(contamination=0.05, random_state=42)
                iso.fit(X_array)
                params["model"] = iso
                params["mode"] = "row"
    
            else:
                raise ValueError(f"Unknown outlier_cleaning method: {method}")
    
            # ---------- FIT IMPUTER ----------
            X_tmp = deepcopy(X_array)
    
            if params["mode"] == "cell":
                mask = (X_tmp < params["lower"]) | (X_tmp > params["upper"])
                X_tmp[mask] = np.nan
            else:
                row_mask = params["model"].predict(X_tmp) == -1
                X_tmp[row_mask, :] = np.nan
    
            params["imputer"] = SimpleImputer(strategy="mean")
            params["imputer"].fit(X_tmp)
    
            return params
    
        # ------------------------------------------------
        # FIT CLEANERS
        # ------------------------------------------------
        if X_num is not None:
            self.outlier_cleaner_num = _fit_cleaner(X_num)
    
        if is_cat_encoded:
            self.outlier_cleaner_cat = _fit_cleaner(X_cat)
    
        return X_num, X_cat
    
    def _transform_outlier_cleaning(self, X_num, X_cat):
    
        is_cat_encoded = (
            X_cat is not None
            and all(pd.api.types.is_numeric_dtype(X_cat[c]) for c in X_cat.columns)
        )
    
        def _apply_cleaner(X, cleaner):
            if cleaner is None:
                return X
    
            X_array = X.values.astype(float)
    
            if cleaner["mode"] == "cell":
                indicator = (X_array < cleaner["lower"]) | (X_array > cleaner["upper"])
                X_array[indicator] = np.nan
    
            else:  # row-wise
                model = cleaner["model"]
                row_mask = model.predict(X_array) == -1
                X_array[row_mask, :] = np.nan
    
            X_repaired = cleaner["imputer"].transform(X_array)
            return pd.DataFrame(X_repaired, columns=X.columns, index=X.index)
    
        if X_num is not None:
            X_num = _apply_cleaner(X_num, self.outlier_cleaner_num)
    
        if is_cat_encoded:
            X_cat = _apply_cleaner(X_cat, self.outlier_cleaner_cat)
    
        return X_num, X_cat

    # ==================================================
    # 3. ENCODING (UPDATED)
    # ==================================================
    def _fit_encoding(self, X_cat):
        self.encoder = None
        self.label_encoders = None
    
        if X_cat is None or self.config["encoding"] == "none":
            return X_cat
    
        method = self.config["encoding"]
    
        if method == "onehot":
            self.encoder = OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
            arr = self.encoder.fit_transform(X_cat)
            return pd.DataFrame(
                arr,
                index=X_cat.index,
                columns=self.encoder.get_feature_names_out(X_cat.columns).astype(str)
            )
    
        if method == "label":
            self.label_encoders = {}
            X_enc = pd.DataFrame(index=X_cat.index)
    
            for col in X_cat.columns:
                le = LabelEncoder()
                X_enc[col] = le.fit_transform(X_cat[col].astype(str))
                self.label_encoders[col] = le
    
            return X_enc
    
        if method == "frequency":
            self.encoder = FrequencyEncoder()
            arr = self.encoder.fit_transform(X_cat)
            return pd.DataFrame(arr, index=X_cat.index, columns=X_cat.columns.astype(str))
    
        raise ValueError(f"Unknown encoding: {method}")

    def _transform_encoding(self, X_cat):
        if X_cat is None:
            return X_cat
    
        if self.label_encoders is not None:
            X_enc = pd.DataFrame(index=X_cat.index)
            for col, le in self.label_encoders.items():
                X_enc[col] = le.transform(X_cat[col].astype(str))
            return X_enc
    
        if self.encoder is not None:
            arr = self.encoder.transform(X_cat)
            return pd.DataFrame(
                arr,
                index=X_cat.index,
                columns=self.encoder.get_feature_names_out(X_cat.columns)
                if hasattr(self.encoder, "get_feature_names_out")
                else X_cat.columns.astype(str)
            )
    
        return X_cat


    
    # -----------------------------
    # 4. FEATURE SELECTION (FIT)
    # -----------------------------
    def _fit_feature_selection(self, X_num, X_cat, y):
        fs = self.config["feature_selection"]

        # reset state
        self.selector = None
        self.selected_columns_ = None
        self.num_columns_ = pd.Index([])
        self.cat_columns_ = pd.Index([])

        if fs == "none":
            return X_num, X_cat

        # check if categorical is already encoded
        is_cat_encoded = (
            X_cat is not None
            and all(pd.api.types.is_numeric_dtype(X_cat[c]) for c in X_cat.columns)
        )

        # nothing to select from
        if X_num is None and not is_cat_encoded:
            return X_num, X_cat

        # build X_all (IMPORTANT: deterministic order)
        if X_num is not None and is_cat_encoded:
            X_all = pd.concat([X_num, X_cat], axis=1)
            self.num_columns_ = X_num.columns
            self.cat_columns_ = X_cat.columns

        elif X_num is not None:
            X_all = X_num.copy()
            self.num_columns_ = X_num.columns

        else:
            X_all = X_cat.copy()
            self.cat_columns_ = X_cat.columns

        # -----------------------------
        # SELECTOR
        # -----------------------------
        if fs == "variance_threshold":
            self.selector = VarianceThreshold(threshold=0.01)
            self.selector.fit(X_all)

        else:
            k = min(20, X_all.shape[1])

            if fs == "k_best":
                self.selector = SelectKBest(f_classif, k=k)
                self.selector.fit(X_all, y.values.ravel())

            elif fs == "mutual_info":
                self.selector = SelectKBest(
                    lambda X, y: mutual_info_classif(
                        X,
                        y,
                        discrete_features="auto"
                    ),
                    k=k
                )
                self.selector.fit(X_all, y.values.ravel())

            else:
                raise ValueError(f"Unknown feature_selection: {fs}")

        # -----------------------------
        # STORE SELECTED COLUMNS
        # -----------------------------
        support = self.selector.get_support()
        self.selected_columns_ = X_all.columns[support]

        X_selected = X_all[self.selected_columns_]

        # split back safely by column name
        X_num_sel = (
            X_selected[self.selected_columns_.intersection(self.num_columns_)]
            if len(self.num_columns_) > 0 else None
        )

        X_cat_sel = (
            X_selected[self.selected_columns_.intersection(self.cat_columns_)]
            if is_cat_encoded else X_cat
        )

        return X_num_sel, X_cat_sel

    # -----------------------------
    # FEATURE SELECTION (TRANSFORM)
    # -----------------------------
    def _transform_feature_selection(self, X_num, X_cat):
        if self.selector is None:
            return X_num, X_cat

        is_cat_encoded = (
            X_cat is not None
            and all(pd.api.types.is_numeric_dtype(X_cat[c]) for c in X_cat.columns)
        )

        # rebuild X_all in the SAME ORDER as fit
        if len(self.num_columns_) > 0 and len(self.cat_columns_) > 0:
            X_all = pd.concat(
                [X_num[self.num_columns_], X_cat[self.cat_columns_]],
                axis=1
            )
        elif len(self.num_columns_) > 0:
            X_all = X_num[self.num_columns_]
        else:
            X_all = X_cat[self.cat_columns_]

        # transform
        arr = self.selector.transform(X_all)

        X_selected = pd.DataFrame(
            arr,
            index=X_all.index,
            columns=self.selected_columns_
        )

        # split back safely
        X_num_sel = (
            X_selected[self.selected_columns_.intersection(self.num_columns_)]
            if len(self.num_columns_) > 0 else None
        )

        X_cat_sel = (
            X_selected[self.selected_columns_.intersection(self.cat_columns_)]
            if is_cat_encoded else X_cat
        )

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
            return pd.DataFrame(self.scaler.fit_transform(X), index=X.index, columns=X.columns)
        return X

    def _transform_scaling(self, X):
        if X is None or self.scaler is None:
            return X
        return pd.DataFrame(self.scaler.transform(X), index=X.index, columns=X.columns)

    # -----------------------------
    # 6. Dimensionality Reduction
    # -----------------------------
    def _fit_dim_reduction(self, X):
        dr = self.config["dimensionality_reduction"]
        if (
            X is None
            or dr == "none"
            or X.shape[1] <= 1
            or len(X) < 2
        ):
            self.reducer = None
            return X

        n_components = min(10, X.shape[1], len(X)-1)

        if dr == "pca":
            self.reducer = PCA(n_components=n_components)
        else:
            self.reducer = TruncatedSVD(n_components=n_components)

        arr = self.reducer.fit_transform(X)
        return pd.DataFrame(arr, index=X.index)

    def _transform_dim_reduction(self, X):
        if X is None or self.reducer is None:
            return X
        arr = self.reducer.transform(X)
        return pd.DataFrame(arr, index=X.index)