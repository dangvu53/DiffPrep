# AutoGluon evaluation

from autogluon.tabular import TabularPredictor
from autogluon.features.generators import IdentityFeatureGenerator
from sklearn.metrics import accuracy_score, r2_score
import tempfile, os, uuid, shutil, warnings
import numpy as np


import autogluon.tabular
print(autogluon.tabular.__version__)

AUTOGLUON_CONFIG = {
    "eval_metric": "accuracy",
    "time_limit": 300,  # 5 minutes per dataset
    "presets": "best_quality",
    "verbosity": 1,
    "hyperparameter_tune_kwargs": None,
    "ag_args_fit": {
        "ag.max_memory_usage_ratio": 0.9,
    },
    "seed": 42
}


def evaluate_pipeline_with_autogluon_for_test(dataset, pipeline_config):
    """Evaluate a preprocessing pipeline using AutoGluon"""
    try:
        X, y = dataset['X'], dataset['y']

        # Detect problem type BEFORE splitting
        unique = np.unique(y)
        if np.issubdtype(y.dtype, np.number) and len(unique) > 100:
            problem_type = "regression"
        elif len(unique) == 2:
            problem_type = "binary"
        else:
            problem_type = "multiclass"

        # ---------------------------
        # 1) Split BEFORE preprocessing (using repo's split method)
        # ---------------------------
        # Use same split logic as experiment_utils.py to ensure consistent test/train sets
        random_state = 1  # Match repo's default split_seed
        np.random.seed(random_state)
        N = len(y)
        
        val_ratio = 0.2
        test_ratio = 0.2
        n_val = int(N * val_ratio)
        n_test = int(N * test_ratio)
        n_train = N - n_test - n_val

        indices = np.random.permutation(N)
        test_indices = indices[:n_test]
        val_indices = indices[n_test:n_test+n_val]
        train_indices = indices[n_test+n_val:n_test+n_val+n_train]
        
        X_train_raw = X.iloc[train_indices].reset_index(drop=True)
        y_train_raw = y[train_indices]
        X_val_raw = X.iloc[val_indices].reset_index(drop=True)
        y_val_raw = y[val_indices]
        X_test_raw = X.iloc[test_indices].reset_index(drop=True)
        y_test_raw = y[test_indices]

        # ---------------------------
        # 2) Fit preprocessing on train only
        # ---------------------------
        pre = Preprocessor(pipeline_config)
        X_train, y_train = pre.fit_transform(X_train_raw, y_train_raw)
        X_test = pre.transform(X_test_raw)
        y_test = y_test_raw.reset_index(drop=True)

        if X_train.empty or len(y_train) == 0:
            print(f"Empty dataset after preprocessing for {pipeline_config['name']}")
            return np.nan

        # ---------------------------
        # 3) Train AutoGluon
        # ---------------------------

        train_data = X_train.copy()
        train_data["target"] = y_train
        test_data = X_test.copy()
        test_data["target"] = y_test

        temp_dir = os.path.join(tempfile.gettempdir(), f"autogluon_{uuid.uuid4().hex}")
        os.makedirs(temp_dir, exist_ok=True)

        warnings.filterwarnings("ignore", message="path already exists!")

        try:
            predictor = TabularPredictor(
                label="target",
                path=temp_dir,
                problem_type=problem_type,
                eval_metric=("r2" if problem_type == "regression" else "accuracy"),
                verbosity=AUTOGLUON_CONFIG["verbosity"]
            )

            predictor.fit(
                train_data=train_data,
                time_limit=AUTOGLUON_CONFIG["time_limit"],
                presets=AUTOGLUON_CONFIG["presets"],
                #hyperparameter_tune_kwargs=AUTOGLUON_CONFIG["hyperparameter_tune_kwargs"],
                #ag_args_fit=AUTOGLUON_CONFIG["ag_args_fit"],
                feature_generator=IdentityFeatureGenerator()
            )

            pred = predictor.predict(X_test)
            
            test_results = predictor.evaluate(test_data)
            
            if problem_type == "regression":
                return r2_score(y_test, pred)
            else:
                return test_results['accuracy']

        except Exception as e:
            print(f"AutoGluon error for {pipeline_config['name']}: {e}")
            print("Fallback: RandomForest")

            if problem_type == "regression":
                from sklearn.ensemble import RandomForestRegressor
                model = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=10)
            else:
                from sklearn.ensemble import RandomForestClassifier
                model = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=10)

            model.fit(X_train, y_train)
            pred = model.predict(X_test)

            if problem_type == "regression":
                return r2_score(y_test, pred)
            else:
                return accuracy_score(y_test, pred)

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    except Exception as e:
        print(f"Error evaluating pipeline {pipeline_config['name']} on {dataset['name']}: {e}")
        return np.nan
