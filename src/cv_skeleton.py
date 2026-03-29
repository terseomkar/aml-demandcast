"""
cv.py — Time-series cross-validation for DemandCast
=====================================================
This script evaluates a trained model using TimeSeriesSplit cross-validation.
Unlike standard k-fold CV, TimeSeriesSplit always trains on the past and tests
on the future — preserving the temporal ordering of the data.

Usage (from project root with .venv active)
-------------------------------------------
    python cv.py

Before running
--------------
1. MLflow UI must be running:
       mlflow ui
   Then open http://localhost:5000 in your browser.
2. features.parquet must exist in data/.
3. You should already have ≥1 MLflow run from train.py — CV results will be
   logged alongside those runs so you can compare them in the UI.

Why TimeSeriesSplit instead of standard k-fold?
-----------------------------------------------
Standard k-fold randomly shuffles rows into folds. For taxi demand data this
means a test fold can contain rows from January 1st while the training fold
contains rows from January 31st — the model "sees the future." TimeSeriesSplit
avoids this by always placing the test window after the training window:

    Fold 1:  [===train===]                 [=test=]
    Fold 2:  [=====train======]            [=test=]
    Fold 3:  [========train========]       [=test=]
    Fold 4:  [===========train===========] [=test=]
    Fold 5:  [==============train========] [=test=]
                                   time ──────────►

Each test block is always in the future relative to its training block.

Functions
---------
time_series_cv    Run n-fold time-series CV on one model. Log per-fold metrics
                  to MLflow. Return a DataFrame of fold results.
                  This is your TODO.
"""

from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from typing import Any

from src.features import FEATURE_COLS  # src/ is a direct subfolder of the project root


# ---------------------------------------------------------------------------
# Configuration — keep in sync with train.py
# ---------------------------------------------------------------------------

MLFLOW_TRACKING_URI = "http://localhost:5000"
EXPERIMENT_NAME     = "DemandCast"

DATA_PATH   = Path(__file__).parent / "data" / "features.parquet"
VAL_CUTOFF  = "2024-01-22"   # CV runs only on train+val — test set stays sealed
TARGET      = "demand"


# ---------------------------------------------------------------------------
# time_series_cv() — your TODO
# ---------------------------------------------------------------------------

def time_series_cv(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    run_name: str = "cv_run",
) -> pd.DataFrame:
    """Evaluate a model using time-series cross-validation and log results to MLflow.

    Splits X and y into n_splits folds using TimeSeriesSplit, trains a fresh
    copy of the model on each training fold, evaluates on the corresponding
    test fold, and logs per-fold metrics to a single MLflow run.

    ⚠️  KEY PATTERN — clone() inside the loop ⚠️
    Always use clone(model) to create a fresh, unfitted copy for each fold.
    Reusing the same model object across folds carries over fitted state from
    the previous fold and produces corrupted, misleadingly optimistic results.

        fold_model = clone(model)   ← CORRECT — fresh copy every fold
        fold_model = model          ← WRONG   — reuses fitted weights

    Parameters
    ----------
    model : sklearn estimator
        An unfitted sklearn-compatible regression model. The same model
        architecture is cloned and trained independently on each fold.
        Example: RandomForestRegressor(n_estimators=100, random_state=42)
    X : pd.DataFrame
        Feature matrix. Must contain only the columns in FEATURE_COLS.
        Must be sorted by time before calling this function.
    y : pd.Series
        Target values (demand), aligned with X.
    n_splits : int, optional
        Number of CV folds. Default: 5.
        More folds = more stable estimate, but slower and less training data
        per fold. 5 is a reasonable default for one month of hourly data.
    run_name : str, optional
        MLflow run label, e.g. "cv_random_forest_100est".
        Use a name that matches the corresponding train.py run so you can
        compare them side by side in the MLflow UI.

    Returns
    -------
    pd.DataFrame
        One row per fold, columns: ['fold', 'mae', 'rmse', 'r2'].
        Use this to compute the mean and std across folds:
            results['mae'].mean()   — average performance
            results['mae'].std()    — stability across time windows

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> results = time_series_cv(
    ...     model=RandomForestRegressor(n_estimators=100, random_state=42),
    ...     X=X_trainval,
    ...     y=y_trainval,
    ...     n_splits=5,
    ...     run_name="cv_random_forest_100est",
    ... )
    >>> print(f"CV MAE: {results['mae'].mean():.2f} ± {results['mae'].std():.2f}")
    """
    # TODO: Implement this function following the steps below.
    #
    # --- 1. Set up MLflow and TimeSeriesSplit ---
    #   mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    #   mlflow.set_experiment(EXPERIMENT_NAME)
    #   tscv = TimeSeriesSplit(n_splits=n_splits)
    #
    # --- 2. Open a single MLflow run for the entire CV study ---
    #   with mlflow.start_run(run_name=run_name) as run:
    #       mlflow.log_param("model", type(model).__name__)
    #       mlflow.log_param("n_splits", n_splits)
    #
    #       results = []
    #
    # --- 3. Fold loop (inside the with block) ---
    #       for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
    #           X_fold_train = X.iloc[train_idx]
    #           X_fold_test  = X.iloc[test_idx]
    #           y_fold_train = y.iloc[train_idx]
    #           y_fold_test  = y.iloc[test_idx]
    #
    #           # Clone creates a fresh, unfitted copy — never skip this
    #           fold_model = clone(model)
    #           fold_model.fit(X_fold_train, y_fold_train)
    #           preds = fold_model.predict(X_fold_test)
    #
    #           fold_metrics = {
    #               "fold": fold,
    #               "mae":  round(mean_absolute_error(y_fold_test, preds), 4),
    #               "rmse": round(root_mean_squared_error(y_fold_test, preds), 4),
    #               "r2":   round(r2_score(y_fold_test, preds), 4),
    #           }
    #           results.append(fold_metrics)
    #
    #           # Log per-fold metrics with the fold number as a step
    #           mlflow.log_metrics(
    #               {f"fold_mae": fold_metrics["mae"],
    #                f"fold_rmse": fold_metrics["rmse"],
    #                f"fold_r2": fold_metrics["r2"]},
    #               step=fold,
    #           )
    #           print(f"  Fold {fold}: MAE={fold_metrics['mae']:.2f}  "
    #                 f"RMSE={fold_metrics['rmse']:.2f}  R²={fold_metrics['r2']:.3f}")
    #
    # --- 4. Log summary metrics and return results (still inside the with block) ---
    #       results_df = pd.DataFrame(results)
    #       mlflow.log_metrics({
    #           "cv_mae_mean": round(results_df["mae"].mean(), 4),
    #           "cv_mae_std":  round(results_df["mae"].std(), 4),
    #           "cv_rmse_mean": round(results_df["rmse"].mean(), 4),
    #           "cv_r2_mean":  round(results_df["r2"].mean(), 4),
    #       })
    #       return results_df
    pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # TODO: Load features.parquet, build X and y from train+val rows only,
    # then call time_series_cv() on your best model from train.py.
    #
    # --- Load data ---
    # df = pd.read_parquet(DATA_PATH)
    #
    # --- Use only train+val rows — do NOT include the test set ---
    # trainval = df[df['hour'] < VAL_CUTOFF]
    # X_trainval = trainval[FEATURE_COLS]
    # y_trainval = trainval[TARGET]
    #
    # --- Run CV on your best model ---
    # Replace the model below with whichever model had the best val MAE in train.py.
    # from sklearn.ensemble import RandomForestRegressor
    # results = time_series_cv(
    #     model=RandomForestRegressor(n_estimators=100, random_state=42),
    #     X=X_trainval,
    #     y=y_trainval,
    #     n_splits=5,
    #     run_name="cv_random_forest_100est",
    # )
    #
    # --- Print summary ---
    # print(f"\nCV MAE:  {results['mae'].mean():.2f} ± {results['mae'].std():.2f}")
    # print(f"CV RMSE: {results['rmse'].mean():.2f} ± {results['rmse'].std():.2f}")
    # print(f"CV R²:   {results['r2'].mean():.3f} ± {results['r2'].std():.3f}")
    # print("\nPer-fold breakdown:")
    # print(results.to_string(index=False))
    pass
