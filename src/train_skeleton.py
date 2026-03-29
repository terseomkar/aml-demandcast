"""
train.py — Model training and MLflow logging for DemandCast
============================================================
This script loads the engineered feature set, applies a temporal train/val/test
split, and trains regression models to predict hourly taxi demand per zone.
Every run is logged to MLflow — parameters, metrics, and the model artifact.

Usage (from project root with .venv active)
-------------------------------------------
    python train.py

Before running
--------------
1. MLflow UI must be running:
       mlflow ui
   Then open http://localhost:5000 in your browser.
2. features.parquet must exist in data/:
       pipelines/build_features.py produces this file.

Functions
---------
evaluate          Compute MAE, RMSE, and R² for a set of predictions.
                  Already implemented — use it, don't rewrite it.
train_and_log     Load data, split, train one model, log everything to MLflow.
                  This is your TODO.
"""

from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from typing import Any

from src.features import FEATURE_COLS  # src/ is a direct subfolder of the project root


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MLFLOW_TRACKING_URI = "http://localhost:5000"
EXPERIMENT_NAME     = "DemandCast"

DATA_PATH = Path(__file__).parent / "data" / "features.parquet"

# Temporal split cutoffs — January 2024 dataset
# Train:      Jan 1  – Jan 21  (~70%)
# Validation: Jan 22 – Jan 31  (~15%)
# Test:       Feb 1  onward    (~15%, sealed until final evaluation)
VAL_CUTOFF  = "2024-01-22"
TEST_CUTOFF = "2024-02-01"

TARGET = "demand"


# ---------------------------------------------------------------------------
# evaluate() — already implemented, use it as-is
# ---------------------------------------------------------------------------

def evaluate(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    """Compute MAE, RMSE, and R² for a set of predictions.

    This function is pre-built for you. Call it on both the validation set
    and, at the very end, the test set (once only).

    Parameters
    ----------
    y_true : pd.Series
        Ground-truth demand values.
    y_pred : np.ndarray
        Model predictions, same length as y_true.

    Returns
    -------
    dict[str, float]
        Keys: 'mae', 'rmse', 'r2'. Values are floats rounded to 4 decimal places.

    Examples
    --------
    >>> val_preds = model.predict(X_val)
    >>> metrics = evaluate(y_val, val_preds)
    >>> print(f"Val MAE: {metrics['mae']:.2f}  RMSE: {metrics['rmse']:.2f}  R²: {metrics['r2']:.3f}")
    """
    return {
        "mae":  round(mean_absolute_error(y_true, y_pred), 4),
        "rmse": round(root_mean_squared_error(y_true, y_pred), 4),
        "r2":   round(r2_score(y_true, y_pred), 4),
    }


# ---------------------------------------------------------------------------
# train_and_log() — your TODO
# ---------------------------------------------------------------------------

def train_and_log(
    model: Any,
    run_name: str,
    params: dict,
) -> str:
    """Train one regression model and log everything to MLflow.

    This function handles the full training loop for a single model run:
      1. Load data/features.parquet
      2. Apply temporal train/val/test split
      3. Separate features (X) and target (y) for each split
      4. Fit the model on the training set
      5. Evaluate on the validation set using evaluate()
      6. Log params, val metrics, and model artifact to MLflow
      7. Print a summary line and return the MLflow run ID

    The test set must NOT be touched here. Seal it until final evaluation.

    MLflow logging checklist (every run must include all three)
    ----------------------------------------------------------
    mlflow.log_params(params)             — algorithm name + hyperparameters
    mlflow.log_metrics(val_metrics)       — val_mae, val_rmse, val_r2
    mlflow.sklearn.log_model(model, "model") — the fitted model artifact

    Consistent metric naming matters: the MLflow comparison view matches runs
    by key name. Always use 'val_mae', 'val_rmse', 'val_r2' exactly.

    Parameters
    ----------
    model : sklearn estimator
        An unfitted sklearn-compatible regression model, e.g.:
            LinearRegression()
            RandomForestRegressor(n_estimators=100, random_state=42)
    run_name : str
        Human-readable label shown in the MLflow UI, e.g. "linear_regression_baseline".
        Use snake_case. Be specific — "rf_100_estimators" beats "random_forest_v2".
    params : dict
        Dictionary of parameters to log. Must include at minimum:
            {"model": type(model).__name__, ...hyperparameters...}
        Example:
            {"model": "RandomForestRegressor", "n_estimators": 100, "max_depth": 10}

    Returns
    -------
    str
        The MLflow run ID (a hex string). Print it or save it — you can use it
        to retrieve this exact run later with mlflow.get_run(run_id).

    Raises
    ------
    FileNotFoundError
        If data/features.parquet does not exist. Run build_features.py first.
    mlflow.exceptions.MlflowException
        If the MLflow server is not reachable at MLFLOW_TRACKING_URI.
        Fix: start the server with `mlflow ui` from your project root.

    Examples
    --------
    >>> from sklearn.linear_model import LinearRegression
    >>> run_id = train_and_log(
    ...     model=LinearRegression(),
    ...     run_name="linear_regression_baseline",
    ...     params={"model": "LinearRegression"},
    ... )
    >>> print(f"Run logged: {run_id}")
    """
    # TODO: Implement this function following the steps in the docstring.
    #
    # Step-by-step hints:
    #
    # --- 1. Load data ---
    #   df = pd.read_parquet(DATA_PATH)
    #
    # --- 2. Temporal split ---
    #   train = df[df['hour'] < VAL_CUTOFF]
    #   val   = df[(df['hour'] >= VAL_CUTOFF) & (df['hour'] < TEST_CUTOFF)]
    #   # Do not create test splits here — seal the test set
    #
    # --- 3. Separate features and target ---
    #   X_train, y_train = train[FEATURE_COLS], train[TARGET]
    #   X_val,   y_val   = val[FEATURE_COLS],   val[TARGET]
    #
    # --- 4–7. MLflow run ---
    #   mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    #   mlflow.set_experiment(EXPERIMENT_NAME)
    #
    #   with mlflow.start_run(run_name=run_name) as run:
    #       mlflow.log_params(params)
    #
    #       model.fit(X_train, y_train)
    #       val_preds = model.predict(X_val)
    #       val_metrics = evaluate(y_val, val_preds)
    #
    #       # Prefix keys with "val_" so the comparison view groups them together
    #       mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})
    #       mlflow.sklearn.log_model(model, "model")
    #
    #       print(f"[{run_name}] val_mae={val_metrics['mae']:.2f}  "
    #             f"val_rmse={val_metrics['rmse']:.2f}  val_r2={val_metrics['r2']:.3f}")
    #       return run.info.run_id
    pass


# ---------------------------------------------------------------------------
# Entry point — call train_and_log() once per model
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # TODO: Train at least 3 models. Linear Regression baseline is required.
    # For each model:
    #   - Choose initial hyperparameters (not tuning yet — that is Week 4)
    #   - Add a comment explaining WHY you chose those values
    #   - Call train_and_log() with a descriptive run_name
    #
    # --- Model 1: Linear Regression baseline (required) ---
    # Always start with the simplest possible model. If you can't beat this,
    # something is wrong with your features, not your algorithm choice.
    train_and_log(
        model=LinearRegression(),
        run_name="linear_regression_baseline",
        params={"model": "LinearRegression"},
    )

    # --- Model 2: Random Forest ---
    # TODO: Choose n_estimators and max_depth, add a comment justifying them.
    # train_and_log(
    #     model=RandomForestRegressor(n_estimators=100, max_depth=None, random_state=42),
    #     run_name="random_forest_100est",
    #     params={"model": "RandomForestRegressor", "n_estimators": 100, "max_depth": "None"},
    # )

    # --- Model 3: Your choice ---
    # TODO: Pick a third algorithm. Options: Ridge, GradientBoostingRegressor,
    # or XGBRegressor (pip install xgboost). Justify your choice in a comment.
    # train_and_log(
    #     model=GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
    #     run_name="gradient_boosting_baseline",
    #     params={
    #         "model": "GradientBoostingRegressor",
    #         "n_estimators": 100,
    #         "learning_rate": 0.1,
    #     },
    # )
