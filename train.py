"""
train.py — Model training and MLflow logging for DemandCast
============================================================
This script loads the engineered feature set, applies a temporal train/val/test
split, and trains regression models to predict hourly taxi demand per zone.
Every run is logged to MLflow — parameters, metrics, and the model artifact.

Run from project root with the `.venv` active:
    python train.py
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
from pandas.api.types import is_datetime64_any_dtype

from src.features_skeleton import FEATURE_COLS


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MLFLOW_TRACKING_URI = "http://localhost:5000"
EXPERIMENT_NAME = "DemandCast"

DATA_PATH = Path(__file__).parent / "data" / "features.parquet"

VAL_CUTOFF = "2025-01-22"
TEST_CUTOFF = "2025-02-01"

FEATURE_COLS = ['hour', 'day_of_week', 'is_weekend', 'month', 'is_rush_hour',
                'demand_lag_1h', 'demand_lag_24h', 'demand_lag_168h']
TARGET = "demand"


def evaluate(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    """Compute MAE, RMSE, and R² for a set of predictions.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"mae": round(mae, 4), "rmse": round(rmse, 4), "r2": round(r2, 4)}


def train_and_log(model: Any, run_name: str, params: dict) -> str:
    """Train one regression model and log everything to MLflow.

    See src/train_skeleton.py for detailed requirements.
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Features file not found: {DATA_PATH}")

    # Load data
    df = pd.read_parquet(DATA_PATH)

    # Ensure 'hour' is datetime for temporal splitting
    df['hour'] = pd.to_datetime(df['hour'])

    # Temporal split
    val_cut = pd.to_datetime(VAL_CUTOFF)
    test_cut = pd.to_datetime(TEST_CUTOFF)

    train = df[df['hour'] < val_cut]
    val = df[(df['hour'] >= val_cut) & (df['hour'] < test_cut)]

    if train.empty or val.empty:
        raise ValueError('Train or validation split is empty — check VAL_CUTOFF/TEST_CUTOFF or data range')

    # Convert 'hour' datetime to integer hour for modelling on train/val splits
    train = train.copy()
    val = val.copy()
    if is_datetime64_any_dtype(train['hour']):
        train['hour'] = train['hour'].dt.hour
    if is_datetime64_any_dtype(val['hour']):
        val['hour'] = val['hour'].dt.hour

    # Select features and target
    X_train = train[FEATURE_COLS]
    y_train = train[TARGET]
    X_val = val[FEATURE_COLS]
    y_val = val[TARGET]

    # Configure MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)

        # Fit model
        model.fit(X_train, y_train)

        # Validation predictions and metrics
        val_preds = model.predict(X_val)
        val_metrics = evaluate(y_val, val_preds)

        # Log metrics with 'val_' prefix
        mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})

        # Log model artifact
        mlflow.sklearn.log_model(model, "model")

        print(f"[{run_name}] val_mae={val_metrics['mae']:.2f}  "
              f"val_rmse={val_metrics['rmse']:.2f}  val_r2={val_metrics['r2']:.3f}")
        return run.info.run_id


if __name__ == "__main__":
    # Model 1: Linear Regression baseline
    run_id = train_and_log(
        model=LinearRegression(),
        run_name="linear_regression_baseline",
        params={"model": "LinearRegression"},
    )

    # Model 2: Random Forest baseline — reasonable defaults (100 trees)
    run_id = train_and_log(
        model=RandomForestRegressor(n_estimators=100, max_depth=None, random_state=42),
        run_name="random_forest_100est",
        params={"model": "RandomForestRegressor", "n_estimators": 100, "max_depth": None},
    )

    # Model 3: Gradient Boosting baseline — conservative learning rate
    run_id = train_and_log(
        model=GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
        run_name="gradient_boosting_100_lr01",
        params={"model": "GradientBoostingRegressor", "n_estimators": 100, "learning_rate": 0.1},
    )
