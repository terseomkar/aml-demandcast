"""
tune.py — Hyperparameter tuning for DemandCast (Optuna + MLflow)
===============================================================
Runs an Optuna study to tune a RandomForestRegressor on the train/val
split. Each trial is logged to MLflow; the best run can be registered
to the MLflow Model Registry.

Run from project root with the `.venv` active:
    python tune.py
"""
from pathlib import Path
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import optuna
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from typing import Any

from src.features_skeleton import FEATURE_COLS


# ---------------------------------------------------------------------------
# Configuration — keep in sync with train.py and cv.py
# ---------------------------------------------------------------------------

MLFLOW_TRACKING_URI = "http://localhost:5000"
EXPERIMENT_NAME = "DemandCast"
MODEL_REGISTRY_NAME = "DemandCast"

DATA_PATH = Path(__file__).parent / "data" / "features.parquet"
VAL_CUTOFF = "2025-01-22"
TEST_CUTOFF = "2025-02-01"
TARGET = "demand"

N_TRIALS = 20


def load_splits():
    """Load features.parquet and return train and validation splits.

    Returns
    -------
    X_train, y_train, X_val, y_val
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Features file not found: {DATA_PATH}")

    df = pd.read_parquet(DATA_PATH)
    df["hour"] = pd.to_datetime(df["hour"])

    train = df[df["hour"] < pd.to_datetime(VAL_CUTOFF)].copy()
    val = df[(df["hour"] >= pd.to_datetime(VAL_CUTOFF)) & (df["hour"] < pd.to_datetime(TEST_CUTOFF))].copy()

    # convert hour to integer (consistent with train.py preprocessing)
    train["hour"] = train["hour"].dt.hour
    val["hour"] = val["hour"].dt.hour

    return train[FEATURE_COLS], train[TARGET], val[FEATURE_COLS], val[TARGET]


def objective(trial: optuna.Trial) -> float:
    """Optuna objective: suggest hyperparams, train, evaluate, and log to MLflow.

    Returns validation MAE (to be minimized).
    """
    # --- Part 1: Search space ---
    params = {
        # number of trees: baseline used 100; search up to 500 in steps
        "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
        # tree depth: allow moderate to deep trees to capture nonlinearity
        "max_depth": trial.suggest_int("max_depth", 5, 30),
        # minimum samples per leaf: helps regularize small zones
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        # min samples to split: small values allow fine splits; cap at 20
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        # features considered per split: common choices plus fractional option
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5]),
        "random_state": 42,
        "n_jobs": -1,
    }

    # --- Part 2: Load data, train, evaluate ---
    X_train, y_train, X_val, y_val = load_splits()

    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
    val_preds = model.predict(X_val)
    val_mae = float(mean_absolute_error(y_val, val_preds))

    # --- Part 3: Log trial to MLflow ---
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    run_name = f"optuna_trial_{trial.number}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_metric("val_mae", val_mae)
        # save the trained model artifact so best trials are inspectable
        mlflow.sklearn.log_model(model, "model")

    return val_mae


def register_best(stage: str = "Production") -> None:
    """Find the best MLflow run by val_mae and register it in the Model Registry.

    This function searches runs in the DemandCast experiment and registers
    the model artifact from the run with lowest `val_mae`.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    best_run = mlflow.search_runs(
        experiment_names=[EXPERIMENT_NAME],
        order_by=["metrics.val_mae ASC"],
    ).iloc[0]

    model_uri = f"runs:/{best_run['run_id']}/model"
    registered = mlflow.register_model(model_uri, MODEL_REGISTRY_NAME)

    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(
        name=MODEL_REGISTRY_NAME,
        version=registered.version,
        stage=stage,
    )

    print(f"Registered: {MODEL_REGISTRY_NAME} v{registered.version} → {stage}")
    print(f"  Run ID:  {best_run['run_id']}")
    print(f"  val_mae: {best_run['metrics.val_mae']:.4f}")


if __name__ == "__main__":
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    study = optuna.create_study(direction="minimize")
    print(f"Starting Optuna study ({N_TRIALS} trials)...")
    study.optimize(objective, n_trials=N_TRIALS)

    print(f"\nBest val MAE: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")

    # register the best run across all DemandCast runs (baseline + tuning)
    register_best(stage="Production")
