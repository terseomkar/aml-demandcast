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
from sklearn.model_selection import TimeSeriesSplit
import datetime

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

N_TRIALS = 10


def _sort_for_time_series_cv(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Return X/y sorted in chronological order for TimeSeriesSplit.

    TimeSeriesSplit uses row order only, so we must sort explicitly by the
    original timestamp before generating folds.
    """
    if isinstance(X.index, pd.DatetimeIndex):
        order = X.index.argsort()
        return X.iloc[order], y.iloc[order]

    timestamp_cols = [
        col for col in ("timestamp", "datetime", "date", "ds")
        if col in X.columns and pd.api.types.is_datetime64_any_dtype(X[col])
    ]
    if timestamp_cols:
        sort_col = timestamp_cols[0]
        ordered = X.assign(__target__=y).sort_values(sort_col, kind="stable")
        return ordered.drop(columns="__target__"), ordered["__target__"]

    raise ValueError(
        "TimeSeriesSplit requires chronologically ordered samples, but no "
        "datetime index or datetime timestamp column was found in X_train."
    )


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
    """Optuna objective: suggest hyperparams, run TimeSeriesSplit CV on `train`,
    log per-fold metrics to MLflow, and return the mean CV MAE (minimize).
    """
    # --- Part 1: Search space ---
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
        "max_depth": trial.suggest_int("max_depth", 5, 30),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5]),
        "random_state": 42,
        "n_jobs": -1,
    }

    # --- Part 2: Load train (and val kept separate) ---
    X_train, y_train, X_val, y_val = load_splits()
    X_train, y_train = _sort_for_time_series_cv(X_train, y_train)

    # Use TimeSeriesSplit CV on the training partition to get a robust objective
    tscv = TimeSeriesSplit(n_splits=5)
    fold_maes = []

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    run_name = f"optuna_trial_{trial.number}"
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_name_ts = f"{run_name}_{ts}"
    with mlflow.start_run(run_name=run_name_ts) as run:
        # Timestamp marker for when this trial logged metrics/artifacts (timezone-aware UTC)
        mlflow.log_param("logged_at_utc", datetime.datetime.now(datetime.timezone.utc).isoformat())
        mlflow.log_params(params)
        mlflow.log_param("objective", "tscv_train")

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X_train), start=1):
            X_tr, X_te = X_train.iloc[train_idx], X_train.iloc[test_idx]
            y_tr, y_te = y_train.iloc[train_idx], y_train.iloc[test_idx]

            model = RandomForestRegressor(**params)
            model.fit(X_tr, y_tr)
            preds = model.predict(X_te)
            fold_mae = float(mean_absolute_error(y_te, preds))
            fold_maes.append(fold_mae)
            mlflow.log_metric(f"fold_{fold}_mae", fold_mae, step=fold)

        mean_cv_mae = float(np.mean(fold_maes))
        mlflow.log_metric("mean_cv_mae", mean_cv_mae)

        # Also evaluate the trial's model (trained on full X_train) on the held-out val
        final_model = RandomForestRegressor(**params)
        final_model.fit(X_train, y_train)
        val_preds = final_model.predict(X_val)
        val_mae = float(mean_absolute_error(y_val, val_preds))
        mlflow.log_metric("val_mae", val_mae)

        # Save the model trained on full X_train (so trial artifacts are inspectable)
        mlflow.sklearn.log_model(final_model, "model")

    # Primary objective: mean CV MAE on train (minimize)
    return mean_cv_mae


def retrain_and_register(best_params: dict, stage: str = "Production") -> None:
    """Retrain the chosen hyperparameters on train+val, evaluate on test,
    log test metrics, and register the final model to the Model Registry.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    # Load full train+val and test splits
    df = pd.read_parquet(DATA_PATH)
    df["hour"] = pd.to_datetime(df["hour"])

    trainval = df[df["hour"] < pd.to_datetime(TEST_CUTOFF)].copy()
    test = df[df["hour"] >= pd.to_datetime(TEST_CUTOFF)].copy()

    if trainval.empty:
        raise ValueError("Train+val split is empty; cannot retrain final model")

    # Prepare data
    trainval["hour"] = trainval["hour"].dt.hour
    X_trainval = trainval[FEATURE_COLS]
    y_trainval = trainval[TARGET]

    if test.empty:
        print("Warning: Test split is empty; registering model without test evaluation")
        X_test = None
        y_test = None
    else:
        test["hour"] = test["hour"].dt.hour
        X_test = test[FEATURE_COLS]
        y_test = test[TARGET]

    # Retrain final model on train+val
    final_model = RandomForestRegressor(**best_params)
    final_model.fit(X_trainval, y_trainval)

    # Start an MLflow run to log final model + test metrics
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_name_final = f"final_retrain_and_register_{ts}"
    with mlflow.start_run(run_name=run_name_final) as run:
        # Timestamp marker for when the final retrain and registration happened (timezone-aware UTC)
        mlflow.log_param("logged_at_utc", datetime.datetime.now(datetime.timezone.utc).isoformat())
        mlflow.log_params(best_params)
        if X_test is not None:
            test_preds = final_model.predict(X_test)
            test_mae = float(mean_absolute_error(y_test, test_preds))
            mlflow.log_metric("test_mae", test_mae)
            print(f"Final test_mae: {test_mae:.4f}")

        # Log and register model
        mlflow.sklearn.log_model(final_model, "model")
        model_uri = f"runs:/{run.info.run_id}/model"
        registered = mlflow.register_model(model_uri, MODEL_REGISTRY_NAME)

        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=MODEL_REGISTRY_NAME,
            version=registered.version,
            stage=stage,
        )

        print(f"Registered final model: {MODEL_REGISTRY_NAME} v{registered.version} → {stage}")
        print(f"  Run ID: {run.info.run_id}")


if __name__ == "__main__":
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    study = optuna.create_study(direction="minimize")
    print(f"Starting Optuna study ({N_TRIALS} trials)...")
    study.optimize(objective, n_trials=N_TRIALS)

    print(f"\nBest mean CV MAE (objective): {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")

    # Retrain the best configuration on train+val and register the final model
    best_params = study.best_params
    # Ensure keys like 'n_jobs' and 'random_state' exist to control behavior
    best_params.setdefault('random_state', 42)
    # Retrain and register
    retrain_and_register(best_params, stage="Production")
