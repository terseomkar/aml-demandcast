"""
tune_refactored.py — Hyperparameter tuning for DemandCast (Optuna + MLflow)
=====================================================================
This refactor uses random splits for train/val/test (70%/20%/10%) and
uses regular K-Fold cross-validation (with shuffling) instead of
TimeSeriesSplit. Everything else mirrors `tune.py` so you can compare
results quickly.

Run from project root with the `.venv` active:
    python tune_refactored.py
"""
from pathlib import Path
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import optuna
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, train_test_split
import datetime

from src.features_skeleton import FEATURE_COLS


# ---------------------------------------------------------------------------
# Configuration — keep in sync with train.py and cv.py
# ---------------------------------------------------------------------------

MLFLOW_TRACKING_URI = "http://localhost:5000"
EXPERIMENT_NAME = "DemandCast"
MODEL_REGISTRY_NAME = "DemandCast"

DATA_PATH = Path(__file__).parent / "data" / "features.parquet"
TARGET = "demand"

N_TRIALS = 10
RANDOM_STATE = 42


def load_splits():
    """Load features.parquet and return random train/val/test splits.

    Splits: train 70%, val 20%, test 10% (random, reproducible via
    `RANDOM_STATE`). Returns X_train, y_train, X_val, y_val.
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Features file not found: {DATA_PATH}")

    df = pd.read_parquet(DATA_PATH)
    # Keep hour as integer (consistent with preprocessing used elsewhere)
    df["hour"] = pd.to_datetime(df["hour"]).dt.hour

    X = df[FEATURE_COLS]
    y = df[TARGET]

    # Step 1: hold out test (10%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.10, random_state=RANDOM_STATE, shuffle=True
    )

    # From remaining 90%, take validation = 20% overall → 20/90 ≈ 0.22222
    val_frac_of_temp = 0.20 / 0.90
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_frac_of_temp, random_state=RANDOM_STATE, shuffle=True
    )

    return X_train, y_train, X_val, y_val


def objective(trial: optuna.Trial) -> float:
    """Optuna objective: suggest hyperparams, run KFold CV on `train`,
    log per-fold metrics to MLflow, and return the mean CV MAE (minimize).
    """
    # --- Part 1: Search space ---
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
        "max_depth": trial.suggest_int("max_depth", 5, 30),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5]),
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }

    # --- Part 2: Load train (val/test kept separate) ---
    X_train, y_train, X_val, y_val = load_splits()

    # Use regular K-Fold CV (with shuffling) on the training partition
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    fold_maes = []

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    run_name = f"optuna_trial_{trial.number}"
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_name_ts = f"{run_name}_{ts}"
    with mlflow.start_run(run_name=run_name_ts) as run:
        mlflow.log_param("logged_at_utc", datetime.datetime.now(datetime.timezone.utc).isoformat())
        mlflow.log_params(params)
        mlflow.log_param("objective", "kfold_train")

        for fold, (train_idx, test_idx) in enumerate(kf.split(X_train), start=1):
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
    """Retrain the chosen hyperparameters on train+val (random split),
    evaluate on test, log test metrics, and register the final model.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    df = pd.read_parquet(DATA_PATH)
    df["hour"] = pd.to_datetime(df["hour"]).dt.hour

    # Split randomly: test 10%, trainval 90%
    X = df[FEATURE_COLS]
    y = df[TARGET]
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.10, random_state=RANDOM_STATE, shuffle=True
    )

    if X_trainval.empty:
        raise ValueError("Train+val split is empty; cannot retrain final model")

    final_model = RandomForestRegressor(**best_params)
    final_model.fit(X_trainval, y_trainval)

    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_name_final = f"final_retrain_and_register_{ts}"
    with mlflow.start_run(run_name=run_name_final) as run:
        mlflow.log_param("logged_at_utc", datetime.datetime.now(datetime.timezone.utc).isoformat())
        mlflow.log_params(best_params)
        if X_test is not None and not X_test.empty:
            test_preds = final_model.predict(X_test)
            test_mae = float(mean_absolute_error(y_test, test_preds))
            mlflow.log_metric("test_mae", test_mae)
            print(f"Final test_mae: {test_mae:.4f}")

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

    best_params = study.best_params
    best_params.setdefault('random_state', RANDOM_STATE)
    # Retrain and register
    retrain_and_register(best_params, stage="Production")
