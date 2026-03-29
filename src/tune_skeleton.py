"""
tune.py — Hyperparameter tuning for DemandCast
================================================
This script runs a systematic Optuna hyperparameter search on your best model
from train.py. Every trial is logged to MLflow. The best trial's model is
registered in the MLflow Model Registry and promoted to Production.

Usage (from project root with .venv active)
-------------------------------------------
    python tune.py

Before running
--------------
1. MLflow UI must be running:
       mlflow ui
   Then open http://localhost:5000 in your browser.
2. features.parquet must exist in data/.
3. Install Optuna if you haven't already:
       pip install optuna

Why Optuna instead of Grid Search or Random Search?
----------------------------------------------------
Grid Search  — exhaustive, tries every combination. At 4 hyperparameters × 5
               values each = 625 runs. Becomes infeasible quickly.
Random Search — samples combinations randomly. Better than grid search for the
               same compute budget (Bergstra & Bengio, 2012), because not all
               hyperparameters matter equally.
Optuna       — Bayesian optimization. Uses results from previous trials to
               decide where to search next. Analogy: grid search tries every
               restaurant alphabetically; random search picks 20 at random;
               Optuna asks "which neighborhoods have the best restaurants?"
               and focuses there.

Functions
---------
load_splits       Load features.parquet and return train/val splits. Pre-built.
objective         Define the search space, train one trial, log to MLflow.
                  This is your TODO.
register_best     Find the best MLflow run and register it as Production.
                  Pre-built — call it after study.optimize() completes.
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

from src.features import FEATURE_COLS  # src/ is a direct subfolder of the project root


# ---------------------------------------------------------------------------
# Configuration — keep in sync with train.py and cv.py
# ---------------------------------------------------------------------------

MLFLOW_TRACKING_URI = "http://localhost:5000"
EXPERIMENT_NAME     = "DemandCast"
MODEL_REGISTRY_NAME = "DemandCast"       # name used in the MLflow Model Registry

DATA_PATH   = Path(__file__).parent / "data" / "features.parquet"
VAL_CUTOFF  = "2024-01-22"
TEST_CUTOFF = "2024-02-01"
TARGET      = "demand"

N_TRIALS    = 20   # number of Optuna trials — increase for a more thorough search


# ---------------------------------------------------------------------------
# load_splits() — already implemented, use it as-is
# ---------------------------------------------------------------------------

def load_splits() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Load features.parquet and return train and validation splits.

    Returns
    -------
    X_train, y_train, X_val, y_val : tuple of DataFrames and Series
        Feature matrices and target vectors for the training and validation sets.
        The test set is intentionally excluded — it stays sealed.

    Raises
    ------
    FileNotFoundError
        If data/features.parquet does not exist. Run pipelines/build_features.py first.
    """
    df = pd.read_parquet(DATA_PATH)
    train = df[df["hour"] < VAL_CUTOFF]
    val   = df[(df["hour"] >= VAL_CUTOFF) & (df["hour"] < TEST_CUTOFF)]
    return (
        train[FEATURE_COLS], train[TARGET],
        val[FEATURE_COLS],   val[TARGET],
    )


# ---------------------------------------------------------------------------
# objective() — your TODO
# ---------------------------------------------------------------------------

def objective(trial: optuna.Trial) -> float:
    """Define one Optuna trial: suggest hyperparameters, train, evaluate, log.

    Optuna calls this function once per trial. Each call:
      1. Suggests a set of hyperparameter values from your defined search space
      2. Trains a fresh model with those values on the training set
      3. Evaluates on the validation set (val MAE is what Optuna minimizes)
      4. Logs the trial as a named MLflow run so you can compare all trials
      5. Returns val MAE — Optuna uses this to decide where to search next

    Search space ownership
    ----------------------
    You define the bounds for each trial.suggest_* call. Every range must have
    a comment justifying why those bounds were chosen — not just copied from an
    example. Think about what you observed in your Week 3 runs: if n_estimators=100
    was clearly too low, start your lower bound higher.

    Optuna suggest methods
    ----------------------
    trial.suggest_int("name", low, high)           — integer in [low, high]
    trial.suggest_int("name", low, high, step=50)  — integer in steps
    trial.suggest_float("name", low, high)          — float in [low, high]
    trial.suggest_float("name", low, high, log=True) — log scale (good for lr)
    trial.suggest_categorical("name", [a, b, c])   — pick one from a list

    Parameters
    ----------
    trial : optuna.Trial
        Optuna trial object. Use trial.suggest_* to sample hyperparameters.

    Returns
    -------
    float
        Validation MAE for this trial. Lower is better. Optuna minimizes this.

    Examples
    --------
    This function is called automatically by study.optimize() — you do not
    call it directly. Optuna handles the loop.
    """
    # TODO: Implement this function in three parts.

    # --- Part 1: Define your search space ---
    # Use trial.suggest_* to sample each hyperparameter.
    # Add a comment after each line justifying the bounds you chose.
    # Example (Random Forest) — adjust the bounds based on your Week 3 results:
    #
    # params = {
    #     "n_estimators":     trial.suggest_int("n_estimators", 100, 500, step=50),
    #         # lower bound: Week 3 baseline used 100, no sign it was too many
    #         # upper bound: 500 balances training time vs. potential gains
    #     "max_depth":        trial.suggest_int("max_depth", 5, 30),
    #         # 5: shallow enough to prevent overfitting on small zones
    #         # 30: deep enough to capture complex demand patterns
    #     "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
    #         # larger values reduce overfitting; upper bound from Week 3 observation
    #     "max_features":     trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5]),
    #         # standard options from sklearn docs; 0.5 is a common practical choice
    #     "random_state": 42,   # fixed — not a hyperparameter
    # }

    # --- Part 2: Load data, build and evaluate the model ---
    # X_train, y_train, X_val, y_val = load_splits()
    # model = RandomForestRegressor(**params)
    # model.fit(X_train, y_train)
    # val_preds = model.predict(X_val)
    # val_mae = mean_absolute_error(y_val, val_preds)

    # --- Part 3: Log this trial to MLflow ---
    # Each trial gets its own run so you can compare all trials in the UI.
    # Use a run_name that identifies it as a tuning trial (not a baseline run).
    #
    # mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    # mlflow.set_experiment(EXPERIMENT_NAME)
    #
    # with mlflow.start_run(run_name=f"optuna_trial_{trial.number}"):
    #     mlflow.log_params(params)
    #     mlflow.log_metric("val_mae", val_mae)
    #     mlflow.sklearn.log_model(model, "model")
    #
    # return val_mae
    pass


# ---------------------------------------------------------------------------
# register_best() — already implemented, use it as-is
# ---------------------------------------------------------------------------

def register_best(stage: str = "Production") -> None:
    """Find the best MLflow run by val_mae and register it in the Model Registry.

    Searches all runs in the DemandCast experiment, finds the one with the
    lowest val_mae, registers its model artifact under MODEL_REGISTRY_NAME,
    and transitions it to the given stage.

    The dashboard and API server load the model by stage name
    ("models:/DemandCast/Production"), not by run ID. This means you can
    swap in a better model just by promoting it — no code changes needed.

    Parameters
    ----------
    stage : str, optional
        Target stage. One of "Staging", "Production", "Archived". Default: "Production".
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    best_run = mlflow.search_runs(
        experiment_names=[EXPERIMENT_NAME],
        order_by=["metrics.val_mae ASC"],
    ).iloc[0]

    model_uri  = f"runs:/{best_run['run_id']}/model"
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
    print(f"  Load with: mlflow.sklearn.load_model('models:/{MODEL_REGISTRY_NAME}/{stage}')")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # --- Run the Optuna study ---
    # direction="minimize" because we want the lowest val MAE
    study = optuna.create_study(direction="minimize")

    # TODO: Uncomment the line below once you've implemented objective().
    # study.optimize(objective, n_trials=N_TRIALS)

    # --- Print results ---
    # TODO: Uncomment after running the study.
    # print(f"\nBest val MAE: {study.best_value:.2f}")
    # print(f"Best params:  {study.best_params}")

    # --- Register the best model as Production ---
    # TODO: Uncomment after running the study.
    # The best run across ALL DemandCast experiments (baseline + tuning) will
    # be registered. If your tuned model is better, it wins automatically.
    # register_best(stage="Production")
    pass
