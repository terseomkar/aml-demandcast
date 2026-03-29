"""
cv.py — Time-series cross-validation for DemandCast

Run from project root with the `.venv` active:
    python cv.py
"""
from pathlib import Path
import sys

import mlflow
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

# keep config consistent with train.py
MLFLOW_TRACKING_URI = "http://localhost:5000"
EXPERIMENT_NAME = "DemandCast"
DATA_PATH = Path(__file__).parent / "data" / "features.parquet"
VAL_CUTOFF = "2025-01-22"   # CV runs only on train+val — test set stays sealed
TARGET = "demand"

# import FEATURE_COLS used in train.py
from src.features_skeleton import FEATURE_COLS


def _rmse(y_true, y_pred):
    return root_mean_squared_error(y_true, y_pred)


def time_series_cv(model, X: pd.DataFrame, y: pd.Series, n_splits: int = 5, run_name: str = "cv_run") -> pd.DataFrame:
    """Run TimeSeriesSplit CV, log per-fold and aggregate metrics to MLflow."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    tscv = TimeSeriesSplit(n_splits=n_splits)
    results = []

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_param("model", type(model).__name__)
        mlflow.log_param("n_splits", n_splits)

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            fold_model = clone(model)
            fold_model.fit(X_train, y_train)
            preds = fold_model.predict(X_test)

            mae = mean_absolute_error(y_test, preds)
            rmse = _rmse(y_test, preds)
            r2 = r2_score(y_test, preds)

            fold_metrics = {
                "fold": fold,
                "mae": round(float(mae), 4),
                "rmse": round(float(rmse), 4),
                "r2": round(float(r2), 4),
            }
            results.append(fold_metrics)

            mlflow.log_metrics(
                {
                    f"fold_{fold}_mae": fold_metrics["mae"],
                    f"fold_{fold}_rmse": fold_metrics["rmse"],
                    f"fold_{fold}_r2": fold_metrics["r2"],
                },
                step=fold,
            )

            print(f"Fold {fold}: MAE={fold_metrics['mae']:.2f} RMSE={fold_metrics['rmse']:.2f} R2={fold_metrics['r2']:.3f}")

        results_df = pd.DataFrame(results)

        # aggregate metrics
        mlflow.log_metrics(
            {
                "cv_mae_mean": float(results_df["mae"].mean()),
                "cv_mae_std": float(results_df["mae"].std()),
                "cv_rmse_mean": float(results_df["rmse"].mean()),
                "cv_rmse_std": float(results_df["rmse"].std()),
                "cv_r2_mean": float(results_df["r2"].mean()),
                "cv_r2_std": float(results_df["r2"].std()),
            }
        )

        return results_df


if __name__ == "__main__":
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Features file not found: {DATA_PATH}")

    df = pd.read_parquet(DATA_PATH)
    df["hour"] = pd.to_datetime(df["hour"])
    trainval = df[df["hour"] < pd.to_datetime(VAL_CUTOFF)].copy()
    if trainval.empty:
        raise ValueError("Train+val split is empty; check VAL_CUTOFF or data range")

    # Sort by time to ensure correct temporal ordering for TimeSeriesSplit
    trainval = trainval.sort_values("hour").reset_index(drop=True)

    # Convert 'hour' datetime to integer hour for modeling (train.py does same)
    trainval["hour"] = trainval["hour"].dt.hour

    X_trainval = trainval[FEATURE_COLS]
    y_trainval = trainval[TARGET]

    # default model (matches train.py's baseline)
    model = RandomForestRegressor(n_estimators=100, random_state=42)

    print("Running time-series CV (n_splits=5) on train+val...")
    results = time_series_cv(model=model, X=X_trainval, y=y_trainval, n_splits=5, run_name="cv_random_forest_100est")

    print("\nCV summary:")
    print(f"MAE:  {results['mae'].mean():.2f} ± {results['mae'].std():.2f}")
    print(f"RMSE: {results['rmse'].mean():.2f} ± {results['rmse'].std():.2f}")
    print(f"R²:   {results['r2'].mean():.3f} ± {results['r2'].std():.3f}")

    print("\nPer-fold breakdown:")
    print(results.to_string(index=False))