"""
Streamlit dashboard for DemandCast
===============================
Loads the Production model from MLflow and provides inputs to generate
hourly demand forecasts for any NYC pickup zone.

Run with:
    streamlit run app/dashboard.py
"""
import sys
from pathlib import Path

# dashboard.py lives in app/ — add the project root to sys.path so
# src modules can be imported without installing the package.
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np

from src.features_skeleton import FEATURE_COLS


# ---------------------------------------------------------------------------
# Section 1: Model loading
# ---------------------------------------------------------------------------
MLFLOW_TRACKING_URI = "http://localhost:5000"
MODEL_URI = "models:/DemandCast/Production"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


@st.cache_resource
def load_model():
    return mlflow.sklearn.load_model(MODEL_URI)


model = load_model()


# ---------------------------------------------------------------------------
# Section 2: Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="DemandCast", layout="wide")
st.title("🚕 DemandCast — NYC Taxi Demand Forecast")
st.write("Predict hourly taxi pickup demand for any NYC pickup zone. Adjust inputs on the left and see the forecast and hourly profile.")


# ---------------------------------------------------------------------------
# Section 3: Sidebar inputs
# ---------------------------------------------------------------------------
st.sidebar.header("Forecast Inputs")

zone = st.sidebar.slider("Pickup Zone (TLC ID)", min_value=1, max_value=263, value=132)
hour = st.sidebar.slider("Hour of Day", min_value=0, max_value=23, value=8)
day_of_week = st.sidebar.selectbox(
    "Day of Week",
    options=list(range(7)),
    index=0,
    format_func=lambda x: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][x],
)
month = st.sidebar.slider("Month", min_value=1, max_value=12, value=1)

# Derived features
is_weekend = int(day_of_week >= 5)
is_rush_hour = int(((hour in {7, 8}) or (hour in {17, 18})) and day_of_week < 5)

st.sidebar.subheader("Recent demand (lag features)")
demand_lag_1h = st.sidebar.number_input("Demand 1 hour ago", min_value=0, value=0)
demand_lag_24h = st.sidebar.number_input("Demand 24 hours ago", min_value=0, value=0)
demand_lag_168h = st.sidebar.number_input("Demand 1 week ago", min_value=0, value=0)


# ---------------------------------------------------------------------------
# Section 4: Build feature vector (ensure exact order from FEATURE_COLS)
# ---------------------------------------------------------------------------
input_dict = {
    "hour": int(hour),
    "day_of_week": int(day_of_week),
    "is_weekend": int(is_weekend),
    "month": int(month),
    "is_rush_hour": int(is_rush_hour),
    "demand_lag_1h": float(demand_lag_1h),
    "demand_lag_24h": float(demand_lag_24h),
    "demand_lag_168h": float(demand_lag_168h),
}

input_data = pd.DataFrame([input_dict])
# enforce training column order
input_data = input_data[FEATURE_COLS]


# ---------------------------------------------------------------------------
# Section 5: Prediction display
# ---------------------------------------------------------------------------
prediction = model.predict(input_data)[0]
prediction = max(0, int(round(float(prediction))))

st.metric(label=f"Predicted demand — Zone {zone}, {hour:02d}:00", value=f"{prediction} trips")


# ---------------------------------------------------------------------------
# Section 6: Plain-language metric context
# ---------------------------------------------------------------------------
st.markdown("---")
with st.expander("Model accuracy and context (validation)"):
    # try to fetch best val_mae from MLflow experiment
    try:
        exp = mlflow.get_experiment_by_name("DemandCast")
        if exp is not None:
            runs_df = mlflow.search_runs(experiment_ids=[exp.experiment_id])
            if not runs_df.empty and "metrics.val_mae" in runs_df.columns:
                best_val_mae = runs_df["metrics.val_mae"].dropna().astype(float).min()
            else:
                best_val_mae = None
        else:
            best_val_mae = None
    except Exception:
        best_val_mae = None

    if best_val_mae is not None:
        st.info(
            f"**Model accuracy:** On the validation set, this model's predictions are off by an average of **{best_val_mae:.2f} trips per hour per zone (MAE)**. "
            "For a busy zone, that means the forecast could be roughly this many trips higher or lower than actual demand."
        )
    else:
        st.info("Model validation MAE not available — check that MLflow runs exist and the experiment name is 'DemandCast'.")


# ---------------------------------------------------------------------------
# Section 7: Visualization — hourly profile for selected zone (Option C)
# ---------------------------------------------------------------------------
st.subheader(f"Predicted demand — Zone {zone} by hour")
hours = list(range(24))
preds = []
for h in hours:
    is_rh = int(((h in {7, 8}) or (h in {17, 18})) and day_of_week < 5)
    row = pd.DataFrame([{
        "hour": int(h),
        "day_of_week": int(day_of_week),
        "is_weekend": int(is_weekend),
        "month": int(month),
        "is_rush_hour": int(is_rh),
        "demand_lag_1h": float(demand_lag_1h),
        "demand_lag_24h": float(demand_lag_24h),
        "demand_lag_168h": float(demand_lag_168h),
    }])[FEATURE_COLS]
    p = model.predict(row)[0]
    preds.append(max(0, int(round(float(p)))))

chart_df = pd.DataFrame({"hour": hours, "predicted_demand": preds}).set_index("hour")
st.bar_chart(chart_df)
