"""
dashboard.py — DemandCast Streamlit Dashboard
==============================================
A Streamlit app that loads the Production MLflow model and serves real-time
hourly taxi demand predictions for any NYC pickup zone.

Usage (from project root with .venv active)
-------------------------------------------
    streamlit run app/dashboard.py

Opens at http://localhost:8501.

Before running
--------------
1. MLflow UI must be running:
       mlflow ui
   A model must be registered as "DemandCast/Production" in the Registry.
   If no Production model exists, run tune.py → register_best() first.
2. Streamlit must be installed:
       pip install streamlit

Sections (all student-owned except model loading)
--------------------------------------------------
1. Model loading     Pre-built. Do not modify.
2. Page config       Title, description, layout.       ← TODO
3. Sidebar inputs    User controls for feature values. ← TODO
4. Feature vector    Assemble inputs → DataFrame.      ← TODO  ⚠️ critical
5. Prediction        Run model.predict(), display.     ← TODO
6. Metric context    Plain-language explanation.       ← TODO
7. Visualization     One chart (your choice).          ← TODO
"""

import sys
from pathlib import Path

# dashboard.py lives in app/ — add the project root to sys.path so
# src.features can be imported without installing the package.
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np

from src.features import FEATURE_COLS  # the ground truth for column order


# ---------------------------------------------------------------------------
# Section 1: Model loading — pre-built, do not modify
# ---------------------------------------------------------------------------
# @st.cache_resource loads the model once and caches it across all interactions.
# Without this, every slider move reloads a multi-MB model from MLflow,
# making the app unusably slow.

MLFLOW_TRACKING_URI = "http://localhost:5000"
MODEL_URI           = "models:/DemandCast/Production"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


@st.cache_resource
def load_model():
    """Load the Production model from the MLflow Model Registry."""
    return mlflow.sklearn.load_model(MODEL_URI)


model = load_model()


# ---------------------------------------------------------------------------
# Section 2: Page config — TODO
# ---------------------------------------------------------------------------
# Set the browser tab title, page layout, and a brief description.
# st.set_page_config() must be the first Streamlit call in the script.
#
# Hint:
#   st.set_page_config(page_title="DemandCast", layout="wide")
#   st.title("🚕 DemandCast — NYC Taxi Demand Forecast")
#   st.write("Predict hourly taxi pickup demand for any NYC zone.")

# TODO: Add st.set_page_config(), st.title(), and a one-sentence description.


# ---------------------------------------------------------------------------
# Section 3: Sidebar inputs — TODO
# ---------------------------------------------------------------------------
# Build user controls in the sidebar. Each input maps to one feature column.
# Students choose widget types (slider, selectbox, checkbox, number_input).
#
# Required inputs (minimum):
#   zone        — integer, 1 to 263 (NYC TLC pickup zone IDs)
#   hour        — integer, 0 to 23
#   day_of_week — integer, 0 (Monday) to 6 (Sunday)
#   month       — integer, 1 to 12
#
# Derived inputs (compute from above — do not add separate widgets):
#   is_weekend  — 1 if day_of_week >= 5, else 0
#   is_rush_hour — 1 if (hour in {7,8} OR hour in {17,18}) AND day_of_week < 5
#
# Lag inputs (enter a recent known value or 0 as a proxy):
#   demand_lag_1h   — demand in this zone 1 hour ago
#   demand_lag_24h  — demand in this zone 24 hours ago
#   demand_lag_168h — demand in this zone 168 hours ago (one week)
#
# Hint:
#   st.sidebar.header("Forecast Inputs")
#   zone        = st.sidebar.slider("Pickup Zone", min_value=1, max_value=263, value=132)
#   hour        = st.sidebar.slider("Hour of Day", min_value=0, max_value=23, value=8)
#   day_of_week = st.sidebar.selectbox("Day of Week",
#                     options=[0,1,2,3,4,5,6],
#                     format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x])
#   month       = st.sidebar.slider("Month", min_value=1, max_value=12, value=1)
#   is_weekend  = int(day_of_week >= 5)
#   is_rush_hour = int((hour in {7, 8} or hour in {17, 18}) and day_of_week < 5)
#
#   st.sidebar.subheader("Recent Demand (Lag Features)")
#   demand_lag_1h   = st.sidebar.number_input("Demand 1 hour ago",   min_value=0, value=0)
#   demand_lag_24h  = st.sidebar.number_input("Demand 24 hours ago", min_value=0, value=0)
#   demand_lag_168h = st.sidebar.number_input("Demand 1 week ago",   min_value=0, value=0)

# TODO: Add sidebar widgets and compute derived features.


# ---------------------------------------------------------------------------
# Section 4: Feature vector assembly — TODO  ⚠️ most common bug source
# ---------------------------------------------------------------------------
# Build a single-row DataFrame from the sidebar inputs.
#
# ⚠️  COLUMN ORDER MATTERS ⚠️
# The model was trained with features in the order defined by FEATURE_COLS.
# If your DataFrame has different column names or a different order, predict()
# will either throw a confusing error or silently return wrong values.
#
# Rule: always build the input dict using FEATURE_COLS as the key reference,
# then wrap it with [FEATURE_COLS] to enforce the exact training order.
#
# FEATURE_COLS (imported from src/features.py):
#   ['hour', 'day_of_week', 'is_weekend', 'month', 'is_rush_hour',
#    'demand_lag_1h', 'demand_lag_24h', 'demand_lag_168h']
#
# Hint:
#   input_data = pd.DataFrame([{
#       "hour":            hour,
#       "day_of_week":     day_of_week,
#       "is_weekend":      is_weekend,
#       "month":           month,
#       "is_rush_hour":    is_rush_hour,
#       "demand_lag_1h":   demand_lag_1h,
#       "demand_lag_24h":  demand_lag_24h,
#       "demand_lag_168h": demand_lag_168h,
#   }])
#   input_data = input_data[FEATURE_COLS]   # enforce training column order

# TODO: Build input_data DataFrame and apply [FEATURE_COLS] slice.


# ---------------------------------------------------------------------------
# Section 5: Prediction display — TODO
# ---------------------------------------------------------------------------
# Run model.predict() on the feature vector and display the result.
# st.metric() is the right widget for a single headline number.
#
# Hint:
#   prediction = model.predict(input_data)[0]
#   prediction = max(0, round(prediction))   # demand can't be negative
#
#   st.metric(
#       label=f"Predicted demand — Zone {zone}, {hour:02d}:00",
#       value=f"{prediction} trips",
#   )

# TODO: Run prediction and display with st.metric().


# ---------------------------------------------------------------------------
# Section 6: Plain-language metric context — TODO
# ---------------------------------------------------------------------------
# Show the model's validation MAE alongside the prediction so users understand
# the forecast's typical margin of error. Reuse the interpretation you wrote
# in the Day 1 metrics exercise.
#
# Example format (replace the MAE value with your actual result):
#   st.info(
#       "**Model accuracy:** On the validation set, this model's predictions "
#       "are off by an average of **X trips per hour per zone** (MAE). "
#       "For a busy zone like Midtown at 8am, that means the forecast could "
#       "be X trips higher or lower than actual demand."
#   )

# TODO: Add your plain-language metric explanation using st.info() or st.write().


# ---------------------------------------------------------------------------
# Section 7: Visualization — TODO
# ---------------------------------------------------------------------------
# Add one chart that helps a non-technical user understand demand patterns.
# Choose one:
#
# Option A — Average demand by hour of day (shows rush hour pattern)
#   Use st.bar_chart() or plotly express for a simple hourly profile.
#   Copilot prompt: "# plotly bar chart of average hourly demand by hour of day,
#   with rush hours (7-9am and 5-7pm) highlighted in orange"
#
# Option B — Compare predicted demand across zones for this hour
#   Generate predictions for all 263 zones at the currently selected hour
#   and plot as a bar chart sorted by demand.
#
# Option C — Demand trend for the selected zone over a day
#   Generate predictions for hours 0–23 at the selected zone and day_of_week
#   to show how demand varies through the day.
#
# Hint for Option C:
#   hours   = list(range(24))
#   preds   = []
#   for h in hours:
#       is_rh = int((h in {7,8} or h in {17,18}) and day_of_week < 5)
#       row   = pd.DataFrame([{
#           "hour": h, "day_of_week": day_of_week, "is_weekend": is_weekend,
#           "month": month, "is_rush_hour": is_rh,
#           "demand_lag_1h": demand_lag_1h, "demand_lag_24h": demand_lag_24h,
#           "demand_lag_168h": demand_lag_168h,
#       }])[FEATURE_COLS]
#       preds.append(max(0, round(model.predict(row)[0])))
#
#   chart_df = pd.DataFrame({"hour": hours, "predicted_demand": preds})
#   st.subheader(f"Predicted demand — Zone {zone} by hour")
#   st.bar_chart(chart_df.set_index("hour"))

# TODO: Add your chosen visualization.
