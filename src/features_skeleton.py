"""
features.py — Feature engineering for DemandCast
=================================================
This module contains all feature engineering logic for the NYC taxi demand
forecasting pipeline. It is imported by pipelines/build_features.py and
src/train.py.

Functions
---------
filter_outliers         Remove rows with implausible sensor/trip values
create_temporal_features Add time-based features from the pickup datetime column
aggregate_to_hourly_demand Aggregate individual trips into hourly demand per zone
add_lag_features        Add lagged demand columns (1h, 24h, 168h) per zone

Constants
---------
FEATURE_COLS            The exact feature columns used during model training.
                        Copy this list verbatim into train.py and dashboard.py
                        to avoid column mismatch errors at prediction time.
"""

import pandas as pd
import numpy as np
from typing import Optional


# ---------------------------------------------------------------------------
# Feature column contract
# ---------------------------------------------------------------------------
# IMPORTANT: Keep this list in sync with train.py and app/dashboard.py.
# Changing a name here without updating those files will break prediction.

FEATURE_COLS: list[str] = [
    "hour",           # 0–23 — extracted from pickup datetime
    "day_of_week",    # 0 = Monday … 6 = Sunday
    "is_weekend",     # 1 if Saturday or Sunday, else 0
    "month",          # 1–12
    "is_rush_hour",   # 1 during 7–9am and 5–7pm on weekdays, else 0
    "demand_lag_1h",  # demand for this zone 1 hour ago
    "demand_lag_24h", # demand for this zone 24 hours ago (same hour yesterday)
    "demand_lag_168h",# demand for this zone 168 hours ago (same hour last week)
]


# ---------------------------------------------------------------------------
# 1. filter_outliers
# ---------------------------------------------------------------------------

def filter_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows with implausible trip values before feature engineering.

    Use the thresholds you determined during EDA (Section 4 of your notebook).
    The defaults below are reasonable starting points — override them if your
    EDA revealed different breakpoints for your data sample.

    Columns checked
    ---------------
    trip_distance : float
        Trips with distance == 0 or > 100 miles are almost certainly errors.
    fare_amount : float
        Negative fares and fares above $500 are not plausible NYC taxi trips.
    passenger_count : int
        Zero passengers and more than 6 passengers are sensor errors.

    Parameters
    ----------
    df : pd.DataFrame
        Raw trip-level DataFrame loaded from the parquet file.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame. Index is reset so it is contiguous after row drops.

    Examples
    --------
    >>> clean_df = filter_outliers(df)
    >>> print(f"Rows removed: {len(df) - len(clean_df)}")
    """
    # Apply filters: keep rows that satisfy all plausibility checks
    mask = pd.Series(True, index=df.index)

    if 'trip_distance' in df.columns:
        mask &= (df['trip_distance'] > 0) & (df['trip_distance'] <= 100)

    if 'fare_amount' in df.columns:
        mask &= (df['fare_amount'] >= 0) & (df['fare_amount'] <= 500)

    if 'passenger_count' in df.columns:
        mask &= (df['passenger_count'] >= 1) & (df['passenger_count'] <= 6)

    return df.loc[mask].reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. create_temporal_features
# ---------------------------------------------------------------------------

def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract time-based features from the tpep_pickup_datetime column.

    All features are derived from a single source column so there is no risk
    of data leakage — we are only decomposing information already present at
    prediction time.

    New columns added
    -----------------
    pickup_hour : datetime64
        The pickup datetime floored to the nearest hour.
        Used as the groupby key in aggregate_to_hourly_demand().
    hour : int
        Hour of day (0–23).
    day_of_week : int
        Day of week (0 = Monday, 6 = Sunday). Use dt.dayofweek.
    is_weekend : int
        1 if day_of_week >= 5, else 0.
    month : int
        Month of year (1–12).
    is_rush_hour : int
        1 if (hour is 7, 8 OR hour is 17, 18) AND day_of_week < 5, else 0.
        Morning rush: 7–9am. Evening rush: 5–7pm. Weekdays only.

    Parameters
    ----------
    df : pd.DataFrame
        Trip-level DataFrame. Must contain column tpep_pickup_datetime.

    Returns
    -------
    pd.DataFrame
        Original DataFrame with new feature columns appended.

    Examples
    --------
    >>> df = create_temporal_features(df)
    >>> df[['hour', 'day_of_week', 'is_weekend', 'is_rush_hour']].head()
    """
    # Locate pickup datetime column (common names: 'tpep_pickup_datetime' or 'pickup_datetime')
    dt_col = None
    for c in df.columns:
        low = c.lower()
        if 'pickup' in low and ('tpep' in low or 'datetime' in low):
            dt_col = c
            break
    if dt_col is None:
        # fallback: any column name that contains 'pickup' and looks datetime-like
        for c in df.columns:
            if 'pickup' in c.lower():
                dt_col = c
                break

    if dt_col is None:
        raise KeyError('No pickup datetime column found in DataFrame')

    df[dt_col] = pd.to_datetime(df[dt_col], errors='coerce')
    df['pickup_hour'] = df[dt_col].dt.floor('H')
    df['hour'] = df[dt_col].dt.hour
    df['day_of_week'] = df[dt_col].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['month'] = df[dt_col].dt.month

    # Define rush hours: morning 7-9, evening 17-19 on weekdays
    is_am_rush = df['hour'].isin([7, 8, 9])
    is_pm_rush = df['hour'].isin([17, 18, 19])
    is_weekday = df['day_of_week'] < 5
    df['is_rush_hour'] = ((is_am_rush | is_pm_rush) & is_weekday).astype(int)

    return df


# ---------------------------------------------------------------------------
# 3. aggregate_to_hourly_demand
# ---------------------------------------------------------------------------

def aggregate_to_hourly_demand(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate individual trips into hourly demand counts per pickup zone.

    This function performs the core transformation that converts the raw
    trip-level data (one row per trip) into the modeling target (one row per
    zone per hour, where the value is the number of pickups).

    Input shape  : (n_trips, many columns)  — e.g. 2.5M rows for January 2024
    Output shape : (n_zones × n_hours, 3)   — e.g. ~260 zones × 744 hours

    Output columns
    --------------
    PULocationID : int
        Pickup zone ID (1–265 in NYC TLC data).
    hour : datetime64
        The hour bucket (pickup_hour floored to the nearest hour).
    demand : int
        Number of taxi pickups in this zone during this hour.

    Parameters
    ----------
    df : pd.DataFrame
        Trip-level DataFrame after create_temporal_features() has been called.
        Must contain columns: PULocationID, pickup_hour.

    Returns
    -------
    pd.DataFrame
        Aggregated demand DataFrame with columns [PULocationID, hour, demand].

    Examples
    --------
    >>> hourly = aggregate_to_hourly_demand(df)
    >>> print(hourly.shape)   # expect (n_zones * n_hours, 3)
    >>> hourly.head()
    """
    if 'PULocationID' not in df.columns:
        # try common alternative names
        pu = next((c for c in df.columns if 'pulocation' in c.lower() or 'pickup_zone' in c.lower()), None)
        if pu is None:
            raise KeyError('PULocationID column not found')
        zone_col = pu
    else:
        zone_col = 'PULocationID'

    if 'pickup_hour' not in df.columns:
        raise KeyError('pickup_hour column not found — run create_temporal_features first')
    # Ensure temporal helper columns exist; derive from pickup_hour if missing
    if 'day_of_week' not in df.columns:
        df['day_of_week'] = df['pickup_hour'].dt.dayofweek
    if 'is_weekend' not in df.columns:
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    if 'month' not in df.columns:
        df['month'] = df['pickup_hour'].dt.month
    if 'is_rush_hour' not in df.columns:
        hour_series = df['pickup_hour'].dt.hour
        is_am = hour_series.isin([7, 8, 9])
        is_pm = hour_series.isin([17, 18, 19])
        df['is_rush_hour'] = ((is_am | is_pm) & (df['day_of_week'] < 5)).astype(int)

    # Aggregate: demand count plus representative temporal columns (first value per group)
    hourly = (
        df.groupby([zone_col, 'pickup_hour'])
          .agg(
              demand=('pickup_hour', 'size'),
              day_of_week=('day_of_week', 'first'),
              is_weekend=('is_weekend', 'first'),
              month=('month', 'first'),
              is_rush_hour=('is_rush_hour', 'first'),
          )
          .reset_index()
    )

    # Normalize column names: zone column -> PULocationID, pickup_hour -> hour
    hourly = hourly.rename(columns={zone_col: 'PULocationID', 'pickup_hour': 'hour'})
    # Reorder columns as requested
    cols = ['PULocationID', 'hour', 'day_of_week', 'is_weekend', 'month', 'is_rush_hour', 'demand']
    hourly = hourly[cols]
    return hourly


# ---------------------------------------------------------------------------
# 4. add_lag_features
# ---------------------------------------------------------------------------

def add_lag_features(
    df: pd.DataFrame,
    zone_col: str = "PULocationID",
    target_col: str = "demand",
) -> pd.DataFrame:
    """Add lagged demand features, computed separately for each zone.

    ⚠️  COMMON BUG WARNING ⚠️
    Lag features MUST be computed per zone using groupby. If you call
    df[target_col].shift(n) without groupby, you will bleed one zone's demand
    into the previous/next zone's lag column. This is a subtle data quality
    bug — the model will train without errors, but the features are wrong.

    Correct pattern:
        df[target_col].shift(n)                          ← WRONG
        df.groupby(zone_col)[target_col].shift(n)        ← CORRECT

    New columns added
    -----------------
    demand_lag_1h : float
        Demand for this zone 1 time-step ago (= 1 hour in the hourly table).
    demand_lag_24h : float
        Demand for this zone 24 time-steps ago (= same hour yesterday).
    demand_lag_168h : float
        Demand for this zone 168 time-steps ago (= same hour last week).

    Note: The first n rows for each zone will be NaN for a lag of n.
    Drop these rows after calling this function, or handle them in your
    training pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Hourly demand DataFrame returned by aggregate_to_hourly_demand().
        Must be sorted by zone and hour before calling this function.
        Must contain columns: zone_col, target_col.
    zone_col : str, optional
        Name of the zone identifier column. Default: 'PULocationID'.
    target_col : str, optional
        Name of the demand column to lag. Default: 'demand'.

    Returns
    -------
    pd.DataFrame
        DataFrame with three new lag columns appended.

    Examples
    --------
    >>> hourly = hourly.sort_values(['PULocationID', 'hour'])
    >>> hourly = add_lag_features(hourly, zone_col='PULocationID', target_col='demand')
    >>> hourly[['PULocationID', 'hour', 'demand', 'demand_lag_1h']].head(10)
    """
    # Ensure expected columns exist
    if zone_col not in df.columns:
        raise KeyError(f'Zone column "{zone_col}" not found in DataFrame')
    if target_col not in df.columns:
        raise KeyError(f'Target column "{target_col}" not found in DataFrame')

    # Sort by zone and hour to ensure shifts are correct
    if 'hour' in df.columns:
        df = df.sort_values([zone_col, 'hour']).reset_index(drop=True)
    else:
        df = df.sort_values([zone_col]).reset_index(drop=True)

    df['demand_lag_1h'] = df.groupby(zone_col)[target_col].shift(1)
    df['demand_lag_24h'] = df.groupby(zone_col)[target_col].shift(24)
    df['demand_lag_168h'] = df.groupby(zone_col)[target_col].shift(168)

    return df
