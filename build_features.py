# build_features.py — or a notebook cell
from src.features_skeleton import create_temporal_features, aggregate_to_hourly_demand, add_lag_features, filter_outliers
import pandas as pd

df = pd.read_parquet("data/yellow_tripdata_2025-01.parquet")
df = filter_outliers(df)
df = create_temporal_features(df)
hourly = aggregate_to_hourly_demand(df)
hourly = add_lag_features(hourly, zone_col='PULocationID', target_col='demand')
hourly.dropna(inplace=True)   # drop rows where lags are undefined (first hours of data)

# FEATURE_COLS = ['hour', 'day_of_week', 'is_weekend', 'month', 'is_rush_hour',
#                 'demand_lag_1h', 'demand_lag_24h', 'demand_lag_168h']
# TARGET_COL = 'demand'

# hourly_df = hourly[['PULocationID', 'pickup_hour'] + FEATURE_COLS + [TARGET_COL]]
print(df.head())
print(hourly.head())
hourly.to_parquet("data/features.parquet", index=False)
print(f"Feature matrix: {hourly.shape}")