# aml-demandcast
Demand Forecasting - project 1

## What Is the Project?

- **Summary:** Build a demand-forecasting system for New York City Yellow Taxis that predicts short-term passenger demand across the city to improve dispatching, reduce passenger wait times, and inform dynamic pricing or staffing.
- **Scope:** Produce hourly (or finer) demand forecasts at taxi-zone or neighborhood level using historical trips, time features, and external signals; compare baseline time-series models and ML approaches; provide point forecasts and uncertainty estimates.

## What Data?

- **Core data:** Historical taxi trip records (pickup_datetime, pickup_zone, dropoff_zone, passenger_count, trip_distance, fare components, vendor/medallion ID).
- **External signals:** Weather (temperature, precipitation, wind), public events/holidays, transit incidents, traffic/road closures, and POI or population/commuting patterns.
- **Derived features:** Aggregated demand counts per zone/time interval (e.g., pickups per hour), lag/rolling statistics, time-of-day / day-of-week / holiday flags, weather indicators, and spatial neighboring-zone demand.

## What You're Predicting?

- **Primary target:** Number of taxi pickup requests per zone for each forecast interval (e.g., hourly pickups per taxi zone).
- **Optional targets:** Probability of demand surge (large increase), expected trip durations or average wait time, and aggregated citywide demand.
- **Outputs & evaluation:** Produce point forecasts plus prediction intervals; evaluate with MAE/RMSE and calibration metrics; measure operational impact (reduced wait, better utilization).
