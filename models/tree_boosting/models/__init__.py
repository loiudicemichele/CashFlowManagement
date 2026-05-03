"""
Models package for time series forecasting.

Provides a unified wrapper for XGBoost models with support for:
    - Single-step forecasting (standard XGBRegressor)
    - Multi-step direct forecasting (MultiOutputRegressor)
    - Separate models per step (one model per future horizon)
"""