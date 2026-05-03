"""
Data Package for Chronos Forecasting

Handles loading of partitioned train/test datasets from CSV files,
optionally aggregating data if needed (e.g., summing across stores when
working with aggregated cash balances). The main function `load_data()`
returns pandas Series/DataFrames ready for forecasting.
"""