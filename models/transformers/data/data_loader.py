"""
Data Loading Utilities for Chronos Forecasting

This module provides functions to load training and test datasets from CSV files,
with proper date parsing and sorting. It supports both aggregated data (single
time series) and panel data (multiple stores) – for Chronos we typically use
aggregated daily cash balance.

The loaded data is returned as pandas Series (for the target column) or
DataFrame (if other features are needed). Chronos works with univariate time
series, so we focus on the target column only.
"""

import os
import pandas as pd
from typing import Tuple, Optional


def load_data(config) -> Tuple[pd.Series, pd.Series]:
    """
    Load training and test datasets for the target column.

    Reads CSV files specified in the config, parses the 'date' column as
    datetime index, sorts chronologically, and extracts the target column
    as a pandas Series.

    Args:
        config (ChronosConfig): Configuration object containing:
            - data_dir: directory where CSV files are stored
            - train_file: filename of training set
            - test_file: filename of test set
            - target_col: name of the target column to forecast

    Returns:
        tuple: (train_series, test_series) where each is a pandas Series
               with datetime index and values as float.

    """
    train_path = os.path.join(config.data_dir, config.train_file)
    test_path = os.path.join(config.data_dir, config.test_file)

    # Load CSVs with date parsing
    train_df = pd.read_csv(train_path, index_col='date', parse_dates=['date'])
    test_df = pd.read_csv(test_path, index_col='date', parse_dates=['date'])

    # Sort by date just in case
    train_df.sort_index(inplace=True)
    test_df.sort_index(inplace=True)

    # Extract target column as Series
    train_series = train_df[config.target_col].copy()
    test_series = test_df[config.target_col].copy()

    print(f"[+] Training data loaded: {len(train_series)} days ({train_series.index.min()} to {train_series.index.max()})")
    print(f"[+] Test data loaded:     {len(test_series)} days ({test_series.index.min()} to {test_series.index.max()})")

    return train_series, test_series


def load_data_with_features(config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load full DataFrames (including features) for potential future use.

    Chronos performs zero-shot forecasting on the raw time series without
    external features, but this function is provided for consistency with
    the LSTM pipeline and in case features are needed for comparison.

    Args:
        config (ChronosConfig): Configuration object.

    Returns:
        tuple: (train_df, test_df) as complete DataFrames with datetime index.
    """
    train_path = os.path.join(config.data_dir, config.train_file)
    test_path = os.path.join(config.data_dir, config.test_file)

    train_df = pd.read_csv(train_path, index_col='date', parse_dates=['date'])
    test_df = pd.read_csv(test_path, index_col='date', parse_dates=['date'])

    train_df.sort_index(inplace=True)
    test_df.sort_index(inplace=True)

    print(f"[+] Training DataFrame loaded: {train_df.shape}")
    print(f"[+] Test DataFrame loaded:     {test_df.shape}")

    return train_df, test_df