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
from typing import Tuple, Optional, List
import numpy as np


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


def load_covariates_df(
    config,
    covariate_cols: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load data as panel DataFrames for Chronos-2's `predict_df` API.

    Chronos-2 requires a "Long" DataFrame format containing:
    - An identifier column (defined by config.item_id_col)
    - A timestamp column (date)
    - The target column
    - Any covariate columns

    Args:
        config (ChronosConfig): Configuration object.
        covariate_cols (List[str], optional): Column names of covariates.

    Returns:
        Tuple containing:
            train_df (pd.DataFrame): Training data with target and covariates.
            test_df  (pd.DataFrame): Test data with target and covariates.
    """
    if covariate_cols is None:
        covariate_cols = config.covariate_cols
        
    train_path = os.path.join(config.data_dir, config.train_file)
    test_path = os.path.join(config.data_dir, config.test_file)

    train_df = pd.read_csv(train_path, parse_dates=['date'])
    test_df = pd.read_csv(test_path, parse_dates=['date'])

    # Get the identifier column from config
    item_id = config.id_col

    # Build the list of strictly required columns
    all_cols = [item_id, 'date', config.target_col] + list(covariate_cols)
    
    # Validate column existence
    missing_train = [c for c in all_cols if c not in train_df.columns]
    missing_test  = [c for c in all_cols if c not in test_df.columns]
    
    if missing_train or missing_test:
        raise ValueError(
            f"Columns not found.\n"
            f"Missing in train: {missing_train}\n"
            f"Missing in test: {missing_test}\n"
            f"Available columns: {list(train_df.columns)}"
        )

    # Filter columns, sort properly (ID first, then Date), and reset index
    train_df = train_df[all_cols].copy().sort_values(by=[item_id, 'date']).reset_index(drop=True)
    test_df = test_df[all_cols].copy().sort_values(by=[item_id, 'date']).reset_index(drop=True)

    # Validate no NaN values (GluonTS/Chronos will crash with NaNs)
    for split_name, df in [("train", train_df), ("test", test_df)]:
        nan_counts = df.isnull().sum()
        if nan_counts.any():
            raise ValueError(f"NaN values detected in {split_name} split:\n{nan_counts[nan_counts > 0]}")

    print(f"\n[+] Panel DataFrames assembled for Chronos-2:")
    print(f"    Item ID column:   {item_id}")
    print(f"    Target column:    {config.target_col}")
    print(f"    Covariate cols:   {len(covariate_cols)}")
    print(f"    Train shape:      {train_df.shape} (Num unique series: {train_df[item_id].nunique()})")
    print(f"    Test shape:       {test_df.shape}\n")

    return train_df, test_df