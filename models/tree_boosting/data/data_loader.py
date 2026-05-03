"""
Data loading utilities for time series forecasting.

This module handles:
    - Loading train/test CSV files with proper date parsing.
    - Optional scaling of features and target using StandardScaler.
    - Saving and loading scalers with joblib (compatible with sklearn).
"""

import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional, Dict, Any
from pathlib import Path
import json

def load_data(train_path: str,
              test_path: str,
              index_col: str = 'date',
              parse_dates: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load train and test datasets from CSV files.

    Args:
        train_path: Path to training CSV file.
        test_path: Path to testing CSV file.
        index_col: Column to use as index (default: 'date').
        parse_dates: Whether to parse dates (default: True).

    Returns:
        Tuple of (train_df, test_df) as pandas DataFrames.

    Example:
        train, test = load_data("../Datasets/train.csv", "../Datasets/test.csv")
    """
    train_df = pd.read_csv(train_path, index_col=index_col, parse_dates=parse_dates)
    test_df = pd.read_csv(test_path, index_col=index_col, parse_dates=parse_dates)

    print(f"Loaded train shape: {train_df.shape}")
    print(f"Loaded test shape: {test_df.shape}")
    return train_df, test_df

def save_best_params(params: Dict[str, Any], filepath: str) -> None:
    """
    Save best hyperparameters to a JSON file.

    Args:
        params: Dictionary of hyperparameters (e.g., grid_search.best_params_).
        filepath: Destination path (will save as .json).
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.with_suffix('.json'), 'w') as f:
        json.dump(params, f, indent=4)
    print(f"Best parameters saved to {path.with_suffix('.json')}")


def load_best_params(filepath: str) -> Dict[str, Any]:
    """
    Load best hyperparameters from a JSON file.

    Args:
        filepath: Path to the .json file.

    Returns:
        Dictionary of hyperparameters.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, 'r') as f:
        params = json.load(f)
    return params