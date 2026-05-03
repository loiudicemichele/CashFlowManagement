"""
Sequence generation utilities for LSTM forecasting.

This module transforms 2D tabular data into 3D tensors (samples, time steps, features)
required by recurrent neural networks. It supports both single-step and multi-step
forecasting by adjusting the `n_outputs` parameter.
"""

import numpy as np
import torch


def create_sequences(data_df, target_col, n_lags, n_outputs):
    """
    Transform a 2D DataFrame into 3D sequences for supervised learning.

    A sliding window of size `n_lags` is used to create input sequences `X`,
    and the subsequent `n_outputs` values of the target column form the output `y`.

    Args:
        data_df (pd.DataFrame): Scaled DataFrame including features and target.
        target_col (str): Name of the target column to forecast.
        n_lags (int): Number of past time steps to include in each input sequence.
        n_outputs (int): Number of future time steps to predict per sample.

    Returns:
        tuple: (X, y) where:
            - X is a 3D numpy array of shape (samples, n_lags, n_features)
            - y is a 2D numpy array of shape (samples, n_outputs)
    """
    X, y = [], []
    data_array = data_df.values
    target_idx = data_df.columns.get_loc(target_col)

    for i in range(len(data_df) - n_lags - n_outputs + 1):
        # Features from t to t + n_lags - 1
        X.append(data_array[i : i + n_lags, :])
        # Target from t + n_lags to t + n_lags + n_outputs - 1
        y.append(data_array[i + n_lags : i + n_lags + n_outputs, target_idx])

    X = np.array(X)
    y = np.array(y)

    print(f"[+] Sequences generated: X shape {X.shape}, y shape {y.shape}")
    return X, y

def create_sequences_all_stores(data_df, target_col, n_lags, n_outputs, store_col='store_id'):
    """
    Transform a 2D DataFrame into 3D sequences for supervised learning (Panel Data).

    Sequences are generated strictly WITHIN each store to avoid mixing histories.
    After generation, sequences are sorted chronologically.

    Args:
        data_df (pd.DataFrame): Scaled DataFrame including features and target.
        target_col (str): Name of the target column to forecast.
        n_lags (int): Number of past time steps to include in each input sequence.
        n_outputs (int): Number of future time steps to predict per sample.
        store_col (str): Column name identifying the different stores/entities.

    Returns:
        tuple: (X_sorted, y_sorted, dates_sorted)
    """
    X, y, sequence_dates = [], [], []
    
    # Drop store_id to keep only numerical features in the tensor
    features_df = data_df.drop(columns=[store_col]) if store_col in data_df.columns else data_df
    target_idx = features_df.columns.get_loc(target_col)

    # Iterate over each store independently
    for store_id, group in data_df.groupby(store_col):
        group = group.sort_index()
        data_array = group.drop(columns=[store_col]).values
        index_array = group.index
        
        for i in range(len(group) - n_lags - n_outputs + 1):
            X.append(data_array[i : i + n_lags, :])
            y.append(data_array[i + n_lags : i + n_lags + n_outputs, target_idx])
            # Save the exact date the prediction corresponds to (t + n_lags)
            sequence_dates.append(index_array[i + n_lags])

    X = np.array(X)
    y = np.array(y)
    sequence_dates = np.array(sequence_dates)

    # Global chronological sorting
    sort_indices = np.argsort(sequence_dates)
    X_sorted = X[sort_indices]
    y_sorted = y[sort_indices]
    dates_sorted = sequence_dates[sort_indices]

    print(f"[+] Panel sequences generated: X shape {X_sorted.shape}, y shape {y_sorted.shape}")
    return X_sorted, y_sorted, dates_sorted

def prepare_tensors(X, y, device):
    """
    Convert numpy arrays to PyTorch tensors and move them to the specified device.

    Args:
        X (np.ndarray): Input sequences array of shape (samples, n_lags, n_features).
        y (np.ndarray): Target array of shape (samples, n_outputs).
        device (torch.device): Device to place the tensors on (cuda or cpu).

    Returns:
        tuple: (X_tensor, y_tensor) as torch.float32 tensors on the target device.
    """
    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    y_t = torch.tensor(y, dtype=torch.float32).to(device)

    print(f"[+] Tensors created and moved to {device}")
    return X_t, y_t