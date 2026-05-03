"""
Evaluation utilities for forecasting models.

This module provides:
    - Computation of regression metrics (MSE, MAE, MAPE) with multi-output support.
    - Inverse transformation for detrended predictions (log differencing).
    - Saving predictions to CSV along with metadata.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from typing import Union, Dict, Optional, List
from pathlib import Path


def compute_metrics(y_true: Union[pd.Series, pd.DataFrame, np.ndarray],
                    y_pred: Union[pd.Series, pd.DataFrame, np.ndarray],
                    multioutput: str = 'raw_values') -> Dict[str, Union[float, List[float]]]:
    """
    Compute regression metrics (MSE, MAE, MAPE) for single or multi-step forecasts.

    Args:
        y_true: Ground truth values.
                - For single-step: 1D array-like.
                - For multi-step: 2D array-like (n_samples, n_steps).
        y_pred: Predicted values (same shape as y_true).
        multioutput: How to aggregate metrics for multi-step:
                     - 'raw_values': returns a list of metrics per step.
                     - 'uniform_average': returns global average across steps.
                     Default 'raw_values' gives per-step granularity.

    Returns:
        Dictionary with keys 'MSE', 'MAE', 'MAPE', each being either a float
        (if single-step or uniform_average) or a list of floats (if raw_values).
    """
    # Convert to numpy arrays for consistent handling
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Check dimensions
    if y_true.ndim == 1:
        # Single-output case
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred)
        return {'MSE': mse, 'MAE': mae, 'MAPE': mape}
    else:
        # Multi-output case (n_samples, n_steps)
        if multioutput == 'uniform_average':
            # Global average across all steps
            mse = mean_squared_error(y_true, y_pred, multioutput='uniform_average')
            mae = mean_absolute_error(y_true, y_pred, multioutput='uniform_average')
            mape = mean_absolute_percentage_error(y_true, y_pred, multioutput='uniform_average')
            return {'MSE': mse, 'MAE': mae, 'MAPE': mape}
        else:  # raw_values
            mse_per_step = mean_squared_error(y_true, y_pred, multioutput='raw_values')
            mae_per_step = mean_absolute_error(y_true, y_pred, multioutput='raw_values')
            mape_per_step = mean_absolute_percentage_error(y_true, y_pred, multioutput='raw_values')
            return {'MSE': mse_per_step.tolist(),
                    'MAE': mae_per_step.tolist(),
                    'MAPE': mape_per_step.tolist()}


def inverse_detrend_predictions(y_pred_diff: np.ndarray,
                                base_log_series: pd.Series,
                                detrend_period: int = 7) -> np.ndarray:
    """
    Convert differenced log predictions back to original cash_balance scale.

    This is the inverse of the detrending transformation:
        diff = log(cash_balance) - lag_{period}(log(cash_balance))
    So: log(cash_balance) = diff + lag_log
    Then: cash_balance = exp(log_cash_balance)

    Args:
        y_pred_diff: Predicted differenced log values (shape: n_samples,).
        base_log_series: Series of log(cash_balance) at time t - period.
                         Must have same length as y_pred_diff and aligned index.
        detrend_period: The differencing period used (default 7).

    Returns:
        Array of predicted cash_balance values (original scale).
    """
    # Ensure alignment
    if len(y_pred_diff) != len(base_log_series):
        raise ValueError("Length mismatch between predictions and base log series.")
    log_cash_pred = y_pred_diff + base_log_series.values
    cash_pred = np.exp(log_cash_pred)
    return cash_pred


def save_predictions(y_true: Union[pd.Series, pd.DataFrame, np.ndarray],
                     y_pred: Union[pd.Series, pd.DataFrame, np.ndarray],
                     metadata: pd.DataFrame,
                     filepath: str,
                     step_names: Optional[List[str]] = None) -> None:
    """
    Save predictions and true values to CSV, optionally with metadata (date, store_id).

    Args:
        y_true: Ground truth values (1D or 2D).
        y_pred: Predicted values (same shape).
        metadata: DataFrame containing at least 'date' and optionally 'store_id' or other identifiers.
                  Must have same number of rows as y_true.
        filepath: Destination CSV path.
        step_names: List of column names for multi-step predictions (e.g., ['t+1', 't+2']).
                    If None and y_true is 2D, uses default names 'step_1', 'step_2', ...
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to DataFrame for easier handling
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if y_true.ndim == 1:
        # Single step
        df_out = metadata.copy()
        df_out['true_value'] = y_true
        df_out['predicted_value'] = y_pred
    else:
        # Multi-step
        n_steps = y_true.shape[1]
        if step_names is None:
            step_names = [f't+{i+1}' for i in range(n_steps)]
        else:
            if len(step_names) != n_steps:
                raise ValueError("step_names length must match number of steps.")
        df_out = metadata.copy()
        for i, name in enumerate(step_names):
            df_out[f'true_{name}'] = y_true[:, i]
            df_out[f'pred_{name}'] = y_pred[:, i]

    df_out.to_csv(path, index=False)
    print(f"Predictions saved to {path}")