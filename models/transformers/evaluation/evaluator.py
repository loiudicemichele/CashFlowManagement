"""
Model Evaluation and Metrics Calculation

This module provides:
- compute_metrics(): calculates MAE, RMSE, MAPE given actuals and predictions.
- evaluate_and_save(): runs evaluation, prints results, saves predictions to CSV.
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from typing import Dict, Tuple, Optional


def compute_metrics(actuals: np.ndarray, predictions: np.ndarray) -> Dict[str, float]:
    """
    Compute standard regression metrics on the original scale.

    Args:
        actuals (np.ndarray): Ground truth values (1D array).
        predictions (np.ndarray): Predicted values (1D array, same length as actuals).

    Returns:
        dict: Dictionary containing 'MAE', 'RMSE', 'MAPE' keys with float values.

    Example:
        >>> metrics = compute_metrics(y_true, y_pred)
        >>> print(f"RMSE: {metrics['RMSE']:.2f} €")
    """
    # Ensure inputs are 1D numpy arrays
    actuals = np.asarray(actuals).ravel()
    predictions = np.asarray(predictions).ravel()

    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mape = mean_absolute_percentage_error(actuals, predictions)

    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape
    }


def evaluate_and_save(
    predictions: np.ndarray,
    actuals: np.ndarray,
    date_index: pd.DatetimeIndex,
    config,
    output_dir: Optional[str] = None
) -> Dict[str, float]:
    """
    Evaluate predictions, print metrics, and save results to CSV.

    Args:
        predictions (np.ndarray): 1D array of predicted values.
        actuals (np.ndarray): 1D array of ground truth values.
        date_index (pd.DatetimeIndex): Dates corresponding to the predictions.
        config: ChronosConfig object (used to get target column name and output dir).
        output_dir (str, optional): Directory to save CSV. If None, uses config.output_dir.

    Returns:
        dict: Dictionary with metrics ('MAE', 'RMSE', 'MAPE').
    """
    # Compute metrics
    metrics = compute_metrics(actuals, predictions)

    # Print results
    print("TEST SET EVALUATION RESULTS")
    print(f"MAE:  {metrics['MAE']:,.2f} €")
    print(f"RMSE: {metrics['RMSE']:,.2f} €")
    print(f"MAPE: {metrics['MAPE']:.4%}")
    print("=" * 50 + "\n")

    # Save predictions to CSV
    save_dir = output_dir if output_dir else config.output_dir
    os.makedirs(save_dir, exist_ok=True)

    results_df = pd.DataFrame({
        'date': date_index,
        'actual': actuals,
        'predicted': predictions.ravel()
    })
    results_df.set_index('date', inplace=True)

    csv_path = os.path.join(save_dir, 'forecast_predictions.csv')
    results_df.to_csv(csv_path)
    print(f"[+] Predictions saved to: {csv_path}")

    # Also save metrics as JSON for later reference
    from utils.helpers import save_json
    metrics_path = os.path.join(save_dir, 'test_metrics.json')
    save_json(metrics, metrics_path)

    return metrics