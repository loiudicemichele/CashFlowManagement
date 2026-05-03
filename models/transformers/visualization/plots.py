"""
Plotting Utilities for Forecast Diagnostics

This module provides:
- plot_forecast(): generates a time series plot comparing actual and predicted
  values.

The function saves the figure to the output directory as 'real_vs_predicted.png'.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Union


def plot_forecast(
    actuals: np.ndarray,
    predictions: np.ndarray,
    date_index: Union[pd.DatetimeIndex, np.ndarray],
    output_dir: str,
    title: Optional[str] = None,
    figsize: tuple = (14, 5),
    actual_color: str = "#196194",
    pred_color: str = "#8d1a1a"
) -> None:
    """
    Plot actual vs predicted cash flow on the test set.

    The function creates a line plot with actual values in solid blue and
    predictions in dashed red. The plot is saved as 'real_vs_predicted.png'
    in the specified output directory.

    Args:
        actuals (np.ndarray): 1D array of ground truth values (original scale).
        predictions (np.ndarray): 1D array of predicted values (same length).
        date_index (pd.DatetimeIndex or np.ndarray): Dates corresponding to the values.
        output_dir (str): Directory where the plot image will be saved.
        title (str, optional): Custom plot title. If None, a default title is used.
        figsize (tuple): Figure dimensions (width, height) in inches. Default (14, 5).
        actual_color (str): Color code for the actual line. Default "#196194" (deep blue).
        pred_color (str): Color code for the predicted line. Default "#8d1a1a" (dark red).

    Example:
        >>> plot_forecast(actuals, preds, test_dates, './outputs/')
    """
    # Ensure inputs are 1D
    actuals = np.asarray(actuals).ravel()
    predictions = np.asarray(predictions).ravel()

    # Create figure
    plt.figure(figsize=figsize)

    # Plot actual values
    plt.plot(date_index, actuals,
             label='Actual Cash Flow',
             linestyle='-',
             linewidth=1.5,
             color=actual_color)

    # Plot predicted values
    plt.plot(date_index, predictions,
             label='Predicted Cash Flow (Chronos)',
             linestyle='--',
             linewidth=1.5,
             color=pred_color)

    # Customize plot
    if title is None:
        title = 'Real vs Predicted Cash Balance - Chronos Zero-Shot Forecast'
    plt.title(title, fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Cash Balance (€)', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, 'real_vs_predicted.png')
    plt.savefig(plot_path, dpi=300)
    plt.show()
    plt.close()

    print(f"[+] Forecast plot saved to: {plot_path}")


def plot_rolling_comparison(
    actuals: np.ndarray,
    predictions: np.ndarray,
    history: np.ndarray,
    history_dates: pd.DatetimeIndex,
    test_dates: pd.DatetimeIndex,
    output_dir: str,
    title: Optional[str] = None
) -> None:
    """
    Extended plot showing historical training data together with test forecast.

    This is useful for presentations where you want to show the transition
    from past observations into the forecasted period.

    Args:
        actuals (np.ndarray): Actual test values.
        predictions (np.ndarray): Predicted test values.
        history (np.ndarray): Historical training values.
        history_dates (pd.DatetimeIndex): Dates of the training period.
        test_dates (pd.DatetimeIndex): Dates of the test period.
        output_dir (str): Output directory for saving the plot.
        title (str, optional): Custom title.
    """
    plt.figure(figsize=(14, 5))

    # Plot historical training data (faded)
    plt.plot(history_dates, history,
             label='Historical (Training)',
             color='gray',
             alpha=0.6,
             linewidth=1)

    # Plot actual test
    plt.plot(test_dates, actuals,
             label='Actual (Test)',
             color="#196194",
             linewidth=1.5)

    # Plot forecast
    plt.plot(test_dates, predictions,
             label='Chronos Forecast',
             color="#8d1a1a",
             linestyle='--',
             linewidth=1.5)

    if title is None:
        title = 'Chronos Zero-Shot Forecast with Historical Context'
    plt.title(title, fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Cash Balance (€)', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, 'forecast_with_history.png')
    plt.savefig(plot_path, dpi=300)
    plt.show()
    plt.close()

    print(f"[+] Extended forecast plot saved to: {plot_path}")