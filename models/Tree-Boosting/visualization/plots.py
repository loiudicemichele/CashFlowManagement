"""
Plotting utilities for model diagnostics and forecast evaluation.

This module provides:
    - Learning curve plot (training vs validation loss over boosting rounds).
    - Forecast comparison plot (actual vs predicted time series, optionally aggregated by date).
    - Feature importance bar plot (for single XGBoost models).
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Optional, List, Union


def plot_learning_curve(evals_result: dict,
                        metric: str = 'mae',
                        title: Optional[str] = None,
                        figsize: tuple = (10, 6),
                        save_path: Optional[str] = None) -> None:
    """
    Plot training and validation loss curves from XGBoost evals_result.

    Args:
        evals_result: Dictionary returned by model.evals_result().
                      Expected structure: {'validation_0': {metric: [...]}, 'validation_1': {...}}
        metric: Metric to plot (e.g., 'mae', 'rmse', 'logloss').
        title: Optional plot title. If None, defaults to 'Learning Curve (XGBoost)'.
        figsize: Figure size (width, height).
        save_path: If provided, save the figure to this path.
    """
    plt.figure(figsize = figsize)

    # Extract data
    train_metric = evals_result['validation_0'][metric]
    val_metric = evals_result['validation_1'][metric]
    print(evals_result)
    epochs = range(1, len(train_metric) + 1)

    plt.plot(epochs, train_metric, label=f'Train {metric.upper()}', color='dodgerblue', linewidth=2)
    plt.plot(epochs, val_metric, label=f'Validation {metric.upper()}', color='red', linewidth=2)

    plt.xlabel('Boosting rounds')
    plt.ylabel(metric.upper())
    plt.title(title or f'Learning Curve (XGBoost - {metric.upper()})')
    plt.legend()
    plt.grid(alpha = 0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_forecast(y_true: Union[pd.Series, pd.DataFrame],
                  y_pred: Union[pd.Series, pd.DataFrame],
                  dates: Optional[pd.Index] = None,
                  title: str = 'Forecast vs Actual',
                  xlabel: str = 'Date',
                  ylabel: str = 'Cash Balance',
                  aggregate_by_date: bool = True,
                  figsize: tuple = (14, 5),
                  save_path: Optional[str] = None) -> None:
    """
    Plot actual vs predicted time series.

    Args:
        y_true: True values (1D or 2D). If 2D and aggregate_by_date=True, will sum over columns.
        y_pred: Predicted values (same shape as y_true).
        dates: Optional datetime index for x-axis. If None, uses index of y_true (if pandas) else range.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        aggregate_by_date: If True and data is multi-store (2D), sum across stores per date.
        figsize: Figure size.
        save_path: If provided, save the figure to this path.
    """
    # Convert to pandas Series/DataFrame if necessary
    if isinstance(y_true, np.ndarray):
        y_true = pd.Series(y_true) if y_true.ndim == 1 else pd.DataFrame(y_true)
    if isinstance(y_pred, np.ndarray):
        y_pred = pd.Series(y_pred) if y_pred.ndim == 1 else pd.DataFrame(y_pred)

    # Handle multi-store aggregation
    if aggregate_by_date and y_true.ndim == 2:
        # Sum across stores (assumes rows are time steps, columns are stores or steps)
        y_true_agg = y_true.sum(axis=1)
        y_pred_agg = y_pred.sum(axis=1)
    elif aggregate_by_date and isinstance(y_true, pd.Series):
        # Already aggregated
        y_true_agg = y_true
        y_pred_agg = y_pred
    else:
        y_true_agg = y_true
        y_pred_agg = y_pred

    # Determine x-axis values
    if dates is not None:
        x = dates
    elif hasattr(y_true_agg, 'index'):
        x = y_true_agg.index
    else:
        x = range(len(y_true_agg))

    plt.figure(figsize=figsize)
    plt.plot(x, y_true_agg, label='Actual', color='#196194', linewidth=1.5)
    plt.plot(x, y_pred_agg, label='Predicted', linestyle='--', color='#8d1a1a', linewidth=1.5)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.4)
    plt.xticks(rotation=45)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_feature_importance(model,
                            importance_type: str = 'weight',
                            max_num_features: Optional[int] = 20,
                            figsize: tuple = (10, 8),
                            title: Optional[str] = None,
                            save_path: Optional[str] = None) -> None:
    """
    Plot feature importance from a fitted XGBoost model (single-output).

    Args:
        model: Fitted XGBRegressor (or XGBoostModel wrapper with get_feature_importance method).
        importance_type: Type of importance ('weight', 'gain', 'cover', 'total_gain', etc.).
        max_num_features: Maximum number of features to display (top N).
        figsize: Figure size.
        title: Plot title. If None, uses 'Feature Importance ({importance_type})'.
        save_path: If provided, save the figure.
    """
    # Extract importance scores
    if hasattr(model, 'get_booster'):
        booster = model.get_booster()
        importance_dict = booster.get_score(importance_type=importance_type)
        if not importance_dict:
            print(f"No feature importance of type '{importance_type}' found. Try 'weight' or 'gain'.")
            return
        # Convert to DataFrame
        imp_df = pd.DataFrame(list(importance_dict.items()), columns=['feature', 'importance'])
        imp_df = imp_df.sort_values('importance', ascending=True)  # ascending for horizontal bar
    elif hasattr(model, 'feature_importances_'):
        # For sklearn-compatible models
        if hasattr(model, 'get_booster'):
            # Already handled
            pass
        else:
            # Assume it's a XGBRegressor with feature_importances_ attribute
            importances = model.feature_importances_
            if hasattr(model, 'get_booster'):
                feature_names = model.get_booster().feature_names
            else:
                feature_names = [f'f{i}' for i in range(len(importances))]
            imp_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
            imp_df = imp_df.sort_values('importance', ascending=True)
    else:
        raise ValueError("Model does not provide feature importance. Use a fitted XGBRegressor or XGBoostModel.")

    # Limit number of features
    if max_num_features and len(imp_df) > max_num_features:
        imp_df = imp_df.tail(max_num_features)

    # Plot horizontal bar chart
    plt.figure(figsize=figsize)
    plt.barh(imp_df['feature'], imp_df['importance'], color='steelblue')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title(title or f'Feature Importance ({importance_type})')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def plot_multi_step_forecast(y_true: Union[pd.Series, pd.DataFrame, np.ndarray],
                             y_pred: Union[pd.DataFrame, np.ndarray],
                             metadata: pd.DataFrame,
                             n_steps: int,
                             title: str = 'Real Vs Predicted Cash Flow (Multi-Step)',
                             xlabel: str = 'Date',
                             ylabel: str = 'Aggregated Cash Balance',
                             figsize: tuple = (15, 6),
                             save_path: Optional[str] = None) -> None:
    """
    Plot actual continuous time series vs sequential multi-step forecast segments.
    Aggregates multi-store data by date and plots a dashed forecast segment 
    every `n_steps` days to avoid overlapping clutter.

    Args:
        y_true: Ground truth values (2D DataFrame where first column is t+1).
        y_pred: Predicted values (2D array/DataFrame of shape n_samples x n_steps).
        metadata: DataFrame containing the 'date' index and 'store_id'.
        n_steps: The forecasting horizon length.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        figsize: Figure size.
        save_path: If provided, save the figure to this path.
    """
    # Aggregate predictions by date
    if isinstance(y_pred, np.ndarray):
        pred_cols = [f'step_{i+1}' for i in range(n_steps)]
        df_preds = pd.DataFrame(y_pred, index=metadata.index, columns=pred_cols)
    else:
        df_preds = y_pred.copy()
        df_preds.index = metadata.index
        
    df_preds_agg = df_preds.groupby(df_preds.index).sum()
    
    # Aggregate real values by date (using the first step t+1 as the continuous real line)
    if isinstance(y_true, np.ndarray):
        real_target = y_true[:, 0]
    else:
        real_target = y_true.iloc[:, 0].values
        
    df_real = pd.DataFrame({'real_target': real_target}, index=metadata.index)
    real_cash_balance = df_real.groupby(df_real.index)['real_target'].sum()

    # Plotting
    plt.figure(figsize=figsize)
    plt.plot(real_cash_balance.index, real_cash_balance.values, 
             label='Real Cash Balance', color='#196194', linewidth=1.5)
    
    dates = df_preds_agg.index
    
    # Plot a segment every n_steps days
    for i in range(0, len(dates), n_steps):
        start_date = dates[i]
        forecast_dates = pd.date_range(start=start_date + pd.Timedelta(days=1), periods=n_steps)
        
        # Ensure we don't plot beyond our available axis dates
        valid_length = min(n_steps, len(real_cash_balance[real_cash_balance.index >= forecast_dates[0]]))
        if valid_length == 0: 
            continue
        
        forecast_values = df_preds_agg.iloc[i].values[:valid_length]
        
        plt.plot(forecast_dates[:valid_length], forecast_values, color='#8d1a1a', alpha=0.9, 
                 linestyle='--', linewidth=1.5, 
                 label='Forecast (Multi-Step)' if i == 0 else "")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()