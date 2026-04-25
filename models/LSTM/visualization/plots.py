"""
Plotting Utilities for Model Diagnostics

This module provides functions to visualize training progress (learning curves)
and forecast results (actual vs predicted time series). Plots are saved to the
output directory as high-resolution PNG images.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_learning_curve(output_dir):
    """
    Plot the training and validation loss curves from the saved CSV history.

    The plot is saved as 'learning_curve.png' in the output directory.

    Args:
        output_dir (str): Directory containing 'training_loss_history.csv'.
    """
    loss_path = os.path.join(output_dir, 'training_loss_history.csv')
    if not os.path.exists(loss_path):
        print(f"[!] Loss history not found at {loss_path}")
        return

    loss_df = pd.read_csv(loss_path)

    plt.figure(figsize=(10, 5))
    plt.plot(loss_df['epoch'], loss_df['train_loss'],
             label='Train Loss (MSE)', color='blue', linewidth=2)
    plt.plot(loss_df['epoch'], loss_df['val_loss'],
             label='Validation Loss (MSE)', color='orange', linewidth=2)

    plt.title('LSTM Learning Curve (Train vs Validation Loss)', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Mean Squared Error (Scaled)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_path = os.path.join(output_dir, 'learning_curve.png')
    plt.savefig(plot_path, dpi=300)
    plt.show()
    plt.close()
    print(f"[+] Learning curve saved to: {plot_path}")


import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_forecast(test_actuals_inv, test_preds_inv, date_index, output_dir, n_outputs=1):
    """
    Plot actual vs predicted cash flow on the test set.
    
    For Panel Data, predictions and actuals are grouped by date and summed 
    to visualize the global aggregated cash flow.

    For multi-step forecasting (n_outputs > 1), both the 1-step ahead and 
    the n-step ahead predictions are shown to illustrate forecast degradation.

    Args:
        test_actuals_inv (np.ndarray): Actual values on original scale.
        test_preds_inv (np.ndarray): Predicted values on original scale.
        date_index (np.ndarray or pd.DatetimeIndex): Exact dates corresponding to each sequence.
        output_dir (str): Directory where the plot will be saved.
        n_outputs (int): Forecast horizon (number of predicted steps). Defaults to 1.
    """
    # 1. Panel Data Aggregation
    df_actuals = pd.DataFrame(test_actuals_inv, index=date_index)
    df_preds = pd.DataFrame(test_preds_inv, index=date_index)

    # Group by index (date) and sum across all stores
    df_actuals_agg = df_actuals.groupby(level=0).sum()
    df_preds_agg = df_preds.groupby(level=0).sum()

    agg_dates = df_actuals_agg.index

    # 2. Plotting
    plt.figure(figsize=(14, 5))

    if n_outputs == 1:
        plt.plot(agg_dates, df_actuals_agg[0], 
                 label='Actual Cash Flow', linestyle='-', linewidth=1.5, color="#196194")
        plt.plot(agg_dates, df_preds_agg[0], 
                 label='Predicted Cash Flow', linestyle='--', linewidth=1.5, color="#8d1a1a")
    else:
        # Plot 1-step ahead
        plt.plot(agg_dates, df_actuals_agg[0], 
                 label='Actual (t+1)', linestyle='-', linewidth=1.5, color="#196194")
        plt.plot(agg_dates, df_preds_agg[0], 
                 label='Predicted (1-Day Ahead)', linestyle='--', linewidth=1.5, color="#8d1a1a")

        # Plot n-step ahead (shifted to align visually with the actual date it predicts)
        shifted_index = agg_dates + pd.Timedelta(days=n_outputs - 1)
        plt.plot(shifted_index, df_preds_agg[n_outputs - 1], 
                 label=f'Predicted ({n_outputs}-Days Ahead)', 
                 linestyle=':', linewidth=1.5, color="#d68b27", alpha=0.8)
        
        plt.plot(agg_dates, df_actuals_agg[n_outputs - 1], 
                 label=f'Actual (t+{n_outputs})', 
                 linestyle='-', linewidth=1, color="#2ca02c", alpha=0.6)

    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Cash Balance (€)', fontsize=12)
    plt.title(f'Real vs Predicted Cash Flow (N_OUTPUTS = {n_outputs})', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    plot_path = os.path.join(output_dir, 'real_vs_predicted.png')
    plt.savefig(plot_path, dpi=300)
    plt.show()
    plt.close()
    
    print(f"[+] Forecast plot saved to: {plot_path}")