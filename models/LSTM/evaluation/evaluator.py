"""
Model Evaluation and Visualization

This module provides functions to evaluate a trained LSTM model on the test set,
compute standard regression metrics (MAE, RMSE, MAPE) on the original scale,
save predictions to CSV, and generate plots of learning curves and forecast
comparisons.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error


def evaluate_test_set(model, X_test_t, y_test_t, target_scaler, config, output_dir):
    """
    Run inference on the test set and compute final performance metrics.

    The model is set to evaluation mode. Predictions and ground truth values
    are collected, inverse-transformed to the original Euro scale, and saved
    to a CSV file. Global metrics (MAE, RMSE, MAPE) are computed over all
    predicted steps.

    Args:
        model (nn.Module): Trained LSTM model.
        X_test_t (torch.Tensor): Test sequences tensor.
        y_test_t (torch.Tensor): Test targets tensor.
        target_scaler: Fitted MinMaxScaler for the target variable.
        config: ExperimentConfig object with n_outputs and other settings.
        output_dir (str): Directory where the predictions CSV will be saved.

    Returns:
        tuple: (test_preds_inv, test_actuals_inv)
            - test_preds_inv (np.ndarray): Inverse-transformed predictions.
            - test_actuals_inv (np.ndarray): Inverse-transformed actual values.
    """
    model.eval()
    device = next(model.parameters()).device

    # Use a reasonable batch size for inference
    batch_size = config.param_grid['batch_size'][0] if 'batch_size' in config.param_grid else 128
    test_loader = DataLoader(
        TensorDataset(X_test_t, y_test_t),
        batch_size=batch_size,
        shuffle=False
    )

    test_preds = []
    test_actuals = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            preds = model(batch_X)
            test_preds.append(preds.cpu().numpy())
            test_actuals.append(batch_y.cpu().numpy())

    test_preds = np.vstack(test_preds)
    test_actuals = np.vstack(test_actuals)

    # Inverse transform to original Euro scale
    n_outputs = config.n_outputs
    test_preds_inv = target_scaler.inverse_transform(
        test_preds.reshape(-1, 1)
    ).reshape(-1, n_outputs)
    test_actuals_inv = target_scaler.inverse_transform(
        test_actuals.reshape(-1, 1)
    ).reshape(-1, n_outputs)

    # Compute global metrics (averaged over all predicted steps)
    mae = mean_absolute_error(test_actuals_inv, test_preds_inv)
    rmse = np.sqrt(mean_squared_error(test_actuals_inv, test_preds_inv))
    mape = mean_absolute_percentage_error(test_actuals_inv, test_preds_inv)

    print(f"\n[+] Final Test Set Results (N_OUTPUTS = {n_outputs})")
    print(f"    RMSE: {rmse:,.2f} €")
    print(f"    MAE:  {mae:,.2f} €")
    print(f"    MAPE: {mape:.4%}")

    # Save predictions to CSV
    if n_outputs == 1:
        pred_cols = ['Predicted']
        actual_cols = ['Actual']
    else:
        pred_cols = [f'Pred_Step_{i+1}' for i in range(n_outputs)]
        actual_cols = [f'Actual_Step_{i+1}' for i in range(n_outputs)]

    results_df = pd.DataFrame(test_actuals_inv, columns=actual_cols)
    for i, col in enumerate(pred_cols):
        results_df[col] = test_preds_inv[:, i]

    results_csv = os.path.join(output_dir, 'final_test_predictions.csv')
    results_df.to_csv(results_csv, index=False)
    print(f"[+] Final predictions saved to: {results_csv}")

    return test_preds_inv, test_actuals_inv