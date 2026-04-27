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

def evaluate_recursive_panel(model, train_scaled_df, test_scaled_df, config, target_scaler, device):
    """
    Perform recursive (autoregressive) forecasting over panel data (multiple stores).
    It initializes the sequence with the last `n_lags` of the training set and 
    predicts step-by-step. The model's own prediction is fed back into the sequence 
    along with the true future exogenous features (e.g., calendar features).
    """
    model.eval()
    
    # Identify the exact index of the target variable to overwrite it iteratively
    features_df = train_scaled_df.drop(columns=[config.store_col])
    target_idx = features_df.columns.get_loc(config.target_col)

    all_preds = []
    all_actuals = []
    all_dates = []
    
    print(f"[+] Starting recursive prediction for {len(test_scaled_df[config.store_col].unique())} stores...")

    with torch.no_grad():
        # Iterate over each single store to avoid mixing temporal sequences
        for store_id, test_group in test_scaled_df.groupby(config.store_col):
            test_group = test_group.sort_index()
            train_group = train_scaled_df[train_scaled_df[config.store_col] == store_id].sort_index()

            # Initial Window: take the last 'n_lags' days from the Training Set
            current_window = train_group.drop(columns=[config.store_col]).values[-config.n_lags:]
            # Convert to tensor (shape: 1, n_lags, n_features)
            current_window_t = torch.tensor(current_window, dtype=torch.float32).unsqueeze(0).to(device)

            test_values = test_group.drop(columns=[config.store_col]).values
            store_preds = []

            # Recursive Loop over the test set's time horizon
            for i in range(len(test_group)):
                # Predict step t+1
                pred = model(current_window_t)  # shape: (1, 1)
                pred_val = pred.item()
                store_preds.append(pred_val)

                # Build the feature vector for step t+1
                # Extract the true "safe features" (e.g., weekend, holiday) from the Test Set
                next_step = test_values[i].copy()
                
                # INJECTION: Replace the true target value with our PREDICTION
                next_step[target_idx] = pred_val

                # Update the window: drop the oldest day and append the new predicted step
                next_step_t = torch.tensor(next_step, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
                current_window_t = torch.cat((current_window_t[:, 1:, :], next_step_t), dim=1)

            # Collect the results
            all_preds.extend(store_preds)
            all_actuals.extend(test_values[:, target_idx]) # The actual scaled values for comparison
            all_dates.extend(test_group.index)

    # Inverse Transform to revert values back to original scale (Euros)
    all_preds_inv = target_scaler.inverse_transform(np.array(all_preds).reshape(-1, 1))
    all_actuals_inv = target_scaler.inverse_transform(np.array(all_actuals).reshape(-1, 1))

    # Globally sort the results chronologically
    results_df = pd.DataFrame({
        'Date': all_dates,
        'Actual': all_actuals_inv.flatten(),
        'Predicted': all_preds_inv.flatten()
    }).set_index('Date').sort_index()

    # Compute Metrics
    mae = mean_absolute_error(results_df['Actual'], results_df['Predicted'])
    rmse = np.sqrt(mean_squared_error(results_df['Actual'], results_df['Predicted']))
    mape = mean_absolute_percentage_error(results_df['Actual'], results_df['Predicted'])

    print(f"\n[+] Final Recursive Test Set Results")
    print(f"    RMSE: {rmse:,.2f} €")
    print(f"    MAE:  {mae:,.2f} €")
    print(f"    MAPE: {mape:.4%}")
    
    results_csv = os.path.join(config.output_dir, 'final_recursive_predictions.csv')
    results_df.to_csv(results_csv)
    print(f"[+] Recursive predictions saved to: {results_csv}")

    return results_df['Predicted'].values, results_df['Actual'].values, results_df.index

def evaluate_test_set_log_diff(model, X_test_t, y_test_t, target_scaler, config, output_dir, original_y_lagged):
    """
    Run inference on the test set and compute final performance metrics,
    inverting both the MinMax scaling and the generalized log-difference transformation.

    Args:
        model (nn.Module): Trained LSTM model.
        X_test_t (torch.Tensor): Test sequences tensor.
        y_test_t (torch.Tensor): Test targets tensor.
        target_scaler: Fitted MinMaxScaler for the transformed target variable.
        config: ExperimentConfig object with n_outputs and other settings.
        output_dir (str): Directory where the predictions CSV will be saved.
        original_y_lagged (np.ndarray): Array of the actual true values (in Euros) at time t-lag.

    Returns:
        tuple: (test_preds_inv, test_actuals_inv)
            - test_preds_inv (np.ndarray): Inverse-transformed predictions (Euros).
            - test_actuals_inv (np.ndarray): Inverse-transformed actual values (Euros).
    """
    model.eval()
    device = next(model.parameters()).device

    # Use a reasonable batch size for inference
    batch_size = config.param_grid['batch_size'][0] if 'batch_size' in config.param_grid else 128
    test_loader = DataLoader(
        TensorDataset ( X_test_t, y_test_t),
        batch_size = batch_size,
        shuffle = False
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

    # Inverse transform the MinMaxScaler (back to log-difference space)
    n_outputs = config.n_outputs
    test_preds_log_diff = target_scaler.inverse_transform(
        test_preds.reshape(-1, 1)
    ).reshape(-1, n_outputs)
    
    test_actuals_log_diff = target_scaler.inverse_transform(
        test_actuals.reshape(-1, 1)
    ).reshape(-1, n_outputs)

    # Inverse the generalized log-difference to get back to original Euro scale
    # Formula: log(y_t) - log(y_{t-lag}) = pred  =>  y_t = y_{t-lag} * exp(pred)
    test_preds_inv = original_y_lagged * np.exp(test_preds_log_diff)
    test_actuals_inv = original_y_lagged * np.exp(test_actuals_log_diff)

    # Compute global metrics
    mae = mean_absolute_error(test_actuals_inv, test_preds_inv)
    rmse = np.sqrt(mean_squared_error(test_actuals_inv, test_preds_inv))
    mape = mean_absolute_percentage_error(test_actuals_inv, test_preds_inv)

    print(f"\n[+] Final Test Set Results (Log-Diff Inverted)")
    print(f"    RMSE: {rmse:,.2f} €")
    print(f"    MAE:  {mae:,.2f} €")
    print(f"    MAPE: {mape:.4%}")

    # Save predictions to CSV
    import os
    import pandas as pd
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

def evaluate_recursive_panel_log_diff(model, train_scaled_df, test_scaled_df, config, target_scaler, device, raw_test_df):
    """
    Perform recursive forecasting over panel data with generalized log-difference inversion.
    The model feeds its own predicted log-returns back into the sequence.
    At the end, it uses the true historical Euro anchors to reconstruct the Euro trajectory.
    """
    model.eval()
    
    features_df = train_scaled_df.drop(columns=[config.store_col])
    target_idx = features_df.columns.get_loc(config.target_col)

    all_preds_euro = []
    all_actuals_euro = []
    all_dates = []
    
    print(f"[+] Starting recursive Log-Diff prediction for {len(test_scaled_df[config.store_col].unique())} stores...")

    with torch.no_grad():
        for store_id, test_group in test_scaled_df.groupby(config.store_col):
            test_group = test_group.sort_index()

            # Initial Window
            current_window = test_group.drop(columns=[config.store_col]).values[:config.n_lags]
            current_window_t = torch.tensor(current_window, dtype=torch.float32).unsqueeze(0).to(device)

            test_values = test_group.drop(columns=[config.store_col]).values
            store_preds_ld = []

            # Recursive Loop in Log-Diff space
            for i in range(config.n_lags, len(test_group)):
                pred = model(current_window_t)
                pred_val = pred.item()
                store_preds_ld.append(pred_val)

                next_step = test_values[i].copy()
                next_step[target_idx] = pred_val
                next_step_t = torch.tensor(next_step, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
                current_window_t = torch.cat((current_window_t[:, 1:, :], next_step_t), dim=1)

            # Inverse scaling
            preds_ld_unscaled = target_scaler.inverse_transform(np.array(store_preds_ld).reshape(-1, 1)).flatten()
            actuals_ld_unscaled = target_scaler.inverse_transform(test_values[config.n_lags:, target_idx].reshape(-1, 1)).flatten()

            # Fetch the true Euro history (anchors) for this store immediately preceding the test set
            start_pred_date = test_group.index[config.n_lags]
            
            store_raw = raw_test_df[(raw_test_df[config.store_col] == store_id)].sort_index()
            history_euros = store_raw[store_raw.index < start_pred_date][config.target_col].values[-config.diff_lag:]
            
            # Buffers to hold the chain of values
            pred_buffer = list(history_euros)
            actual_buffer = list(history_euros)

            # Step-by-step exponential inversion
            for i in range(len(preds_ld_unscaled)):
                # y_t = y_{t-lag} * exp(log_diff)
                curr_actual = actual_buffer[i] * np.exp(actuals_ld_unscaled[i])
                all_actuals_euro.append(curr_actual)
                actual_buffer.append(curr_actual) # The true series is reconstructed using the true chain

                curr_pred = pred_buffer[i] * np.exp(preds_ld_unscaled[i])
                all_preds_euro.append(curr_pred)
                pred_buffer.append(curr_pred) # The predicted series uses its own past predictions

            all_dates.extend(test_group.index[config.n_lags:])

    # Globally sort the results chronologically
    results_df = pd.DataFrame({
        'Date': all_dates,
        'Actual': all_actuals_euro,
        'Predicted': all_preds_euro
    }).set_index('Date').sort_index()

    mae = mean_absolute_error(results_df['Actual'], results_df['Predicted'])
    rmse = np.sqrt(mean_squared_error(results_df['Actual'], results_df['Predicted']))
    mape = mean_absolute_percentage_error(results_df['Actual'], results_df['Predicted'])

    print(f"\n[+] Final Recursive Test Set Results (Log-Diff Inverted)")
    print(f"    RMSE: {rmse:,.2f} €")
    print(f"    MAE:  {mae:,.2f} €")
    print(f"    MAPE: {mape:.4%}")
    
    import os
    results_csv = os.path.join(config.output_dir, 'final_recursive_predictions.csv')
    results_df.to_csv(results_csv)
    print(f"[+] Recursive predictions saved to: {results_csv}")

    return results_df['Predicted'].values, results_df['Actual'].values, results_df.index