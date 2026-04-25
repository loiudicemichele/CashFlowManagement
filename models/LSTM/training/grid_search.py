"""
Grid Search with Time Series Cross-Validation

This module performs hyperparameter tuning using an expanding window
TimeSeriesSplit. For each parameter combination in the grid, it trains
and validates the LSTM model on multiple temporal folds, tracking
MAE, RMSE, and MAPE. The best configuration based on average RMSE
is saved along with its cross-validation metrics.
"""

import copy
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import TimeSeriesSplit, ParameterGrid

from models.lstm_model import CashFlowLSTM
from training.trainer import train_one_epoch, validate, compute_metrics


def run_grid_search(X_train_t, y_train_t, config, device, target_scaler, output_dir):
    """
    Perform Grid Search with expanding window cross-validation.

    For each hyperparameter combination in config.param_grid, the function
    runs TimeSeriesSplit cross-validation. The model is trained for a fixed
    number of epochs on each training fold and evaluated on the corresponding
    validation fold. Metrics are computed on the original scale (Euros).

    The combination achieving the lowest average RMSE is saved to
    `best_params.json` and its CV metrics to `best_cv_metrics.json`.

    Args:
        X_train_t (torch.Tensor): Training sequences tensor (full).
        y_train_t (torch.Tensor): Training targets tensor (full).
        config: ExperimentConfig object with CV and grid parameters.
        device (torch.device): Device to run training on.
        target_scaler: Fitted MinMaxScaler for inverse transforming predictions.
        output_dir (str): Directory where best parameters and metrics will be saved.

    Returns:
        tuple: (best_params, best_cv_metrics)
            - best_params (dict): Hyperparameters with lowest mean RMSE.
            - best_cv_metrics (dict): Dictionary of lists containing metrics per fold.
    """
    tscv = TimeSeriesSplit(n_splits=config.n_splits)
    grid = ParameterGrid(config.param_grid)

    best_rmse = float('inf')
    best_params = None
    best_cv_metrics = None

    print(f"[+] Starting Grid Search over {len(grid)} configurations...")
    print(f"[+] Time Series CV configured with {config.n_splits} splits.")

    # Cross Validation 
    for idx, params in enumerate(grid):
        print(f"\n--- Testing Config {idx + 1}/{len(grid)}: {params} ---")

        cv_metrics = {'MAE': [], 'RMSE': [], 'MAPE': []}

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train_t.cpu().numpy())):
            # Set seed for reproducibility within each fold
            torch.manual_seed(config.seed + fold)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(config.seed + fold)
                torch.cuda.manual_seed_all(config.seed + fold)

            print(f"  Fold {fold + 1}/{config.n_splits}")

            # Prepare data loaders for this fold
            train_loader = DataLoader(
                TensorDataset(X_train_t[train_idx], y_train_t[train_idx]),
                batch_size=params['batch_size'],
                shuffle=False
            )
            val_loader = DataLoader(
                TensorDataset(X_train_t[val_idx], y_train_t[val_idx]),
                batch_size=params['batch_size'],
                shuffle=False
            )

            # Initialize a fresh model for each fold
            model = CashFlowLSTM(
                input_dim=X_train_t.shape[2],
                hidden_dim=params['hidden_dim'],
                num_layers=params['num_layers'],
                output_dim=config.n_outputs,
                dropout=params['dropout']
            ).to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr = params['learning_rate'])
            criterion = torch.nn.MSELoss()

            # Training loop
            for epoch in range(params['epochs']):
                train_one_epoch(model, train_loader, optimizer, criterion, device)

            # Validation
            preds, actuals = validate(model, val_loader, criterion, device, return_Preds=True)

            # Inverse transform to original Euro scale
            preds_inv = target_scaler.inverse_transform(
                preds.reshape(-1, 1)
            ).reshape(-1, config.n_outputs)
            actuals_inv = target_scaler.inverse_transform(
                actuals.reshape(-1, 1)
            ).reshape(-1, config.n_outputs)

            # Compute fold metrics
            mae, rmse, mape = compute_metrics(actuals_inv, preds_inv, target_scaler)
            cv_metrics['MAE'].append(mae)
            cv_metrics['RMSE'].append(rmse)
            cv_metrics['MAPE'].append(mape)

            print(f"    Fold RMSE: {rmse:.2f} € | MAE: {mae:.2f} € | MAPE: {mape:.4%}")

        # Compute average metrics for this configuration
        mean_rmse = np.mean(cv_metrics['RMSE'])
        mean_mae = np.mean(cv_metrics['MAE'])
        mean_mape = np.mean(cv_metrics['MAPE'])
        std_rmse = np.std(cv_metrics['RMSE'])

        print(f"  -> Config Avg RMSE: {mean_rmse:,.2f} € ± {std_rmse:,.2f} | "
              f"Avg MAE: {mean_mae:,.2f} € | Avg MAPE: {mean_mape:.4%}")

        # Update best configuration if improved
        if mean_rmse < best_rmse:
            best_rmse = mean_rmse
            best_params = copy.deepcopy(params)
            best_cv_metrics = copy.deepcopy(cv_metrics)
            print(f"  *** New best configuration found! ***")

    # Save best configuration and metrics
    best_params_path = f"{output_dir}/best_params.json"
    best_metrics_path = f"{output_dir}/best_cv_metrics.json"

    with open(best_params_path, 'w') as f:
        json.dump(best_params, f, indent=4)
    with open(best_metrics_path, 'w') as f:
        json.dump(best_cv_metrics, f, indent=4)

    print("\n" + "=" * 50)
    print("GRID SEARCH COMPLETED")
    print(f"Best Parameters: {best_params}")
    print(f"Best Configuration Average RMSE: {best_rmse:,.2f} €")
    for metric_name, values in best_cv_metrics.items():
        mean_val = np.mean(values)
        std_val = np.std(values)
        if metric_name == 'MAPE':
            print(f"Best Config Mean {metric_name}: {mean_val:.4%} ± {std_val:.4%}")
        else:
            print(f"Best Config Mean {metric_name}: {mean_val:,.2f} € ± {std_val:,.2f} €")
    print("=" * 50)

    print(f"[+] Best parameters saved to {best_params_path}")
    print(f"[+] CV metrics saved to {best_metrics_path}")

    return best_params, best_cv_metrics