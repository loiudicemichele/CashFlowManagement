# %% [markdown]
# # LSTM Forecasting: Generalized Log-Differenced Cash Balance
# 
# This notebook orchestrates the forecasting pipeline using a generalized log-difference
# transformation on the target variable to enforce stationarity and remove seasonality:
# Load configuration and set up the environment.
# Define the differencing lag.
# Load data and apply log-difference grouped by store.
# Generate sequences and tensors.
# Extract the actual t-lag values for final inversion.
# Run hyperparameter grid search (optional).
# Train the final model.
# Evaluate on the test set (inverting the generalized log-diff transformation).
# Visualize results.

# %% [markdown]
# ## Imports and Environment Setup

import os
import sys
import pandas as pd
import numpy as np
import torch

# Add project root to path (adjust if notebook is in a subfolder)
sys.path.append(os.path.abspath('..'))

from config.experiment_config import ExperimentConfig
from data.data_loader import load_data, scale_data, save_scalers, apply_log_diff
from features.sequence_builder import create_sequences_all_stores, prepare_tensors
from training.grid_search import run_grid_search
from training.final_train import final_training
from evaluation.evaluator import evaluate_test_set_log_diff
from visualization.plots import plot_learning_curve, plot_forecast
from utils.helpers import set_seed_and_device

print("All modules imported successfully.\n")

# %% [markdown]
# ## Experiment Configuration

config = ExperimentConfig()

# --- Task selection ---
config.n_outputs = 1
config.data_dir = '../../../Datasets/data_partitioned'

# Set the lag for the difference
config.diff_lag = 31

config.output_dir = f'../outputs/one_step_log_diff_lag{config.diff_lag}'
os.makedirs(config.output_dir, exist_ok=True)

# config.no_scale_cols.append('original_target')
config.no_scale_cols.append('store_id')


# %% [markdown]
# ## Device and Reproducibility

device = set_seed_and_device(config.seed)
print(f"Output directory: {config.output_dir}")
print(f"Forecast horizon: {config.n_outputs} step(s)")
print(f"Differencing Lag: {config.diff_lag} step(s)\n")

# %% [markdown]
# ## Data Loading, Transformation, and Scaling

train_df, test_df = load_data(config)
raw_test_df = test_df.copy() # Saving raw data to recover the log-tranformation

# Apply Generalized Log-Diff transformation
print(f"[+] Applying Log-Diff transformation to the target (lag={config.diff_lag})...")
train_df = apply_log_diff(train_df, config.target_col, config.store_col, config.diff_lag)
test_df = apply_log_diff(test_df, config.target_col, config.store_col, config.diff_lag)

# Scale the data (the target scaler will now map generalized log-differences)
train_scaled, test_scaled, feature_scaler, target_scaler = scale_data(train_df, test_df, config)

# Save scalers for later inference
save_scalers(feature_scaler, target_scaler, config.output_dir)
# %% [markdown]
# ## Sequence Generation and Tensor Preparation

X_train, y_train, train_dates = create_sequences_all_stores(
    train_scaled, config.target_col, config.n_lags, config.n_outputs, config.store_col
)
X_test, y_test, test_dates = create_sequences_all_stores(
    test_scaled, config.target_col, config.n_lags, config.n_outputs, config.store_col
)

# Calculating the hystorical target
raw_test_df['y_lagged'] = raw_test_df.groupby(config.store_col)[config.target_col].shift(config.diff_lag)
raw_test_df = raw_test_df.dropna(subset=['y_lagged'])
# Calculating the prediction tensors (of the lagged target variable) that we expect as output. 
_, original_y_lagged, _ = create_sequences_all_stores(
    raw_test_df, 'y_lagged', config.n_lags, config.n_outputs, config.store_col
)


X_train_t, y_train_t = prepare_tensors(X_train, y_train, device)
X_test_t, y_test_t = prepare_tensors(X_test, y_test, device)

print(f"N_FEATURES: {X_train_t.shape[2]}\n")
# %% [markdown]
# ## Hyperparameter Grid Search (Optional)

skip_grid_search = False

if not skip_grid_search:
    best_params, best_metrics = run_grid_search(
        X_train_t, y_train_t, config, device, target_scaler, config.output_dir
    )
else:
    # Load previously saved best parameters
    from utils.helpers import load_json
    best_params = load_json(f"{config.output_dir}/best_params.json")
    print(f"[+] Loaded existing best parameters: {best_params}")

# %% [markdown]
# ## Final Model Training

skip_training = False

if not skip_training:
    final_model = final_training(
        X_train_t, y_train_t, best_params, config, device, config.output_dir
    )

# %% [markdown]
# ## Evaluation on the Test Set

load_pretrained = True   # Set to True to load from disk

if load_pretrained:
    from models.lstm_model import CashFlowLSTM
    from utils.helpers import load_json

    best_params = load_json(f"{config.output_dir}/best_params.json")
    final_model = CashFlowLSTM(
        input_dim = X_train_t.shape[2],
        hidden_dim = best_params['hidden_dim'],
        num_layers = best_params['num_layers'],
        output_dim = config.n_outputs,
        dropout = best_params['dropout']
    ).to(device)
    model_path = f"{config.output_dir}/best_lstm_model.pth"
    final_model.load_state_dict(torch.load(model_path, map_location=device))
    final_model.eval()
    print(f"[+] Model loaded from {model_path}")

# Run evaluation with Generalized Log-Diff inversion
test_preds_inv, test_actuals_inv = evaluate_test_set_log_diff(
    final_model, X_test_t, y_test_t, target_scaler, config, config.output_dir, original_y_lagged
)

# %% [markdown]
# ## Visualizations

# Learning curve
plot_learning_curve(config.output_dir)

# Forecast plot
plot_forecast(test_actuals_inv, test_preds_inv, test_dates, config.output_dir, config.n_outputs)
# %%