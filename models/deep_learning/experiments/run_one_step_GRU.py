# %% [markdown]
# # LSTM Forecasting Experiment Driver
# 
# This notebook orchestrates the full forecasting pipeline:
# 1. Load configuration and set up the environment.
# 2. Load and preprocess data.
# 3. Generate sequences and tensors.
# 4. Run hyperparameter grid search (optional).
# 5. Train the final model.
# 6. Evaluate on the test set.
# 7. Visualize results.
# 
# Each step is modular and can be executed independently.

# %% [markdown]
# ## Imports and Environment Setup

import os
import sys
import pandas as pd
import numpy as np
import torch

# %load_ext autoreload
# %autoreload 2
# Add project root to path (adjust if notebook is in a subfolder)
sys.path.append(os.path.abspath('..'))

from config.experiment_config import ExperimentConfig
from data.data_loader import load_data, scale_data, save_scalers
from features.sequence_builder import create_sequences_all_stores, prepare_tensors
from training.grid_search import run_grid_search
from training.final_train import final_training
from evaluation.evaluator import evaluate_test_set
from visualization.plots import plot_learning_curve, plot_forecast
from utils.helpers import set_seed_and_device

print("All modules imported successfully.\n")

# %% [markdown]
# ## Experiment Configuration

config = ExperimentConfig()

# --- Task selection ---
config.n_outputs = 1
config.data_dir = '../../../Datasets/data_partitioned'
config.output_dir = '../outputs/one_step_GRU'
config.model_type = 'GRU'

# Adding the store ID as a column to consider
config.no_scale_cols.append('store_id')

os.makedirs(config.output_dir, exist_ok=True)

# %% [markdown]
# ## Device and Reproducibility

device = set_seed_and_device(config.seed)
print(f"Output directory: {config.output_dir}")
print(f"Forecast horizon: {config.n_outputs} step(s)\n")

# %% [markdown]
# ## Data Loading and Scaling

train_df, test_df = load_data(config)
train_scaled, test_scaled, feature_scaler, target_scaler = scale_data(train_df, test_df, config)

# Save scalers for later inference
save_scalers(feature_scaler, target_scaler, config.output_dir)

# inspect the first rows
# train_df.head()

# %% [markdown]
# ## Sequence Generation and Tensor Preparation

X_train, y_train, train_dates = create_sequences_all_stores(train_scaled, config.target_col, config.n_lags, config.n_outputs, config.store_col)
X_test, y_test, test_dates = create_sequences_all_stores(test_scaled, config.target_col, config.n_lags, config.n_outputs, config.store_col)

X_train_t, y_train_t = prepare_tensors(X_train, y_train, device)
X_test_t, y_test_t = prepare_tensors(X_test, y_test, device)

print(f"N_FEATURES: {X_train_t.shape[2]}\n")
# %% [markdown]
# ## Hyperparameter Grid Search (Optional)

# To skip grid search, set `skip_grid_search = True`
skip_grid_search = False

# config.param_grid = {'hidden_dim': [64], 'num_layers': [1], 'learning_rate': [0.001], 
#                      'dropout': [0], 'batch_size': [64], 'epochs': [5]}

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

# Note: If you already have a trained model and only want to evaluate,
#       set `skip_training = True` and jump to the evaluation section.

skip_training = False

if not skip_training:
    final_model = final_training(
        X_train_t, y_train_t, best_params, config, device, config.output_dir
    )

# %% [markdown]
# ## Evaluation on the Test Set

# If evaluating a previously saved model without re‑training:
load_pretrained = False   # Set to True to load from disk

if load_pretrained:
    from models.gru_model import CashFlowGRU
    from utils.helpers import load_json

    best_params = load_json(f"{config.output_dir}/best_params.json")
    final_model = CashFlowGRU(
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

# Run evaluation
test_preds_inv, test_actuals_inv = evaluate_test_set(
    final_model, X_test_t, y_test_t, target_scaler, config, config.output_dir
)

# %% [markdown]
# ## Visualizations

# Learning curve (requires that final training was run and loss CSV saved)
plot_learning_curve(config.output_dir)

# Forecast plot
#test_dates = test_df.index[config.n_lags : config.n_lags + len(test_preds_inv)]
plot_forecast(test_actuals_inv, test_preds_inv, test_dates, config.output_dir, config.n_outputs)
# %%
