# %% [markdown]
# # LSTM Recursive Multi-Step Forecasting
# 
# This script handles the full pipeline for recursive forecasting:
# * Loads the configuration pointing to the datasets with only "safe features".
# * Loads and scales the data.
# * Generates 3D tensors ONLY for the training set.
# * Executes Grid Search to find the best hyperparameters on the reduced feature space.
# * Trains the final Many-to-One LSTM model on the safe features.
# * Performs dynamic recursive inference on the test set.
# * Visualizes the aggregated global results.

# %% [markdown]
# ## Imports and Environment Setup
"""
Importing necessary libraries and custom modules.
"""
import os
import sys
import pandas as pd
import numpy as np
import torch

# Add project root to path
sys.path.append(os.path.abspath('..'))

from config.experiment_config import ExperimentConfig
from data.data_loader import load_data, scale_data, save_scalers
from features.sequence_builder import create_sequences_all_stores, prepare_tensors
from training.grid_search import run_grid_search
from training.final_train import final_training
from evaluation.evaluator import evaluate_recursive_panel
from visualization.plots import plot_learning_curve, plot_forecast
from utils.helpers import set_seed_and_device

print("All modules imported successfully.\n")

# %% [markdown]
# ## Experiment Configuration
"""
Setting up the configuration for the multi-step experiment.
We explicitly set the filenames to load the datasets containing only the safe features.
"""
config = ExperimentConfig()

# --- Task selection ---
config.n_outputs = 1                     # The architecture trains for 1-step ahead!
config.data_dir = '../../../Datasets/data_partitioned'
config.train_file = 'train_multi_step.csv' 
config.test_file = 'test_multi_step.csv'
config.validation_split = 0.01
config.output_dir = '../outputs/multi_step_small_network'
os.makedirs(config.output_dir, exist_ok=True)

# Ensure store_id is not scaled
if 'store_id' not in config.no_scale_cols:
    config.no_scale_cols.append('store_id')

# %% [markdown]
# ## Device and Reproducibility
"""
Setting the global seed and detecting CPU/GPU.
"""
device = set_seed_and_device(config.seed)
print(f"Output directory: {config.output_dir}")

# %% [markdown]
# ## Data Loading and Scaling
"""
Loading the 'safe' datasets and applying MinMax scaling.
"""
train_df, test_df = load_data(config)
train_scaled, test_scaled, feature_scaler, target_scaler = scale_data(train_df, test_df, config)

# Save scalers for later inference
save_scalers(feature_scaler, target_scaler, config.output_dir)

# %% [markdown]
# ## Sequence Generation (Training Set ONLY)
"""
We only generate the 3D tensors for the Training Set.
The Test Set will be processed dynamically during the recursive evaluation.
"""
X_train, y_train, train_dates = create_sequences_all_stores(
    train_scaled, config.target_col, config.n_lags, config.n_outputs, config.store_col
)

X_train_t, y_train_t = prepare_tensors(X_train, y_train, device)
print(f"N_FEATURES for safe dataset: {X_train_t.shape[2]}\n")


# %% [markdown]
# ## Hyperparameter Grid Search
"""
Performing hyperparameter search on the dataset containing only the Safe Features.
"""
skip_grid_search = True

if not skip_grid_search:
    best_params, best_metrics = run_grid_search(
        X_train_t, y_train_t, config, device, target_scaler, config.output_dir
    )
else:
    from utils.helpers import load_json
    best_params = load_json(f"{config.output_dir}/best_params.json")
    print(f"[+] Loaded existing best parameters: {best_params}")


# %% [markdown]
# ## Final Model Training
"""
Training the Many-to-One LSTM model using the safe features.
"""
skip_training = False

if not skip_training:
    final_model = final_training(
        X_train_t, y_train_t, best_params, config, device, config.output_dir
    )

# %% [markdown]
# ## Recursive Evaluation on the Test Set
"""
Loading the trained model and executing the recursive forecasting loop.
"""
load_pretrained = True

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
    print(f"[+] Model loaded from {model_path}")

# Run Recursive Evaluation
test_preds_inv, test_actuals_inv, test_dates = evaluate_recursive_panel(
    final_model, train_scaled, test_scaled, config, target_scaler, device
)

# %% [markdown]
# ## Visualizations
"""
Plotting learning curves and the aggregated continuous forecast.
"""
plot_learning_curve(config.output_dir)

# We pass n_outputs=1 because the recursive function generates a continuous line (flattened)
plot_forecast(test_actuals_inv, test_preds_inv, test_dates, config.output_dir, n_outputs=1)

# %%