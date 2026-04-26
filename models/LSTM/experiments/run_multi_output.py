# %% [markdown]
# # LSTM Direct Multi-Step Forecasting (Vector Output)
# 
# This notebook uses the full pipeline for a Direct Multi-Step forecast:
# * Loads the configuration and sets the output horizon.
# * Loads and preprocesses the full panel data.
# * Generates 3D tensors where the target `y` contains multiple future steps.
# * Runs hyperparameter grid search on the vector-output architecture.
# * Trains the final model.
# * Evaluates the model globally across all predicted steps.
# * Visualizes the 1-step vs n-step forecast degradation.

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
from evaluation.evaluator import evaluate_test_set
from visualization.plots import plot_learning_curve, plot_forecast
from utils.helpers import set_seed_and_device

print("All modules imported successfully.\n")

# %% [markdown]
# ## Experiment Configuration
"""
Setting up the configuration for the Multi-Output experiment.
Here we change the prediction horizon to 14 days.
"""
config = ExperimentConfig()

# --- Task selection ---
config.n_outputs = 14  # The model will output an array of 14 future days directly
config.data_dir = '../../../Datasets/data_partitioned' # Adjust path as needed
config.output_dir = '../outputs/multi_output'

# Adding the store ID as a column to consider so it is not scaled
if 'store_id' not in config.no_scale_cols:
    config.no_scale_cols.append('store_id')

os.makedirs(config.output_dir, exist_ok=True)

# %% [markdown]
# ## Device and Reproducibility
"""
Setting the global seed and detecting CPU/GPU for deterministic execution.
"""
device = set_seed_and_device(config.seed)
print(f"Output directory: {config.output_dir}")
print(f"Forecast horizon: {config.n_outputs} step(s)\n")

# %% [markdown]
# ## Data Loading and Scaling
"""
Loading the training and test datasets and applying MinMax scaling.
Continuous features are scaled, while cyclical/categorical ones are bypassed.
"""
train_df, test_df = load_data(config)
train_scaled, test_scaled, feature_scaler, target_scaler = scale_data(train_df, test_df, config)

# Save scalers for later inference
save_scalers(feature_scaler, target_scaler, config.output_dir)

# %% [markdown]
# ## Sequence Generation and Tensor Preparation
"""
Generating sliding windows independently for each store.
The target array 'y' will now have shape (samples, 14).
"""
X_train, y_train, train_dates = create_sequences_all_stores(
    train_scaled, config.target_col, config.n_lags, config.n_outputs, config.store_col
)
X_test, y_test, test_dates = create_sequences_all_stores(
    test_scaled, config.target_col, config.n_lags, config.n_outputs, config.store_col
)

X_train_t, y_train_t = prepare_tensors(X_train, y_train, device)
X_test_t, y_test_t = prepare_tensors(X_test, y_test, device)

print(f"N_FEATURES: {X_train_t.shape[2]}\n")

# %% [markdown]
# ## Hyperparameter Grid Search
"""
Running Grid Search using TimeSeriesSplit to find the optimal architecture
for predicting 14 days at once.
"""
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
"""
Training the final model using the best hyperparameters found.
The model checkpointing logic will save the weights with the lowest validation loss.
"""
skip_training = False

if not skip_training:
    final_model = final_training(
        X_train_t, y_train_t, best_params, config, device, config.output_dir
    )

# %% [markdown]
# ## Evaluation on the Test Set
"""
Evaluating the vector-output model on the unseen test set.
Metrics (MAE, RMSE, MAPE) are computed globally over all 14 predicted steps.
"""
load_pretrained = True   # Set to True to load from disk if skipping training

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

# Run evaluation
test_preds_inv, test_actuals_inv = evaluate_test_set(
    final_model, X_test_t, y_test_t, target_scaler, config, config.output_dir
)

# %% [markdown]
# ## Visualizations
"""
Plotting the learning curve and the forecast. 
For n_outputs > 1, the plot will show the 1-step ahead and n-step ahead predictions.
"""
# Learning curve
plot_learning_curve(config.output_dir)

# Forecast plot
plot_forecast(test_actuals_inv, test_preds_inv, test_dates, config.output_dir, config.n_outputs)
# %%
