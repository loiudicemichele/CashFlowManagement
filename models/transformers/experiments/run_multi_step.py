# %% [markdown] # Chronos‑2 Zero‑Shot Rolling One‑Step Forecast
"""
This script performs a rolling (recursive) multi‑step forecast using the Chronos‑2 model.

Workflow:
1. Import required modules and set up paths.
2. Load experiment configuration (data paths, model name, context length, etc.).
3. Set global random seed and hardware device.
4. Load aggregated daily cash balance data (training and test sets).
5. Load the pre‑trained Chronos‑2 model.
6. Perform rolling one‑step predictions using the model's built‑in method.
7. Evaluate predictions (MAE, RMSE, MAPE) and save results to CSV.
8. Visualise actual vs predicted cash flow.
"""

# %% [markdown] ## Imports and Environment Setup
"""
Import all necessary libraries and custom modules.
Add the project root to sys.path so that packages can be imported.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch

sys.path.append(os.path.abspath('..'))

from config.experiment_config import ChronosConfig
from utils.helpers import set_seed_and_device
from data.data_loader import load_data
from models.chronos_model import ChronosModel
from evaluation.evaluator import evaluate_and_save
from visualization.plots import plot_forecast

print("All modules imported successfully.\n")

# %% [markdown] ## Experiment Configuration
"""
Instantiate the configuration object and override paths and settings
for the aggregated one‑step dataset. The output directory is created.
"""

config = ChronosConfig()

# Override paths for the aggregated one‑step dataset
config.data_dir = '../../../Datasets/data_partitioned/aggregated'
config.train_file = 'train_one_step.csv'
config.test_file = 'test_one_step.csv'
config.output_dir = '../outputs/chronos_multi_step'

config.model = 'chronos-2-small'
config.model_name = os.path.join(config.model_dir, config.model)

# Forecasting settings
config.context_len = 512         # Number of past days to use as context
config.prediction_length = 1     # One‑step ahead
config.use_median = True         # Use median of forecast samples

# Create output directory
os.makedirs(config.output_dir, exist_ok=True)

# %% [markdown] ## Set Seed and Device
"""
Set the global random seed for reproducibility and detect the hardware device (CPU/CUDA).
"""

device = set_seed_and_device(config.seed, config.device)
print(f"Output directory: {config.output_dir}\n")

# %% [markdown] ## Load Data
"""
Load the training and test series using the data loader.
The function returns pandas Series with datetime index and cash_balance values.
"""

train_series, test_series = load_data(config)
print(f"Training period: {train_series.index.min()} → {train_series.index.max()}")
print(f"Test period:     {test_series.index.min()} → {test_series.index.max()}\n")
train_series.shape, test_series.shape

# %% [markdown] ## Load Chronos Model
"""
Instantiate the ChronosModel wrapper. This loads the pre‑trained pipeline
from Hugging Face (or a local checkpoint) and moves it to the appropriate device.
"""

model = ChronosModel(config)

# %% [markdown] ## Rolling One‑Step Forecast
"""
Perform iterative one‑step forecasting over the test period.
The model's built‑in `predict_rolling_one_step` method uses the last `context_len`
observations to predict the next day, then appends the prediction to the history.
"""

print(f"[+] Starting rolling forecast over {len(test_series)} days...")
print(f"[+] Using context length = {config.context_len} days")

# Use the convenience method inside ChronosModel
predictions = model.predict_rolling_one_step(
    history=train_series.values,
    test_actuals=test_series.values,
    context_len=config.context_len,
    multi_step = True
)

print("[+] Rolling forecast completed.\n")
# %% [markdown] ## Evaluation
"""
Compute MAE, RMSE, and MAPE on the test set and save predictions and metrics
to the output directory. The metrics are printed and also saved as JSON.
"""

metrics = evaluate_and_save(
    predictions=predictions.ravel(),
    actuals=test_series.values,
    date_index=test_series.index,
    config=config,
    output_dir=config.output_dir
)

# %% [markdown] ## Visualisation
"""
Generate a time series plot comparing actual vs predicted cash flow.
The plot is saved as 'real_vs_predicted.png' and also displayed.
"""

plot_forecast(
    actuals=test_series.values,
    predictions=predictions,
    date_index=test_series.index,
    output_dir=config.output_dir,
    title=f'Chronos‑2 Zero‑Shot Rolling Forecast (context={config.context_len})'
)

print("\n[+] Experiment completed successfully.")
# %%
