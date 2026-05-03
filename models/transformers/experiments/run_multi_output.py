# %% [markdown] # Chronos-2 Zero-Shot Multi-Output Forecast
"""
This experiment uses Chronos-2 to predict the entire test sequence in a
single forward pass (multi-output forecasting).

Unlike the rolling one-step experiment, the model is called exactly once:
the last `context_len` observations of the training set are used as context,
and the model generates all `len(test_series)` future steps simultaneously.

This approach tests the upper bound of Chronos-2's direct generation
capability and produces zero error accumulation across the forecast horizon,
since there is no recursive feedback loop.

Workflow:
1. Import modules and set up environment.
2. Configure the experiment.
3. Set random seed and device.
4. Load aggregated cash balance data.
5. Load Chronos-2 model.
6. Build the single context window from the tail of the training set.
7. Run a single forward pass for the full test horizon.
8. Evaluate predictions (MAE, RMSE, MAPE) and save results.
9. Visualise actual vs predicted and the full series with history.
"""

# %% [markdown] ## Imports and Environment Setup
"""
Import all necessary libraries and add the project root to sys.path so that
all custom packages (config, data, models, evaluation, visualization) are
reachable regardless of the working directory.
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
from visualization.plots import plot_forecast, plot_rolling_comparison

print("All modules imported successfully.\n")

# %% [markdown] ## Experiment Configuration
"""
Instantiate ChronosConfig and override the settings specific to the
multi-output experiment.

Key difference from one-step: prediction_length is set to the full
length of the test set at runtime (after loading data), so the model
produces all future steps in one call.

context_len controls how many tail observations of the training set
are fed as context.  A longer context generally helps for longer horizons,
but is bounded by the model's maximum sequence length (~2048 patches).
"""

config = ChronosConfig()

config.data_dir    = '../../../Datasets/data_partitioned/aggregated'
config.train_file  = 'train_one_step.csv'
config.test_file   = 'test_one_step.csv'
config.output_dir  = '../outputs/chronos_multi_output'
config.model       = 'chronos-2-small'
config.model_name  = os.path.join(config.model_dir, config.model)

# Context fed to the model: last N days of training history.
# For multi-output the longer the better, up to the model's patch limit.
config.context_len      = 512
config.use_median       = True

# prediction_length will be overridden below once the test set is loaded.
config.prediction_length = None

os.makedirs(config.output_dir, exist_ok=True)

# %% [markdown] ## Set Seed and Device
"""
Fix all random seeds (Python, NumPy, PyTorch) and select the hardware
device.  Deterministic mode is enabled on CUDA for reproducibility.
"""

device = set_seed_and_device(config.seed, config.device)
print(f"Output directory : {config.output_dir}")
print(f"Context length   : {config.context_len} days\n")

# %% [markdown] ## Load Data
"""
Load the aggregated daily cash balance series.
The training set provides the historical context window;
the test set provides ground-truth values for evaluation.
"""

train_series, test_series = load_data(config)

# Set prediction_length to the full test horizon now that the data is loaded.
config.prediction_length = len(test_series)

print(f"Training period  : {train_series.index.min()} → {train_series.index.max()}")
print(f"Test period      : {test_series.index.min()} → {test_series.index.max()}")
print(f"Prediction length set to {config.prediction_length} steps (full test horizon)\n")

# %% [markdown] ## Load Chronos-2 Model
"""
Instantiate ChronosModel, which loads the Chronos-2 pipeline from the
local checkpoint directory and moves it to the selected device.
The wrapper handles the Chronos2Pipeline / ChronosPipeline distinction
internally.
"""

model = ChronosModel(config)

# %% [markdown] ## Build Context Window
"""
Extract the last `context_len` observations from the training set.
This single window is the only input the model receives — no test
values are ever revealed during inference.

If the training set is shorter than context_len, the entire training
series is used as context (no padding required; Chronos-2 handles
variable-length inputs natively).
"""

history = train_series.values

# Guard against a training set shorter than context_len.
actual_context_len = min(config.context_len, len(history))
context_window = history[-actual_context_len:]

print(f"Context window   : last {actual_context_len} days of training set")
print(f"Context range    : {train_series.index[-actual_context_len]} → {train_series.index[-1]}")
print(f"Forecast horizon : {config.prediction_length} days\n")

# %% [markdown] ## Single Forward Pass — Full Horizon Prediction
"""
Call model.predict() exactly once with prediction_length equal to the
entire test horizon.  Chronos-2 generates all future steps in a single
encoder forward pass — no recursive feedback, no error accumulation.

The returned array has shape (prediction_length,), one point estimate
per future step.
"""

print(f"[+] Running single forward pass for {config.prediction_length} steps...")

predictions = model.predict(
    context=context_window,
    prediction_length=config.prediction_length
)

print(f"[+] Prediction complete. Output shape: {predictions.shape}\n")

# %% [markdown] ## Evaluation
"""
Compute MAE, RMSE, and MAPE on the full test horizon and persist both the
per-step predictions CSV and the metrics JSON to the output directory.
"""

metrics = evaluate_and_save(
    predictions=predictions,
    actuals=test_series.values,
    date_index=test_series.index,
    config=config,
    output_dir=config.output_dir
)

# %% [markdown] ## Visualisation
"""
Two complementary plots are generated:

  (a) Test period only — actual vs predicted cash balance, matching the
      style of the one-step experiment for direct visual comparison.

  (b) Full series — historical training data (faded) followed by the test
      period with actual and predicted values, showing how the forecast
      connects to the observed past.
"""

# Plot actual vs predicted.
plot_forecast(
    actuals=test_series.values,
    predictions=predictions,
    date_index=test_series.index,
    output_dir=config.output_dir,
    title=(
        f'Chronos-2 Multi-Output Forecast — full horizon '
        f'({config.prediction_length} days, single forward pass)'
    )
)

print("\n[+] Multi-output experiment completed successfully.")