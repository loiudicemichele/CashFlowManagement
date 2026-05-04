# %% [markdown] # Chronos-2 Zero-Shot Panel Forecast with Covariates
"""
This notebook runs a multi-step forecasting experiment using the Chronos-2 model
on panel data (multiple time series evaluated simultaneously).

Methodological Note:
We utilize the native `predict_df` API. This allows the model to leverage 
In-Context Learning (ICL) and cross-series attention. To strictly prevent 
Data Leakage, we split our covariates into 'stochastic' (unknown in the future) 
and 'deterministic' (known in the future). 
Only deterministic covariates are passed to the model's future horizon.

Workflow:
    1. Import modules and set environment.
    2. Configure the experiment (covariates, panel ID, horizons).
    3. Load panel DataFrames (Long format).
    4. Isolate deterministic covariates to build the `future_df`.
    5. Generate predictions for all stores simultaneously.
    6. Merge predictions with ground truth for evaluation.
"""

# %% [markdown] ## Imports and Environment Setup
import os
import sys
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath('..'))

from config.experiment_config import ChronosConfig
from utils.helpers import set_seed_and_device
from data.data_loader import load_covariates_df
from models.chronos_model import ChronosModel
from evaluation.evaluator import evaluate_and_save
from visualization.plots import plot_forecast, plot_forecast_with_uncertainty

print("All modules imported successfully.\n")

# %% [markdown] ## Experiment Configuration
"""
Configure the panel experiment. 
Ensure `item_id_col` correctly points to your categorical store identifier.
"""
config = ChronosConfig()

config.data_dir    = '../../../Datasets/data_partitioned/'
config.output_dir  = '../outputs/chronos_panel_covariates'
config.model       = 'chronos-2-cashflow-finetuned'
config.model_name  = os.path.join(config.model_dir, config.model)
# Explicitly define the panel ID column
config.id_col = 'store_id'
config.apply_log_traansf = True
config.context_len = 512
os.makedirs(config.output_dir, exist_ok=True)

# %% [markdown] ## Set Seed and Load Data
"""
Load the Long-format DataFrames for training (context) and testing (ground truth).
The data loader ensures proper sorting by (store_id, date).
"""
device = set_seed_and_device(config.seed, config.device)

train_df, test_df = load_covariates_df(config)

# Log-transformation
if config.apply_log_traansf:
    for split_name, df in [("train", train_df), ("test", test_df)]:
            df[config.target_col] = np.sign(df[config.target_col]) * np.log1p(np.abs(df[config.target_col]))

# Calculate prediction length dynamically based on the unique dates in test set
config.prediction_length = test_df['date'].nunique()


print(f"[+] Prediction horizon set to: {config.prediction_length} steps.\n")

# %% [markdown] ## Load Chronos-2 Model
model = ChronosModel(config)
if not model.is_chronos2:
    raise RuntimeError("Panel covariate experiment strictly requires a Chronos-2 model.")

# %% [markdown] ## Prepare Future Context (Preventing Data Leakage)
"""
Extract ONLY the known future variables for the prediction horizon.
We strip away the target and any stochastic covariates to simulate a true
production environment.
"""
future_cols = [config.id_col, 'date'] + config.deterministic_covariates

# Testing a lower config.prediction_length
first_pred_len_dates = test_df['date'].drop_duplicates().sort_values().head(config.prediction_length)
test_df_sliced = test_df[test_df['date'].isin(first_pred_len_dates)].copy()

future_df = test_df_sliced[future_cols].copy()

print(f"[+] future_df built with deterministic columns: {config.deterministic_covariates}")
print(f"[+] future_df shape: {future_df.shape}\n")
# train_df.columns
# %% [markdown] ## Generate Panel Forecast
"""
Execute the forward pass. The model processes all stores in parallel.
"""
print("[+] Running multi-step panel forecast via API...")

# forecast_df usually contains item_id, date, and forecast columns (e.g., median/quantiles)
forecast_df = model.predict_panel_covariates(
    context_df=train_df,
    prediction_length=config.prediction_length,
    future_df=future_df
)
if config.apply_log_traansf:
    for col in forecast_df.columns:
        if col not in [config.id_col, 'date', 'target_name']:
            forecast_df[col] = model.inverse_symlog(forecast_df[col])
    test_df_sliced[config.target_col] = model.inverse_symlog(test_df_sliced[config.target_col])

print(f"[+] Forecast complete. Output shape: {forecast_df.shape}\n")
# forecast_df.columns
# %% [markdown] ## Evaluation and Visualization
"""
Merge the predictions with the ground-truth test_df to compute metrics.
We compute a global metric across all stores, and visualize a single sample store.
"""
# Assuming the API returns the target prediction in a column named after the target,
# or under 'median'/'mean' depending on the exact chronos version mapping.
# For standard GluonTS PandasDataset, it often defaults to 'median' or the target_col name.
pred_col = 'predictions'

col_median = '0.5'
col_lower = '0.1'
col_upper = '0.9'

# Merge based on panel ID and date
results_df = pd.merge(
    test_df_sliced[[config.id_col, 'date', config.target_col]], 
    forecast_df[[config.id_col, 'date', pred_col, col_median, col_lower, col_upper]],
    on=[config.id_col, 'date'],
    how='inner'
)

# Compute Global Metrics (flattened across all stores)
print("=" * 50)
print("GLOBAL PANEL METRICS")
print("=" * 50)
metrics = evaluate_and_save(
    predictions=results_df[pred_col].values,
    actuals=results_df[config.target_col].values,
    date_index=results_df['date'],  # Notice: This index will have overlapping dates due to panel structure
    config=config,
    output_dir=config.output_dir
)

# Plot a single sample store to keep the visual clean
sample_store = results_df[config.id_col].unique()[0]
sample_data = results_df[results_df[config.id_col] == sample_store]

agg_results = results_df.groupby('date')[[config.target_col, pred_col]].sum().reset_index()

plot_forecast_with_uncertainty(
    actuals=sample_data[config.target_col].values,
    pred_median=sample_data[col_median].values,
    pred_lower=sample_data[col_lower].values,
    pred_upper=sample_data[col_upper].values,
    date_index=sample_data['date'],
    output_dir=config.output_dir,
    title=f'Chronos-2 Forecast (with Uncertainty) - Sample: {sample_store}'
)

print("\n[+] Panel covariate experiment completed successfully.")
# %%
