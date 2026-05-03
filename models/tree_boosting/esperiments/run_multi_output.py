# %% [markdown]
"""
# Multi-Output Forecasting Experiment with XGBoost

This script trains a direct multi-step model using the Multi-Output approach.
Instead of feeding predictions back recursively, we map the current state 
directly to `n_steps` independent future targets (e.g., day 1, day 2... day 14).
We perform a Grid Search using `MultiOutputRegressor`, and then explicitly train 
separate models for each step to enable early stopping and extract learning curves.
"""

# %% [markdown]
"""
## Initial setup and imports
"""
import sys
from pathlib import Path

# Add project root to path to allow absolute imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import json
import joblib
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import TimeSeriesSplit

# Import modules from the package
from config.experiment_config import ExperimentConfig, DataConfig, FeatureConfig, ModelConfig, GridSearchConfig, TrainingConfig, OutputConfig
from data.data_loader import load_data
from features.feature_engineering import prepare_data
from training.grid_search import run_grid_search, grid_search_results_to_dataframe, save_grid_search, load_grid_search
from evaluation.evaluator import compute_metrics, save_predictions
from visualization.plots import plot_learning_curve, plot_multi_step_forecast

%load_ext autoreload
%autoreload 2

# Visualizing current directory
print(f"Current directory: {Path.cwd()}")

# %% [markdown]
"""
## Experiment configuration

Configuring the experiment for multi-step forecasting.
n_steps is set to 14 (a 2-week horizon), and n_lags to 30.
"""
# Defining the experiment config
config = ExperimentConfig(
    data = DataConfig(
        train_path = "../../../Datasets/data_partitioned/train_one_step.csv",
        test_path = "../../../Datasets/data_partitioned/test_one_step.csv",
        target_col = "cash_balance",
        group_col = "store_id",
        date_col = "date"
    ),
    features = FeatureConfig(
        n_lags = 30,
        n_steps = 31, # Horizon of 14 days = 14 XGBoost trained and 14 days prediction in the future
        time_features = [
            'day_sin', 'day_cos', 'weekday_sin', 'weekday_cos',
            'month_sin', 'month_cos', 'weekend', 'holiday', 'actual_holiday'
        ],
        detrend = False
    ),
    model = ModelConfig(
        model_type = "xgboost",
        n_estimators = 500,
        learning_rate = 0.01,
        max_depth = 5,
        random_state = 42,
        n_jobs = -1,
        eval_metric = "mae"
    ),
    grid_search = GridSearchConfig(
        param_grid = {
            # fix_multioutput_grid=True in run_grid_search will automatically prefix these
            # 'n_estimators': [100, 500],
            # 'learning_rate': [0.01, 0.1],
            # 'max_depth': [5, 10],
            'n_estimators': [100],
            'learning_rate': [0.1],
            'max_depth': [5],
        },
        cv_splits = 5,
        scoring = {
            'MAE': 'neg_mean_absolute_error',
            'MSE': 'neg_mean_squared_error',
            'MAPE': 'neg_mean_absolute_percentage_error'
        },
        refit_metric = 'MAE',
        verbose = 1
    ),
    training = TrainingConfig(
        early_stopping_rounds = 10,
        verbose = False,
        eval_set_fraction = 0.2
    ),
    output = OutputConfig(
        experiment_name = "multi_output",
        base_output_dir = "../outputs/multi_output",
        save_model = True,
        save_predictions = True,
        save_plots = True
    )
)

output_dir = Path(config.output.base_output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

# %% [markdown]
"""
## Data loading
"""
print("Loading data...")
df_train, df_test = load_data(
    config.data.train_path,
    config.data.test_path,
    index_col='date',
    parse_dates=True
)
print(f"Train shape: {df_train.shape}")
print(f"Test shape: {df_test.shape}")
df_train.columns
# %% [markdown]
"""
## Feature engineering
"""
print("Preparing features...")
X_train, y_train, meta_train = prepare_data(
    df = df_train,
    target_col = config.data.target_col,
    n_lags = config.features.n_lags,
    n_steps = config.features.n_steps,
    time_features = config.features.time_features,
    group_col = config.data.group_col,
    detrend = config.features.detrend,
    detrend_period = config.features.detrend_period
)

X_test, y_test, meta_test = prepare_data(
    df=df_test,
    target_col = config.data.target_col,
    n_lags = config.features.n_lags,
    n_steps = config.features.n_steps,
    time_features = config.features.time_features,
    group_col = config.data.group_col,
    detrend = config.features.detrend,
    detrend_period = config.features.detrend_period
)

# ---> PREVENT DATA LEAKAGE <---
# We drop 'net_inflow' because it is an endogenous variable not known in advance
if 'net_inflow' in X_train.columns:
    X_train = X_train.drop(columns=['net_inflow'])
    X_test = X_test.drop(columns=['net_inflow'])
    print("Dropped 'net_inflow' from features to prevent data leakage.")

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
# %%
# X_train.iloc[:100:10, -33:-20]
# y_train.iloc[:100:10]
# %% [markdown]
"""
## Grid search with TimeSeriesSplit

Using MultiOutputRegressor to find the best hyperparameters.
The `run_grid_search` function will automatically add the `estimator__` prefix.
"""
print("\nStarting Multi-Output Grid Search...")

base_estimator = XGBRegressor(
    random_state = config.model.random_state,
    n_jobs = config.model.n_jobs,
    verbosity = 1
)
multi_output_model = MultiOutputRegressor(base_estimator, n_jobs=-1)

grid_search = run_grid_search(
    estimator = multi_output_model,
    X = X_train,
    y = y_train,
    param_grid = config.grid_search.param_grid,
    cv_splits = config.grid_search.cv_splits,
    scoring = config.grid_search.scoring,
    refit = config.grid_search.refit_metric,
    verbose = config.grid_search.verbose,
    n_jobs = config.model.n_jobs,
    fix_multioutput_grid = True 
)

results_df = grid_search_results_to_dataframe(grid_search)
results_df.to_csv(output_dir / "grid_search_results.csv", index=False)
save_grid_search(grid_search, output_dir / "grid_search_object")

# Clean the 'estimator__' prefix from best params to use them directly in single models
best_params_cleaned = {k.replace('estimator__', ''): v for k, v in grid_search.best_params_.items()}
print(f"Cleaned Best parameters: {best_params_cleaned}")

# %% [markdown]
"""
## Final model training

We explicitly iterate over the target steps to train an independent XGBRegressor for each.
This allows us to track validation losses (eval_set) for plotting learning curves,
which MultiOutputRegressor natively hides.
"""
print(f"\nTraining {config.features.n_steps} separate models for the final evaluation...")

# Temporal split: last 20% as validation
n_train = len(X_train)
val_size = int(n_train * config.training.eval_set_fraction)
X_train_final = X_train.iloc[:-val_size] if val_size > 0 else X_train
y_train_final = y_train.iloc[:-val_size] if val_size > 0 else y_train
X_val = X_train.iloc[-val_size:] if val_size > 0 else None
y_val = y_train.iloc[-val_size:] if val_size > 0 else None

models = []
evals_results = []

for i in range(config.features.n_steps):
    step_name = y_train.columns[i]
    print(f"Training model for step: {step_name} ({i+1}/{config.features.n_steps})...")
    
    step_model = XGBRegressor(
        **best_params_cleaned,
        random_state = config.model.random_state,
        n_jobs = config.model.n_jobs,
        eval_metric = config.model.eval_metric
    )
    
    if X_val is not None:
        eval_set = [(X_train_final, y_train_final.iloc[:, i]), (X_val, y_val.iloc[:, i])]
        step_model.fit(X_train_final, y_train_final.iloc[:, i], eval_set=eval_set, verbose=False)
    else:
        step_model.fit(X_train_final, y_train_final.iloc[:, i], verbose=False)
        
    models.append(step_model)
    if X_val is not None:
        evals_results.append(step_model.evals_result())

# %% [markdown]
"""
## Test set evaluation

Predict values by aggregating predictions from all individual models.
Compute per-step metrics and overall averages.
"""
print("\nEvaluating on test set...")
y_pred_test_list = [model.predict(X_test) for model in models]
y_pred_test = np.column_stack(y_pred_test_list)

# Get detailed metrics per step
metrics_per_step = compute_metrics(y_test, y_pred_test, multioutput='raw_values')
# Get average metrics across all steps
metrics_avg = compute_metrics(y_test, y_pred_test, multioutput='uniform_average')

print("\nAverage Test metrics across all steps:")
for metric, value in metrics_avg.items():
    print(f"{metric}: {value:.4f}")

# Save detailed metrics to JSON
with open(output_dir / "test_metrics_detailed.json", "w") as f:
    json.dump(metrics_per_step, f, indent=4)

# Save predictions
save_predictions(
    y_true = y_test,
    y_pred = y_pred_test,
    metadata = meta_test,
    filepath = output_dir / "test_predictions.csv",
    step_names = y_test.columns.tolist()
)

# %% [markdown]
"""
## Learning curve plot

Average the training and validation loss curves across all independent step models
to provide a global view of model convergence.
"""
if evals_results and config.output.save_plots:
    # Averaging the error for each model at each boosting round
    avg_train_loss = np.mean([res['validation_0'][config.model.eval_metric] for res in evals_results], axis=0)
    avg_val_loss = np.mean([res['validation_1'][config.model.eval_metric] for res in evals_results], axis=0)
    
    # Mocking the evals_result dictionary structure expected by our plot_learning_curve function
    mock_evals_result = {
        'validation_0': {config.model.eval_metric: avg_train_loss.tolist()},
        'validation_1': {config.model.eval_metric: avg_val_loss.tolist()}
    }
    
    plot_learning_curve(
        mock_evals_result, 
        metric = config.model.eval_metric,
        title = f'Averaged Learning Curve (Multi-Output {config.features.n_steps} steps)',
        save_path = output_dir / "learning_curve.png"
    )

# %% [markdown]
"""
## Aggregate forecast plot

For multi-step, plotting everything continuously is noisy. Instead, we plot the true
aggregated cash balance and sequentially plot the n-day forecasts over it to visualize
how well each prediction trajectory matches reality.
"""
if config.output.save_plots:
    plot_multi_step_forecast(
        y_true = y_test,
        y_pred = y_pred_test,
        metadata = meta_test,
        n_steps = config.features.n_steps,
        title = f'Real Vs Predicted Cash Flow (Multi-Output Horizon: {config.features.n_steps} days)',
        save_path = output_dir / "forecast_plot.png"
    )

# %% [markdown]
"""
## Save the model

Serialize the list of models. We can wrap them back into a MultiOutputRegressor
for convenience or just save the list.
"""
if config.output.save_model:
    model_path = output_dir / "final_model_list.pkl"
    joblib.dump(models, model_path)
    print(f"Model list saved to {model_path}")
# %%