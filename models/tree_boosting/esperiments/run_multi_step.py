# %% [markdown]
"""
# Iterative Multi-Step Forecasting Experiment with XGBoost

This script trains a XGBoost model for one-step forecasting but evaluates it using
a recursive (iterative) multi-step approach. Instead of using ground-truth lag values
for future days, it feeds its own predictions back into the feature space to forecast
subsequent days, simulating a real-world multi-step forecasting scenario.
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
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit

# Import modules from the package
from config.experiment_config import ExperimentConfig, DataConfig, FeatureConfig, ModelConfig, GridSearchConfig, TrainingConfig, OutputConfig
from data.data_loader import load_data
from features.feature_engineering import prepare_data
from training.grid_search import run_grid_search, grid_search_results_to_dataframe, train_best_model_from_grid, \
                            save_grid_search, load_grid_search
from training.recursive_forcast import recursive_forecast
from evaluation.evaluator import compute_metrics, save_predictions
from visualization.plots import plot_learning_curve, plot_forecast
from data.data_loader import save_best_params, load_best_params

%load_ext autoreload
%autoreload 2

# Visualizing current directory
print(f"Current directory: {Path.cwd()}")

# %% [markdown]
"""
## Experiment configuration

All parameters are centralized in an `ExperimentConfig` object.
Values replicate those used in the one-step notebook, but output is directed
to a dedicated multi-step folder.
"""
# Defining the experiment config
config = ExperimentConfig(
    # Paths and features config
    data = DataConfig(
        train_path = "../../../Datasets/data_partitioned/train_multi_step.csv", # Assuming you have a specific multi-step split or reuse the one-step one
        test_path = "../../../Datasets/data_partitioned/test_multi_step.csv",
        target_col = "cash_balance",
        group_col = "store_id",
        date_col = "date"
    ),
    # Sequence characteristic (Note: n_steps is still 1 for training)
    features = FeatureConfig(
        n_lags = 31,
        n_steps = 1, 
        time_features = [
            'day_sin', 'day_cos', 'weekday_sin', 'weekday_cos',
            'month_sin', 'month_cos', 'weekend', 'holiday', 'actual_holiday'
        ],
        detrend = False
    ),

    # Model configuration
    model = ModelConfig(
        model_type = "xgboost",
        n_estimators = 500,
        learning_rate = 0.01,
        max_depth = 5,
        random_state = 42,
        n_jobs = -1,
        eval_metric = "mae"
    ),

    # Grid search configuration
    grid_search = GridSearchConfig(
        param_grid = {
            'n_estimators': [100, 500],
            'learning_rate': [0.01, 0.1],
            'max_depth': [5, 10],
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

    # Training configuration
    training = TrainingConfig(
        early_stopping_rounds = 10,
        verbose = False,
        eval_set_fraction = 0.2
    ),

    # Output configuration
    output = OutputConfig(
        experiment_name = "multi_step_recursive",
        base_output_dir = "../outputs/multi_step_recursive",
        save_model = True,
        save_predictions = True,
        save_plots = True
    )
)

# Create output directory if it doesn't exist
output_dir = Path(config.output.base_output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

# %% [markdown]
"""
## Data loading

Load the preprocessed CSV files.
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

# %% [markdown]
"""
## Feature engineering

Create:
- Lag features of the target (n_lags=31)
- Target for forecasting (n_steps=1 for training)
- Shift time features by -1 to avoid lookahead
- Drop rows with NaN values
- Split into X and y
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

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

# X_train = X_train.drop(columns=['net_inflow'])
# X_test = X_test.drop(columns=['net_inflow'])
# %% [markdown]
"""
## Grid search with TimeSeriesSplit

Perform hyperparameter search over temporal folds.
"""
print("\nStarting grid search...")
estimator = XGBRegressor(
    random_state = config.model.random_state,
    n_jobs = config.model.n_jobs,
    verbosity = 1
)

grid_search = run_grid_search(
    estimator = estimator,
    X = X_train,
    y = y_train,
    param_grid = config.grid_search.param_grid,
    cv_splits = config.grid_search.cv_splits,
    scoring = config.grid_search.scoring,
    refit = config.grid_search.refit_metric,
    verbose = config.grid_search.verbose,
    n_jobs = config.model.n_jobs,
    fix_multioutput_grid = False
)

# Save grid search results as CSV
results_df = grid_search_results_to_dataframe(grid_search)
results_df.to_csv(output_dir / "grid_search_results.csv", index=False)
save_grid_search(grid_search, output_dir / "grid_search_object")
print(f"Grid search results saved to {output_dir / 'grid_search_results.csv'}")

# %% [markdown]
"""
## Final model training

Use the best hyperparameters found by grid search.
"""
print("\nTraining final model on full training set...")

grid_search = load_grid_search(output_dir / "grid_search_object.pkl")

# Temporal split: last 20% as validation
n_train = len(X_train)
val_size = int(n_train * config.training.eval_set_fraction)
X_train_final = X_train.iloc[:-val_size] if val_size > 0 else X_train
y_train_final = y_train.iloc[:-val_size] if val_size > 0 else y_train
X_val = X_train.iloc[-val_size:] if val_size > 0 else None
y_val = y_train.iloc[-val_size:] if val_size > 0 else None

# Train final model using the utility function
final_model = train_best_model_from_grid(
    grid_search = grid_search,
    X_train = X_train_final,
    y_train = y_train_final,
    X_val = X_val,
    y_val = y_val,
    eval_metric = config.model.eval_metric,
    verbose = config.training.verbose
)

print(f"Best parameters used: {grid_search.best_params_}")

# %% [markdown]
"""
## Test set evaluation (Recursive Inference)

Instead of using the standard `.predict()` which relies on ground-truth lagged values,
we use our `recursive_forecast` function. This simulates a real multi-step scenario
where future lags are populated by our own previous predictions.
"""

print("\nEvaluating on test set using iterative recursive forecasting...")

# Identify lag and non-lag columns dynamically based on the exact names created by prepare_data
target_col = config.data.target_col
n_lags = config.features.n_lags

lag_cols = [target_col] + [f'{target_col}_lag_{i}' for i in range(1, n_lags + 1)]
non_lag_cols = [col for col in X_test.columns if col not in lag_cols]

# Generate predictions recursively
y_pred_test = recursive_forecast(
    model=final_model,
    X_test=X_test,
    meta_test=meta_test,
    lag_cols=lag_cols,
    non_lag_cols=non_lag_cols,
    store_col=config.data.group_col
)

# Compute metrics
metrics = compute_metrics(y_test, y_pred_test)

print("\nTest metrics (Iterative Recursive):")
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")

# Save metrics to JSON
with open(output_dir / "test_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

# Save predictions
save_predictions(
    y_true = y_test,
    y_pred = y_pred_test,
    metadata = meta_test,
    filepath = output_dir / "test_predictions.csv",
    step_names = None
)

# %% [markdown]
"""
## Learning curve plot

Plot the loss (MAE) on the training and validation sets during final model training.
"""

evals_result = final_model.evals_result()
plot_learning_curve(
    evals_result, 
    metric = config.model.eval_metric,
    title = 'Learning Curve - Multi-Step Recursive XGBoost',
    save_path = output_dir / "learning_curve.png" if config.output.save_plots else None
)

# %% [markdown]
"""
## Aggregate forecast plot
Sum values across all stores for each date and compare actual vs recursively predicted series.
"""
test_df_with_pred = meta_test.copy()
test_df_with_pred['true'] = y_test.values
test_df_with_pred['pred'] = y_pred_test

aggregated = test_df_with_pred.groupby('date')[['true', 'pred']].sum()

plot_forecast(
    y_true=aggregated['true'],
    y_pred=aggregated['pred'],
    dates=aggregated.index,
    title='Recursive Multi-Step Forecasting: Real vs Predicted (Aggregated)',
    ylabel='Cash Balance',
    aggregate_by_date=False,
    save_path=output_dir / "forecast_plot.png" if config.output.save_plots else None
)

# %% [markdown]
"""
## Save the model

Serialize the final one-step model using joblib and save it to the output directory.
"""

if config.output.save_model:
    model_path = output_dir / "final_model.pkl"
    joblib.dump(final_model, model_path)
    print(f"Model saved to {model_path}")
# %%