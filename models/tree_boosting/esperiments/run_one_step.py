# %% [markdown]
"""
# One-Step Forecasting Experiment with XGBoost

This script replicates trains a XGBoost model for one-step forecasting using the package built.
It loads preprocessed data, creates features (lags, shifted time features), performs grid search with TimeSeriesSplit,
trains the final model with early stopping, evaluates on the test set, and saves results (metrics, predictions, plots, model).
"""

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
from evaluation.evaluator import compute_metrics, save_predictions, inverse_detrend_predictions
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
Values replicate those used in the original notebook.
"""
# Defining the experiment config
config = ExperimentConfig(
    # Paths and features config
    data = DataConfig(
        train_path = "../../../Datasets/data_partitioned/train_one_step.csv",
        test_path = "../../../Datasets/data_partitioned/test_one_step.csv",
        target_col = "cash_balance",
        group_col = "store_id",
        date_col = "date"
    ),
    # Sequence characteristic
    features = FeatureConfig(
        n_lags = 31,
        n_steps = 1,
        time_features = [
            'day_sin', 'day_cos', 'weekday_sin', 'weekday_cos',
            'month_sin', 'month_cos', 'weekend', 'holiday', 'actual_holiday'
        ],
        detrend = True,
        detrend_period = 31
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
        experiment_name = "one_step",
        base_output_dir = "../outputs/one_step",
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

Load the preprocessed CSV files for the one-step experiment.
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

# Visualizing dataset
# df_train.columns
# %% [markdown]
"""
## Feature engineering

Create:
- Lag features of the target (n_lags=31)
- Target for forecasting (n_steps=1)
- Shift time features by -1 to avoid lookahead
- Drop rows with NaN values
- Split into X and y
"""

print("Preparing features...")
# Preparing data for one-time-step forcasting problem:
"""
Target Col : 'cash_balance'
Group Col : 'store_id'

Time Features : 'day_sin', 'day_cos', 'weekday_sin', 'weekday_cos',
        'month_sin', 'month_cos', 'weekend', 'holiday', 'actual_holiday'

Detrended = False
detrend_period = 7 (Not activated)
"""
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

# X_train.iloc[:100:10, -33:-20]
# y_train.iloc[:100:10]
# %% [markdown]
"""
## Grid search with TimeSeriesSplit

Perform hyperparameter search over 5 temporal folds.
MAE is the reference metric for model selection.
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
For early stopping, the training set is split into train/validation (80%/20%) respecting temporal order.
"""
print("\nTraining final model on full training set...")

grid_search = load_grid_search(output_dir / "grid_search_object.pkl")
# # Temporal split: last 20% as validation
# n_train = len(X_train)
# val_size = int(n_train * 0.2)
# X_train_final = X_train.iloc[:-val_size] if val_size > 0 else X_train
# y_train_final = y_train.iloc[:-val_size] if val_size > 0 else y_train
# X_val = X_train.iloc[-val_size:] if val_size > 0 else None
# y_val = y_train.iloc[-val_size:] if val_size > 0 else None

X_train_final = X_train
y_train_final = y_train
X_val = X_test
y_val = y_test

# Train final model using the utility function
final_model = train_best_model_from_grid(
    grid_search = grid_search,
    X_train = X_train_final,
    y_train = y_train_final,
    X_val = X_val,
    y_val = y_val,
    eval_metric = config.model.eval_metric,
    # early_stopping_rounds = config.training.early_stopping_rounds,
    verbose = config.training.verbose
)

print(f"Best parameters used: {grid_search.best_params_}")
# %% [markdown]
"""
## Test set evaluation

Compute error metrics (MSE, MAE, MAPE) and save them as JSON.
Predictions and ground truth are saved as CSV together with metadata (date, store_id).
"""

print("\nEvaluating on test set...")
y_pred_test_raw = final_model.predict(X_test)

if config.features.detrend:
    print("Inverting the log-differenced predictions to original scale...")
    target_col = config.data.target_col
    period = config.features.detrend_period

    df_test['y_true_orig'] = df_test.groupby(config.data.group_col)[target_col].shift(-1)
    df_test['base_log'] = np.log(df_test[target_col]).groupby(df_test[config.data.group_col]).shift(period - 1)

    meta_reset = meta_test.reset_index()
    df_test_reset = df_test.reset_index()
    aligned = pd.merge(
        meta_reset, 
        df_test_reset[['date', config.data.group_col, 'y_true_orig', 'base_log']],
        on=['date', config.data.group_col], 
        how='left'
    )
    y_test_eval = aligned['y_true_orig']
    base_log_series = aligned['base_log']
    y_pred_test_eval = inverse_detrend_predictions(
        y_pred_diff=y_pred_test_raw,
        base_log_series=base_log_series,
        detrend_period=period
    )
else:
    y_test_eval = y_test
    y_pred_test_eval = y_pred_test_raw

metrics = compute_metrics(y_test_eval, y_pred_test_eval)

print("\nTest metrics:")
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")

# Save metrics to JSON
with open(output_dir / "test_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

# Save predictions
save_predictions(
    y_true = y_test_eval,
    y_pred = y_pred_test_eval,
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
    evals_result , 
    metric = config.model.eval_metric,
    title = 'Learning Curve - One Step XGBoost',
    save_path = output_dir / "learning_curve.png" if config.output.save_plots else None
)

# %% [markdown]
"""
## Aggregate forecast plot
Sum values across all stores for each date and compare actual vs predicted series.
"""
test_df_with_pred = meta_test.copy()
test_df_with_pred['true'] = y_test_eval.values
test_df_with_pred['pred'] = y_pred_test_eval

aggregated = test_df_with_pred.groupby('date')[['true', 'pred']].sum()

plot_forecast(
    y_true=aggregated['true'],
    y_pred=aggregated['pred'],
    dates=aggregated.index,
    title='One-Step Forecasting: Real vs Predicted (Aggregated)',
    ylabel='Cash Balance',
    aggregate_by_date=False,
    save_path=output_dir / "forecast_plot.png" if config.output.save_plots else None
)
# %% [markdown]
"""
## Save the model

Serialize the final model using joblib and save it to the output directory.
"""

if config.output.save_model:
    model_path = output_dir / "final_model.pkl"
    joblib.dump(final_model, model_path)
    print(f"Model saved to {model_path}")

# %%
