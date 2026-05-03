"""
Grid search for time series forecasting with XGBoost.

This module provides a function `run_grid_search` that performs hyperparameter
optimization using TimeSeriesSplit cross-validation. It handles both standard
XGBRegressor and MultiOutputRegressor estimators, with proper alignment of the
parameter grid.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
from typing import Dict, Any, Union, Optional
from copy import deepcopy
import joblib
from pathlib import Path

def run_grid_search(estimator,
                    X: Union[pd.DataFrame, np.ndarray],
                    y: Union[pd.Series, pd.DataFrame, np.ndarray],
                    param_grid: Dict[str, Any],
                    cv_splits: int = 5,
                    scoring: Optional[Dict[str, str]] = None,
                    refit: str = 'MAE',
                    return_train_score: bool = True,
                    verbose: int = 1,
                    n_jobs: int = -1,
                    fix_multioutput_grid: bool = True) -> GridSearchCV:
    """
    Run hyperparameter grid search with TimeSeriesSplit cross-validation.

    Args:
        estimator: A scikit-learn compatible estimator (e.g., XGBRegressor, 
                   MultiOutputRegressor, or XGBoostModel). Must implement fit, predict.
        X: Feature matrix (n_samples, n_features).
        y: Target(s). For single-output: 1D; for multi-output: 2D.
        param_grid: Dictionary of parameters to search over.
        cv_splits: Number of TimeSeriesSplit folds (default 5).
        scoring: Dictionary of scoring metrics (e.g., {'MAE': 'neg_mean_absolute_error'}).
                 If None, uses standard metrics.
        refit: Metric to use for selecting the best model (default 'MAE').
        return_train_score: Whether to compute train scores (default True).
        verbose: Verbosity level for GridSearchCV (default 1).
        n_jobs: Number of parallel jobs (default -1).
        fix_multioutput_grid: If True and estimator is MultiOutputRegressor, automatically
                              prefix parameters with 'estimator__' if not already present.

    Returns:
        Fitted GridSearchCV object.
    """
    # Default scoring if not provided
    if scoring is None:
        scoring = {
            'MAE': 'neg_mean_absolute_error',
            'MSE': 'neg_mean_squared_error',
            'MAPE': 'neg_mean_absolute_percentage_error'
        }

    # Handle MultiOutputRegressor parameter prefix
    if fix_multioutput_grid and isinstance(estimator, MultiOutputRegressor):
        # Check if any key already starts with 'estimator__'
        keys = list(param_grid.keys())
        if not any(k.startswith('estimator__') for k in keys):
            # Prefix all keys
            new_grid = {}
            for k, v in param_grid.items():
                new_grid[f'estimator__{k}'] = v
            param_grid = new_grid
            print("Automatically prefixed param_grid keys with 'estimator__' for MultiOutputRegressor.")

    # Set up TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=cv_splits)

    # Create GridSearchCV
    grid_search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        cv=tscv,
        scoring=scoring,
        refit=refit,
        return_train_score=return_train_score,
        verbose=verbose,
        n_jobs=n_jobs
    )

    print(f"Starting grid search with {cv_splits} TimeSeriesSplit folds.")
    grid_search.fit(X, y)
    print("Grid search completed.")

    # Print best results
    print(f"Best parameters: {grid_search.best_params_}")
    # Printing the best scores  # Print best scores for each metric
    for metric in scoring.keys():
        mean_score = grid_search.cv_results_[f'mean_test_{metric}'][grid_search.best_index_]
        std_score = grid_search.cv_results_[f'std_test_{metric}'][grid_search.best_index_]
        print(f"Best {metric}: {abs(mean_score):.4f} (+/- {std_score:.4f})")
        

    return grid_search


def grid_search_results_to_dataframe(grid_search: GridSearchCV) -> pd.DataFrame:
    """
    Convert GridSearchCV results into a tidy DataFrame for analysis or saving.

    Args:
        grid_search: Fitted GridSearchCV object.

    Returns:
        DataFrame with columns: params, mean_test_MAE, std_test_MAE
    """
    results = grid_search.cv_results_
    # Build a list of dictionaries
    rows = []
    for i in range(len(results['params'])):
        row = {
            'params': str(results['params'][i]),
        }
        # Add all mean_test_*, Best MAE scorestd_test_*.
        for key in results.keys():
            if key.startswith('mean_test_') or key.startswith('std_test_') or key.startswith('rank_test_'):
                row[key] = results[key][i]
        rows.append(row)
    df = pd.DataFrame(rows)
    return df


def train_best_model_from_grid(grid_search: GridSearchCV,
                               X_train: Union[pd.DataFrame, np.ndarray],
                               y_train: Union[pd.Series, pd.DataFrame, np.ndarray],
                               X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
                               y_val: Optional[Union[pd.Series, pd.DataFrame, np.ndarray]] = None,
                               eval_metric: str = 'mae',
                               **fit_kwargs) -> Any:
    """
    Retrain a model on the full training set using the best hyperparameters from grid search.

    If validation data are provided, they are used as eval_set for early stopping (only if the
    estimator supports it, e.g., XGBRegressor). For MultiOutputRegressor, eval_set is ignored.

    Args:
        grid_search: Fitted GridSearchCV object.
        X_train: Full training features.
        y_train: Full training targets.
        X_val: Optional validation features.
        y_val: Optional validation targets.
        **fit_kwargs: Additional arguments passed to .fit() (e.g., verbose, early_stopping_rounds).

    Returns:
        Fitted model (the best estimator retrained on all data).
    """
    best_params = grid_search.best_params_
    estimator = grid_search.best_estimator_

    # If the estimator is a MultiOutputRegressor or does not support eval_set,
    # we just clone and retrain without eval_set.
    if isinstance(estimator, MultiOutputRegressor) or not hasattr(estimator, 'fit'):
        # For MultiOutputRegressor, clone the base estimator and rebuild
        base_estimator = estimator.estimator
        base_estimator.set_params(**best_params)
        new_estimator = MultiOutputRegressor(base_estimator, n_jobs=estimator.n_jobs)
        new_estimator.fit(X_train, y_train)
        return new_estimator
    else:
        best_params['eval_metric'] = eval_metric
        # Assuming XGBRegressor
        estimator_clone = estimator.__class__(**best_params)
        if X_val is not None and y_val is not None:
            eval_set = [(X_train, y_train), (X_val, y_val)]
            estimator_clone.fit(X_train, y_train, eval_set=eval_set, **fit_kwargs)
        else:
            estimator_clone.fit(X_train, y_train, **fit_kwargs)
        return estimator_clone


def save_grid_search(grid_search: GridSearchCV, filepath: str) -> None:
    """
    Save a fitted GridSearchCV object to disk using joblib.

    Args:
        grid_search: Fitted GridSearchCV object.
        filepath: Destination path (will add .pkl extension).
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(grid_search, path.with_suffix('.pkl'))
    print(f"GridSearchCV saved to {path.with_suffix('.pkl')}")


def load_grid_search(filepath: str) -> GridSearchCV:
    """
    Load a saved GridSearchCV object.

    Args:
        filepath: Path to the .pkl file.

    Returns:
        Loaded GridSearchCV object.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return joblib.load(path)