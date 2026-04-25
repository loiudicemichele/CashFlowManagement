"""
XGBoost model wrapper for time series forecasting.

This module defines a class `XGBoostModel` that unifies:
    - Standard XGBRegressor (for n_steps = 1)
    - MultiOutputRegressor (for direct multi-step with n_steps > 1)
    - Separate models per step (custom implementation for diagnostic or heterogeneous hyperparameters)

It also provides robust saving/loading using both joblib (for scikit-learn compatibility)
and XGBoost's native save_model (for best performance when reloading).
"""

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from typing import Union, List, Optional, Dict, Any
from pathlib import Path


class XGBoostModel:
    """
    Unified XGBoost model for forecasting.

    The model can operate in three modes depending on `mode` and `n_steps`:
        - 'single': standard XGBRegressor (requires n_steps=1).
        - 'multioutput': uses sklearn's MultiOutputRegressor with XGBRegressor as base.
        - 'separate': trains one XGBRegressor per target step (useful for analyzing per-step errors).

    Args:
        mode: One of 'single', 'multioutput', 'separate'.
        n_steps: Number of future steps to predict (used only if mode != 'single').
        base_params: Dictionary of parameters to pass to each XGBRegressor.
        random_state: Random seed for reproducibility.
        n_jobs: Number of parallel threads (-1 for all).
        eval_metric: Evaluation metric for training (e.g., 'mae', 'rmse').
    """

    def __init__(self,
                 mode: str = 'single',
                 n_steps: int = 1,
                 base_params: Optional[Dict[str, Any]] = None,
                 random_state: int = 42,
                 n_jobs: int = -1,
                 eval_metric: str = 'mae'):
        self.mode = mode
        self.n_steps = n_steps
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.eval_metric = eval_metric

        # Default XGBoost parameters if not provided
        default_params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 5,
        }
        if base_params is not None:
            default_params.update(base_params)
        self.base_params = default_params
        self.base_params['random_state'] = random_state
        self.base_params['n_jobs'] = n_jobs
        self.base_params['eval_metric'] = eval_metric

        self._model = None          # Will hold the fitted model(s)
        self._is_fitted = False

        self._validate_mode()

    def _validate_mode(self):
        """Check consistency between mode and n_steps."""
        if self.mode == 'single' and self.n_steps != 1:
            raise ValueError("Mode 'single' requires n_steps = 1.")
        if self.mode not in ['single', 'multioutput', 'separate']:
            raise ValueError(f"Unknown mode: {self.mode}. Choose from 'single', 'multioutput', 'separate'.")

    def _create_base_estimator(self) -> XGBRegressor:
        """Create a fresh XGBRegressor with the stored parameters."""
        return XGBRegressor(**self.base_params)

    def fit(self,
            X: Union[pd.DataFrame, np.ndarray],
            y: Union[pd.Series, pd.DataFrame, np.ndarray],
            eval_set: Optional[tuple] = None,
            verbose: bool = False) -> 'XGBoostModel':
        """
        Fit the model according to the selected mode.

        Args:
            X: Feature matrix (shape: n_samples, n_features).
            y: Target(s). 
               - For mode='single': 1D array-like (n_samples,).
               - For mode='multioutput': 2D array-like (n_samples, n_steps).
               - For mode='separate': 2D array-like (n_samples, n_steps).
            eval_set: Optional tuple (X_val, y_val) for monitoring validation loss during training.
                       Only used when mode='single' (for multioutput/separate, use separate callbacks).
            verbose: If True, print training progress.

        Returns:
            self (fitted model).
        """
        # Convert inputs to numpy for consistent handling
        X = self._to_array(X)
        y = self._to_array(y)

        if self.mode == 'single':
            self._fit_single(X, y, eval_set, verbose)
        elif self.mode == 'multioutput':
            self._fit_multioutput(X, y, verbose)
        elif self.mode == 'separate':
            self._fit_separate(X, y, verbose)

        self._is_fitted = True
        return self

    def _fit_single(self, X, y, eval_set, verbose):
        """Train a single XGBRegressor (expects y 1D)."""
        if y.ndim != 1:
            raise ValueError(f"Mode 'single' expects 1D y, got shape {y.shape}")
        model = self._create_base_estimator()
        # Handle eval_set if provided
        if eval_set is not None:
            X_val, y_val = eval_set
            model.fit(X, y, eval_set=[(X, y), (X_val, y_val)], verbose=verbose)
        else:
            model.fit(X, y, verbose=verbose)
        self._model = model

    def _fit_multioutput(self, X, y, verbose):
        """Train a MultiOutputRegressor with XGBRegressor base."""
        if y.ndim == 1:
            # If y is 1D but n_steps > 1, reshape
            y = y.reshape(-1, 1)
        if y.shape[1] != self.n_steps:
            raise ValueError(f"y has {y.shape[1]} columns, but n_steps={self.n_steps}")
        base = self._create_base_estimator()
        multi_model = MultiOutputRegressor(base, n_jobs=self.n_jobs)
        # MultiOutputRegressor does not support eval_set directly, so we fit silently
        multi_model.fit(X, y)
        self._model = multi_model

    def _fit_separate(self, X, y, verbose):
        """Train one XGBRegressor per target step."""
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        if y.shape[1] != self.n_steps:
            raise ValueError(f"y has {y.shape[1]} columns, but n_steps={self.n_steps}")
        models = []
        for step in range(self.n_steps):
            if verbose:
                print(f"Training model for step {step+1}/{self.n_steps}")
            model = self._create_base_estimator()
            model.fit(X, y[:, step], verbose=verbose)
            models.append(model)
        self._model = models

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Generate predictions.

        Args:
            X: Feature matrix.

        Returns:
            Predictions: 
                - For mode='single': shape (n_samples,)
                - For mode='multioutput' or 'separate': shape (n_samples, n_steps)
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        X = self._to_array(X)

        if self.mode == 'single':
            pred = self._model.predict(X)
            return pred
        elif self.mode == 'multioutput':
            pred = self._model.predict(X)
            return pred
        elif self.mode == 'separate':
            preds = [model.predict(X) for model in self._model]
            return np.column_stack(preds)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def save(self, filepath: str, use_native: bool = False) -> None:
        """
        Save the model to disk.

        Args:
            filepath: Destination path (without extension if use_native=False, will add .pkl or .ubj).
            use_native: If True, use XGBoost's save_model (saves as .ubj).
                       If False (default), use joblib.dump (saves as .pkl).
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        if use_native:
            # Native XGBoost saving only works for a single XGBRegressor, not for MultiOutputRegressor or list.
            if self.mode == 'single' and isinstance(self._model, XGBRegressor):
                model_path = path.with_suffix('.ubj')
                self._model.save_model(str(model_path))
                print(f"Model saved natively to {model_path}")
            else:
                raise ValueError("Native save only supported for mode='single' with XGBRegressor. Use use_native=False for other modes.")
        else:
            # joblib works for any scikit-learn compatible object
            save_path = path.with_suffix('.pkl')
            joblib.dump(self, save_path)  # saves the whole XGBoostModel instance
            print(f"Model saved with joblib to {save_path}")

    @classmethod
    def load(cls, filepath: str, use_native: bool = False) -> 'XGBoostModel':
        """
        Load a saved model.

        Args:
            filepath: Path to the saved model file.
            use_native: Must match the saving method. If True, expects a .ubj file and only loads a single XGBRegressor.
                        If False, loads the full XGBoostModel object via joblib.

        Returns:
            Loaded XGBoostModel instance.
        """
        path = Path(filepath)
        if use_native:
            # Load as a plain XGBRegressor and wrap it into a XGBoostModel with mode='single'
            if not path.exists():
                raise FileNotFoundError(f"Native model file not found: {path}")
            xgb = XGBRegressor()
            xgb.load_model(str(path))
            # Create a wrapper instance
            instance = cls(mode='single', n_steps=1)
            instance._model = xgb
            instance._is_fitted = True
            return instance
        else:
            if not path.exists():
                raise FileNotFoundError(f"Joblib model file not found: {path}")
            return joblib.load(path)

    def _to_array(self, data):
        """Convert pandas Series/DataFrame or list to numpy array."""
        if isinstance(data, (pd.Series, pd.DataFrame)):
            return data.values
        elif isinstance(data, np.ndarray):
            return data
        elif data is None:
            return None
        else:
            return np.array(data)

    def get_feature_importance(self, importance_type: str = 'weight') -> Optional[pd.DataFrame]:
        """
        Return feature importance as a DataFrame (only for mode='single').

        Args:
            importance_type: 'weight', 'gain', 'cover', etc.

        Returns:
            DataFrame with columns 'feature' and 'importance', sorted descending.
            Returns None if mode is not 'single' or model not fitted.
        """
        if not self._is_fitted or self.mode != 'single':
            print("Feature importance available only for mode='single' after fitting.")
            return None
        if not hasattr(self._model, 'get_booster'):
            print("Model does not support get_booster()")
            return None
        booster = self._model.get_booster()
        importance = booster.get_score(importance_type=importance_type)
        # Convert to DataFrame
        df_imp = pd.DataFrame(list(importance.items()), columns=['feature', 'importance'])
        df_imp = df_imp.sort_values('importance', ascending=False)
        return df_imp


# Convenience functions for simple saving/loading (without full class)
def save_model(model, filepath: str) -> None:
    """
    Save a fitted XGBoostModel instance using joblib.

    Args:
        model: Instance of XGBoostModel.
        filepath: Destination path (will add .pkl extension).
    """
    if not isinstance(model, XGBoostModel):
        raise TypeError("Expected XGBoostModel instance.")
    model.save(filepath, use_native=False)


def load_model(filepath: str) -> XGBoostModel:
    """
    Load a saved XGBoostModel instance from joblib file.

    Args:
        filepath: Path to the .pkl file.

    Returns:
        Loaded XGBoostModel.
    """
    return XGBoostModel.load(filepath, use_native=False)