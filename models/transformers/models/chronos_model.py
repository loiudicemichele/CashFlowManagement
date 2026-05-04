"""
Chronos Model Wrapper for Zero-Shot Forecasting

This module defines the `ChronosModel` class, which:
- Loads a pre-trained Chronos pipeline.
- Caches the model in memory to avoid reloading between experiments.
- Provides a `predict()` method that takes a univariate time series context
  and returns a point forecast (median or mean) for the specified horizon.

The implementation uses the official ChronosPipeline from the `chronos` package.
"""

import torch
from typing import Optional, Union
import numpy as np
import pandas as pd

# Try to import chronos pipelines
try:
    from chronos import ChronosPipeline, Chronos2Pipeline
    CHRONOS_AVAILABLE = True
except ImportError:
    CHRONOS_AVAILABLE = False
    print("[!] Warning: 'chronos' package not installed. ChronosModel will not work.")


class ChronosModel:
    """
    Wrapper for Chronos zero-shot forecasting models.

    The model is loaded once and reused for multiple predictions.
    Supports both original Chronos and Chronos-2 pipelines.

    Attributes:
        config (ChronosConfig): Configuration object with model name, device, etc.
        pipeline (ChronosPipeline): The loaded Hugging Face pipeline.
        device (torch.device): Device on which the model is placed.
    """

    def __init__(self, config):
        if not CHRONOS_AVAILABLE:
            raise RuntimeError("Chronos package is not installed. Please install it via: pip install chronos-ts")

        self.config = config
        self.device = config.device if config.device else ("cuda" if torch.cuda.is_available() else "cpu")

        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config.seed)

        print(f"[+] Loading Chronos model from: {config.model_name}")
        
        # Try original ChronosPipeline first; if it fails, fallback to Chronos2Pipeline
        try:
            self.pipeline = ChronosPipeline.from_pretrained(
                config.model_name,
                device_map=self.device
            )
            self.is_chronos2 = False
            print("[+] Loaded using ChronosPipeline (original Chronos)")
        except Exception as e:
            print(f"[!] ChronosPipeline failed: {e}")
            print("[+] Attempting to load with Chronos2Pipeline...")
            self.pipeline = Chronos2Pipeline.from_pretrained(
                config.model_name,
                device_map=self.device
            )
            self.is_chronos2 = True
            print("[+] Loaded using Chronos2Pipeline (Chronos-2)")

        print(f"[+] Model loaded on {self.device}")

    def _set_inference_seed(self):
        """Reset PyTorch seeds before each predict call for reproducibility."""
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.config.seed)

    def _samples_to_point(self, samples: torch.Tensor) -> np.ndarray:
        """
        Reduce a (pred_len, num_samples) tensor to a 1-D point forecast.
 
        Args:
            samples: Tensor of shape (pred_len, num_samples).
 
        Returns:
            np.ndarray of shape (pred_len,).
        """
        if self.config.use_median:
            return torch.median(samples, dim=1).values.cpu().numpy()
        return torch.mean(samples, dim=1).cpu().numpy()

    def predict(self, context: Union[np.ndarray, list, torch.Tensor], prediction_length: Optional[int] = None) -> np.ndarray:
        """
        Generate point forecasts for the given context.

        The method takes a univariate time series context, uses the Chronos
        pipeline to sample future trajectories, and returns a summary statistic
        (median or mean) for each step in the forecast horizon.

        Args:
            context (array-like): Past observations as a 1D sequence of numbers.
                Can be a list, numpy array, or torch tensor.
            prediction_length (int, optional): Number of steps to forecast.
                If None, uses `config.prediction_length`.

        Returns:
            np.ndarray: 1D array of point forecasts (length = prediction_length).
        """
        if prediction_length is None:
            prediction_length = self.config.prediction_length

        # Convert context to torch tensor if needed
        if not isinstance(context, torch.Tensor):
            context = torch.tensor(context, dtype=torch.float32)

        # Chronos pipeline expects shape: (batch_size, sequence_length)
        # For univariate, batch_size=1
        if context.dim() == 1:
            context = context.unsqueeze(0)  # (1, seq_len)

        # For Chronos2, add variates dimension -> (1, 1, seq_len)
        if self.is_chronos2:
            context = context.unsqueeze(1)  # (batch, variates, time)

        # Set seed to ensure reproducibility of sampling
        self._set_inference_seed()

        # Generate forecast samples
        with torch.no_grad():
            forecast_samples = self.pipeline.predict(
                context,
                prediction_length=prediction_length,
                limit_prediction_length=False
            )
            # forecast_samples shape: (1, num_samples, prediction_length) for Chronos2pipeline
            #                         (1, prediction_length, num_samples) for ChronosPipeline
            # Extract the batch (first dimension)
            samples = forecast_samples[0].T         # (pred_len, num_samples)

            point_forecast = self._samples_to_point(samples)
            # if self.config.use_median:
            #     point_forecast = torch.median(samples, dim=1).values.cpu().numpy()
            # else:
            #     point_forecast = torch.mean(samples, dim=1).cpu().numpy()
            
        return point_forecast

    def predict_rolling_one_step(self, history: np.ndarray, test_actuals: np.ndarray, context_len: int, multi_step:bool = False) -> np.ndarray:
        """
        Convenience method for rolling one-step-ahead forecasts.

        Iteratively predict the next day using the last `context_len` values
        of the history (which includes training data and previously predicted
        values). Updates history after each prediction with the forecast.

        Args:
            history (np.ndarray): Initial historical values (training set) as 1D array.
            test_actuals (np.ndarray): True test values (used only to know how many
                steps to forecast; not used in prediction itself).
            context_len (int): Number of past days to use as context for each step.

        Returns:
            np.ndarray: Array of one-step predictions for the test period,
                same length as `test_actuals`.

        Note:
            This method uses the model's `predict()` with prediction_length=1
            repeatedly. It does NOT use the true test values to update the
            history (i.e., it is a true recursive forecast).
        """
        predictions = []
        current_history = history.copy()
    

        for i in range(len(test_actuals)):
            # Take the most recent context_len values
            context = current_history[-context_len:]

            # Predict next step
            next_forecast = self.predict(context, prediction_length=1)
            pred_value = next_forecast[0]
            predictions.append(pred_value)

            # Append the prediction to history (autoregressive update)
            current_history = np.append(current_history, 
                                        pred_value if multi_step else test_actuals[i])

        return np.array(predictions)
    
    def inverse_symlog(self,y):
        return np.sign(y) * np.expm1(np.abs(y))

    def predict_panel_covariates(
        self,
        context_df: pd.DataFrame,
        prediction_length: int,
        future_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Generates zero-shot multi-step forecasts for a panel dataset utilizing the 
        Chronos-2 native DataFrame API.
        
        This method processes all time series (grouped by item_id) simultaneously. 
        It leverages cross-series attention and In-Context Learning (ICL) by reading 
        the target and covariate columns directly from the long-format DataFrames. 
        If future deterministic covariates are provided via future_df, the model conditions 
        the forecast on these known future values.

        Args:
            context_df (pd.DataFrame): Historical data in long format. Must include the 
                                       identifier column, timestamp column, target column, 
                                       and any historical covariates.
            prediction_length (int): Number of future steps to forecast.
            future_df (pd.DataFrame, optional): Long-format dataframe containing known future 
                                                values for deterministic covariates across the prediction 
                                                horizon. Defaults to None.

        Returns:
            pd.DataFrame: A dataframe containing the multi-step forecasts for each item_id, 
                          typically including timestamps and predicted quantiles or mean values.
        """
        if not self.is_chronos2:
            raise RuntimeError("predict_panel_covariates() requires a Chronos-2 pipeline.")
        

        # Reset PyTorch and CUDA seeds to ensure deterministic sampling
        self._set_inference_seed()
        item_id = getattr(self.config, 'id_col', 'store_id')
        with torch.no_grad():
            forecast_df = self.pipeline.predict_df(
                df = context_df,
                future_df=future_df,
                prediction_length=prediction_length,
                id_column = item_id,
                timestamp_column='date',
                target=self.config.target_col
            )

        return forecast_df