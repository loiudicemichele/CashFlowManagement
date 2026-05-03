"""
Experiment Configuration for Chronos Forecasting

Defines a dataclass that holds all hyperparameters and paths needed to run
zero‑shot forecasting experiments with the Chronos family of models.
Includes settings for data loading, model selection, rolling window size,
prediction length, reproducibility, and output directories.
"""

from dataclasses import dataclass, field
from typing import Optional, List
import os

@dataclass
class ChronosConfig:
    """
    Central configuration for Chronos‑based forecasting experiments.

    Attributes:
        data_dir (str): Directory containing the train/test CSV files.
        train_file (str): Filename of the training set (e.g., 'train_one_step.csv').
        test_file (str): Filename of the test set (e.g., 'test_one_step.csv').
        output_dir (str): Where to save predictions, metrics, and plots.
        target_col (str): Name of the target column to forecast (default 'cash_balance').
        model_name (str): Hugging Face model identifier or local path to Chronos model.
        context_len (int): Number of past observations to use as history for each prediction.
        prediction_length (int): How many future steps to forecast in one call (1 for one‑step).
        use_median (bool): If True, use the median of the forecast samples; otherwise use mean.
        seed (int): Global random seed for reproducibility.
        device (Optional[str]): Manually set device ('cuda' or 'cpu'), if None autodetect.
    """

    # Paths
    data_dir: str = '../../../Datasets/data_partitioned/aggregated'
    train_file: str = 'train_one_step.csv'
    test_file: str = 'test_one_step.csv'
    output_dir: str = '../outputs/chronos_one_step'

    # Data
    target_col: str = 'cash_balance'
    model_dir: str = '../models/'
    # Model
    model_name: str = os.path.join(model_dir, 'chronos-2-small')   # or "amazon/chronos-t5-small"

    # Forecasting
    context_len: int = 31
    prediction_length: int = 1              # 1 for one‑step, >1 for direct multi‑step
    use_median: bool = True                 # summarise forecast samples

    # Reproducibility
    seed: int = 42

    # Hardware
    device: Optional[str] = None            # 'cuda', 'cpu', or None for auto

    def __post_init__(self):
        """Print a summary of the configuration for tracking purposes."""
        print("\n" + "=" * 60)
        print("CHRONOS EXPERIMENT CONFIGURATION")
        print("=" * 60)
        print(f"Data directory:      {self.data_dir}")
        print(f"Train file:          {self.train_file}")
        print(f"Test file:           {self.test_file}")
        print(f"Output directory:    {self.output_dir}")
        print(f"Target column:       {self.target_col}")
        print(f"Model name:          {self.model_name}")
        print(f"Context length:      {self.context_len} days")
        print(f"Prediction length:   {self.prediction_length} step(s)")
        print(f"Summary statistic:   {'median' if self.use_median else 'mean'}")
        print(f"Random seed:         {self.seed}")
        print(f"Device:              {self.device if self.device else 'auto'}")
        print("=" * 60 + "\n")