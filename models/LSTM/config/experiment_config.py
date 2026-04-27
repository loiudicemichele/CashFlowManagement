
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ExperimentConfig:
    """
    Central configuration for LSTM forecasting experiments.

    Attributes:
        data_dir (str): Path to the directory containing train/test CSV files.
        output_dir (str): Directory where models, scalers, metrics, and plots will be saved.
        target_col (str): Name of the target column to predict.
        store_col (str) : Name of the store column to consider.
        no_scale_cols (List[str]): List of column names that should not be scaled 
                                    (e.g., cyclical time features, binary flags).
        n_lags (int): Number of past time steps (sequence length) used as input.
        n_outputs (int): Number of future time steps to predict (forecast horizon).
        n_splits (int): Number of splits for expanding window TimeSeriesSplit.
        validation_split (float): Fraction of training data to use as validation set
                                  during final training (chronological split).
        seed (int): Global random seed for reproducibility.
        param_grid (dict): Hyperparameter grid for GridSearchCV-style tuning.
        final_epochs (int): Number of epochs for the final model training.
    """
    
    # Paths
    data_dir: str = '../../../Datasets/data_partitioned/aggregated'
    output_dir: str = '../outputs/one_step'          # Will be overridden based on the task
    train_file: str = 'train_one_step.csv'
    test_file: str = 'test_one_step.csv'
    # Data
    target_col: str = 'cash_balance'
    store_col: str = 'store_id'
    diff_lag: int = 7 
    no_scale_cols: List[str] = field(default_factory=lambda: [
        'day_sin', 'day_cos', 'weekday_sin', 'weekday_cos',
        'month_sin', 'month_cos', 'weekend', 'holiday', 'actual_holiday'
    ])
    
    # Sequence parameters
    n_lags: int = 31
    n_outputs: int = 1
    
    # Training and CV
    n_splits: int = 5
    validation_split: float = 0.15
    seed: int = 42
    
    # Grid Search hyperparameters
    param_grid: dict = field(default_factory=lambda: {
        'hidden_dim': [64, 128],
        'num_layers': [1, 2],
        'learning_rate': [0.01, 0.001],
        'dropout': [0, 0.2],
        'batch_size': [128],
        'epochs': [30]
    })
    
    final_epochs: int = 200

    def __post_init__(self):
        """
        Optional post-initialization hook to print a summary of the configuration.
        This helps track which settings are active during experiments.
        """
        print("EXPERIMENT CONFIGURATION LOADED")
        print(f"Data Directory:     {self.data_dir}")
        print(f"Output Directory:   {self.output_dir}")
        print(f"Target Column:      {self.target_col}")
        print(f"Sequence:           {self.n_lags} lags → {self.n_outputs} step(s) ahead")
        print(f"CV Splits:          {self.n_splits} (validation split: {self.validation_split:.0%})")
        print(f"Random Seed:        {self.seed}")
        print(f"Grid Search Epochs: {self.param_grid['epochs']}")
        print(f"Final Epochs:       {self.final_epochs}")