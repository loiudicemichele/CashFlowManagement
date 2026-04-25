"""
Experiment configuration using dataclasses.

This module defines the ExperimentConfig dataclass which holds all
parameters needed for a forecasting experiment (data paths, feature
engineering settings, model hyperparameters, grid search options,
output directories).
"""

import json
import yaml
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
from pathlib import Path


@dataclass
class DataConfig:
    """Configuration for data loading and splitting."""
    train_path: str = "../Datasets/data_partitioned/train_one_step.csv"
    test_path: str = "../Datasets/data_partitioned/test_one_step.csv"
    target_col: str = "cash_balance" # net_inflow for testing
    group_col: str = "store_id"
    date_col: str = "date"
    parse_dates: bool = True


@dataclass
class FeatureConfig:
    """Feature engineering parameters."""
    n_lags: int = 31
    n_steps: int = 1
    time_features: List[str] = field(default_factory=lambda: [
        'day_sin', 'day_cos', 'weekday_sin', 'weekday_cos',
        'month_sin', 'month_cos', 'weekend', 'holiday', 'actual_holiday'
    ])
    detrend: bool = False
    detrend_period: int = 7          # used for differencing when detrend=True
    # Multi-output strategy: 'multioutput_regressor', 'separate_models', or None (for n_steps=1)
    multi_output_strategy: Optional[str] = None
    recursive_horizon: int = 14


@dataclass
class ModelConfig:
    """Base XGBoost model hyperparameters (will be overridden by grid search)."""
    model_type: str = "xgboost"
    n_estimators: int = 100
    learning_rate: float = 0.1
    max_depth: int = 5
    random_state: int = 42
    n_jobs: int = -1
    eval_metric: str = "mae"


@dataclass
class GridSearchConfig:
    """Grid search settings."""
    param_grid: Dict[str, List[Any]] = field(default_factory=lambda: {
        'n_estimators': [100, 500],
        'learning_rate': [0.01, 0.1],
        'max_depth': [5, 10],
    })
    cv_splits: int = 5
    scoring: Dict[str, str] = field(default_factory=lambda: {
        'MAE': 'neg_mean_absolute_error',
        'MSE': 'neg_mean_squared_error',
        'MAPE': 'neg_mean_absolute_percentage_error'
    })
    refit_metric: str = "MAE"
    verbose: int = 1


@dataclass
class TrainingConfig:
    """Final training parameters."""
    early_stopping_rounds: Optional[int] = None 
    verbose: bool = False
    eval_set_fraction: float = 0.2


@dataclass
class OutputConfig:
    """Output directories and saving options."""
    experiment_name: str = "default_experiment"
    base_output_dir: str = "../outputs"
    save_model: bool = True
    save_predictions: bool = True
    save_plots: bool = True


@dataclass
class ExperimentConfig:
    """Top‑level configuration collecting all sub‑configs."""
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    grid_search: GridSearchConfig = field(default_factory=GridSearchConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


# ----------------------------------------------------------------------
# Helper functions for loading / saving configuration
# ----------------------------------------------------------------------

def load_config(file_path: str) -> ExperimentConfig:
    """
    Load an ExperimentConfig from a JSON or YAML file.

    Args:
        file_path: Path to the configuration file (.json or .yaml/.yml).

    Returns:
        ExperimentConfig object populated with the file contents.

    Raises:
        ValueError: If the file extension is not supported.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {file_path}")

    with open(path, 'r', encoding='utf-8') as f:
        if path.suffix in ['.yaml', '.yml']:
            data = yaml.safe_load(f)
        elif path.suffix == '.json':
            data = json.load(f)
        else:
            raise ValueError("Unsupported config file format. Use .json or .yaml/.yml")

    # Recursively reconstruct dataclasses
    return _dict_to_dataclass(data, ExperimentConfig)


def save_config(config: ExperimentConfig, file_path: str) -> None:
    """
    Save an ExperimentConfig to a JSON or YAML file.

    Args:
        config: ExperimentConfig instance.
        file_path: Output path (.json or .yaml/.yml).
    """
    path = Path(file_path)
    data = asdict(config)

    # Create parent directory if it doesn't exist
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w', encoding='utf-8') as f:
        if path.suffix in ['.yaml', '.yml']:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
        elif path.suffix == '.json':
            json.dump(data, f, indent=4, ensure_ascii=False)
        else:
            raise ValueError("Unsupported config file format. Use .json or .yaml/.yml")


def _dict_to_dataclass(data: Dict, dataclass_type):
    """
    Recursively convert a dictionary into a nested dataclass structure.

    This helper is used by load_config to reconstruct nested dataclasses
    from serialized data.
    """
    if not isinstance(data, dict):
        return data

    # Get the field types of the target dataclass
    field_types = {f.name: f.type for f in dataclass_type.__dataclass_fields__.values()}
    kwargs = {}
    for key, value in data.items():
        if key in field_types:
            field_type = field_types[key]
            # If the field is itself a dataclass, recurse
            if hasattr(field_type, '__dataclass_fields__'):
                kwargs[key] = _dict_to_dataclass(value, field_type)
            else:
                kwargs[key] = value
    return dataclass_type(**kwargs)