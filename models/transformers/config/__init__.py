"""
Configuration Package for Chronos Forecasting Experiments

This package provides centralized configuration management using a dataclass.
All parameters (paths, model name, context length, prediction horizon, random seed, etc.)
are defined in one place, making it easy to switch between different experiment
setups (one-step, multi-step, multi-output) without modifying the core logic.

The main class is `ChronosConfig`, which should be instantiated and passed to
other modules (data loading, forecasting, evaluation) to ensure consistency.
"""