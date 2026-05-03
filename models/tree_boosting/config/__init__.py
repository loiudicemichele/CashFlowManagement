"""
Configuration package for forecasting experiments.

This package provides dataclasses and helper functions to define,
validate, and serialize experiment parameters (hyperparameters, paths,
feature settings, etc.). All experiments should load a configuration
object from a YAML/JSON file or create it programmatically.
"""