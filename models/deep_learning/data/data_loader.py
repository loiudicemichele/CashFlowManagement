"""
Data loading and scaling utilities for the forecasting pipeline.

This module handles reading the partitioned train/test CSV files, applying
MinMax scaling to continuous features while preserving cyclical/binary features
unscaled, and saving the fitted scalers for later inference.
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib


def load_data(config) -> pd.DataFrame:
    """
    Load training and test datasets from the specified directory.
    The filenames are dynamically fetched from the config.
    """
    train_df = pd.read_csv(
        f'{config.data_dir}/{config.train_file}',
        index_col='date',
        parse_dates=['date']
    )
    test_df = pd.read_csv(
        f'{config.data_dir}/{config.test_file}',
        index_col='date',
        parse_dates=['date']
    )

    train_df.sort_values(['date'], inplace=True)
    test_df.sort_values(['date'], inplace=True)

    print(f"[+] Training data loaded from {config.train_file}: {train_df.shape}")
    print(f"[+] Test data loaded from {config.test_file}: {test_df.shape}")

    return train_df, test_df


def scale_data(train_df, test_df, config):
    """
    Scale continuous features and target using MinMaxScaler.

    Features listed in config.no_scale_cols (cyclical time encodings, binary flags)
    are left untouched. The target column is scaled independently with its own scaler
    to facilitate inverse transformation after prediction.

    Args:
        train_df (pd.DataFrame): Training data.
        test_df (pd.DataFrame): Test data.
        config: ExperimentConfig object containing no_scale_cols and target_col.

    Returns:
        tuple: (train_scaled_df, test_scaled_df, feature_scaler, target_scaler)
    """
    features_to_scale = [col for col in train_df.columns if col not in config.no_scale_cols]
    unscaled_cols = [col for col in train_df.columns if col in config.no_scale_cols]

    # Initialize scalers

    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    # Fit and transform on training data
    train_scaled_array = feature_scaler.fit_transform(train_df[features_to_scale])
    train_scaled_df = pd.DataFrame(
        train_scaled_array,
        columns=features_to_scale,
        index=train_df.index
    )

    # Transform test data
    test_scaled_array = feature_scaler.transform(test_df[features_to_scale])
    test_scaled_df = pd.DataFrame(
        test_scaled_array,
        columns=features_to_scale,
        index=test_df.index
    )

    # Adding the unscaled Features
    for col in unscaled_cols:
        if col in train_df.columns:
            train_scaled_df[col] = train_df[col]
            test_scaled_df[col] = test_df[col]

    # Fit target scaler separately
    target_scaler.fit(train_df[[config.target_col]])

    print(f"[+] Scaling completed. Scaled {len(features_to_scale)} continuous features.")
    print(f"[+] Left {len(config.no_scale_cols)} cyclical/binary features unscaled.")

    return train_scaled_df, test_scaled_df, feature_scaler, target_scaler


def save_scalers(feature_scaler, target_scaler, output_dir):
    """
    Persist the fitted scalers to disk for later use during inference.

    Args:
        feature_scaler: Fitted MinMaxScaler for the features.
        target_scaler: Fitted MinMaxScaler for the target variable.
        output_dir (str): Directory where the scalers will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(feature_scaler, f"{output_dir}/feature_scaler.pkl")
    joblib.dump(target_scaler, f"{output_dir}/target_scaler.pkl")
    print(f"[+] Scalers saved to {output_dir}/")

def apply_log_diff(df: pd.DataFrame, target_col: str, store_col: str = 'store_id', diff_lag: int = 1) -> pd.DataFrame:
    """
    Apply logarithmic transformation and differencing to the target column.
    The operation is grouped by the store column to prevent data leakage between different series.

    Args:
        df (pd.DataFrame): The input dataframe containing the target column.
        target_col (str): The name of the target column to transform.
        store_col (str): The name of the column identifying the different stores.
        diff_lag (int): The lag period for differencing. Defaults to 1.

    Returns:
        pd.DataFrame: The transformed dataframe with the target column replaced by its log-differences,
                      and an additional 'original_target' column preserved for later inversion.
    """
    df_transformed = df.copy()
    
    # Apply log transformation safely (assuming target > 0)
    df_transformed[f'{target_col}_log'] = np.log(df_transformed[target_col])
    
    # Apply differencing grouped by store with specified lag
    df_transformed[f'{target_col}_log_diff'] = df_transformed.groupby(store_col)[f'{target_col}_log'].diff(periods=diff_lag)
    
    # Drop the first 'diff_lag' rows of each store which now contain NaN due to differencing
    df_transformed = df_transformed.dropna(subset=[f'{target_col}_log_diff'])
    
    # Preserve the original target for inversion during evaluation
    # df_transformed['original_target'] = df_transformed[target_col]
    
    # Replace the target column with the transformed values to reuse existing pipeline
    df_transformed[target_col] = df_transformed[f'{target_col}_log_diff']
    
    # Drop temporary columns
    df_transformed = df_transformed.drop(columns=[f'{target_col}_log', f'{target_col}_log_diff'])
    
    print(f"[+] Log-Diff transformation applied (lag={diff_lag}). Shape after dropping NaNs: {df_transformed.shape}")
    
    return df_transformed
