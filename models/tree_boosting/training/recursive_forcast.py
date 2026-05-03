"""
Recursive (iterative) multi-step forecasting for panel data.
This module provides a single function `recursive_forecast` that extends
a single-step forecasting model to predict multiple steps ahead by
recursively feeding predictions back as lag features.
"""

import pandas as pd
import numpy as np
from typing import List

def recursive_forecast(model,
                       X_test: pd.DataFrame,
                       meta_test: pd.DataFrame,
                       lag_cols: List[str],
                       non_lag_cols: List[str],
                       store_col: str = 'store_id') -> pd.DataFrame:
    """
    Perform recursive multi-step forecasting on panel data (multiple stores).
    
    Args:
        model: Fitted single-step model with `.predict(X)` method.
        X_test: DataFrame of test features (must include `lag_cols` and `non_lag_cols`).
        meta_test: DataFrame with metadata (must contain `store_col` and have dates as index).
        lag_cols: List of lag column names in order from most recent to oldest.
                  Typically: ['target_lag_1', 'target_lag_2', ...]
        non_lag_cols: List of column names that are NOT lag columns (e.g., time features).
        store_col: Name of the column identifying the store (or group).
        
    Returns:
        Numpy array with predictions aligned with the original X_test row order.
    """
    # Create a unified dataframe for easy manipulation during the loop
    current_state = X_test.copy()
    
    # Track original order to avoid Pandas duplicate index explosion later
    current_state['_orig_order'] = np.arange(len(current_state))
    
    current_state[store_col] = meta_test[store_col].values
    current_state['date'] = meta_test.index
    
    # Save the exact original column order for XGBoost
    original_cols = X_test.columns.tolist()
    
    test_dates = sorted(current_state['date'].unique())
    
    # Prepare lag shifting indices (right shift for lags: lag_1 -> lag_2, etc.)
    lags_to_update = lag_cols[1:]   # lag_2, lag_3...
    lags_source = lag_cols[:-1]     # lag_1, lag_2...

    # Extract the very first day to initialize the state
    first_date = test_dates[0]
    daily_state = current_state[current_state['date'] == first_date].copy()
    
    all_predictions = []
    
    for i, date in enumerate(test_dates):
        # Predict next step using current daily state WITH EXACT ORIGINAL COLUMN ORDER
        X_today = daily_state[original_cols]
        preds_today = model.predict(X_today)
        
        # Store predictions
        res_today = pd.DataFrame({
            store_col: daily_state[store_col].values,
            'y_pred': preds_today,
            '_orig_order': daily_state['_orig_order'].values # Keep track of order
        })
        all_predictions.append(res_today)
        
        # Update the state for tomorrow (if tomorrow exists in test_dates)
        if i + 1 < len(test_dates):
            next_date = test_dates[i + 1]
            
            # Shift lag columns (lag_1 becomes lag_2, etc.)
            daily_state[lags_to_update] = daily_state[lags_source].values
            
            # Insert today's prediction as the most recent lag (lag_1)
            daily_state[lag_cols[0]] = preds_today
            
            # Load real ground-truth for non-lag features of tomorrow (e.g. day of week, holidays)
            next_day_real_data = current_state[current_state['date'] == next_date]
            
            daily_state[non_lag_cols] = next_day_real_data[non_lag_cols].values
            daily_state[store_col] = next_day_real_data[store_col].values
            daily_state['_orig_order'] = next_day_real_data['_orig_order'].values
            
            # It's important to also update the date internally
            daily_state['date'] = next_date

    # Concatenate all predictions
    df_preds = pd.concat(all_predictions)
    
    # Sort by original X_test row order to perfectly align with y_test
    df_preds = df_preds.sort_values('_orig_order')
    
    return df_preds['y_pred'].values