"""
Dataset Builder for Chronos-2 Native Fine-Tuning

Converts a long-format Panel DataFrame into the List[dict] format required
by Chronos2Pipeline.fit(). Each dictionary represents one time series and
contains its target values, past covariate arrays (historical), and a
declaration of future covariates (None-valued) that will be available at
inference time.

During training, .fit() handles sliding-window cropping internally, so the
full historical arrays are passed as-is. Deterministic covariates (known at
forecast time) must appear in BOTH past_covariates (with actual values) and
future_covariates (with None) to teach the model their future availability.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any


def build_chronos_training_dataset(
    df: pd.DataFrame,
    item_id_col: str,
    target_col: str,
    timestamp_col: str = 'date',
    past_only_covariate_cols: Optional[List[str]] = None,
    future_known_covariate_cols: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Convert a long-format panel DataFrame into the input format for Chronos2Pipeline.fit().

    Each time series (identified by item_id_col) becomes one dictionary with:
      - "target": full historical float array of the target variable.
      - "past_covariates": dict of all covariate arrays (stochastic + deterministic),
        covering the entire historical window. The .fit() method crops them internally.
      - "future_covariates": dict mapping deterministic covariate names to None,
        signalling to the model that these columns will be available at inference time.
        Passing None instead of actual values is the correct API contract: .fit()
        derives future windows from the same cropped training window automatically.

    Args:
        df (pd.DataFrame): Long-format panel DataFrame sorted by (item_id_col, timestamp_col).
        item_id_col (str): Column name identifying each time series (e.g. 'store_id').
        target_col (str): Column name of the target variable to forecast.
        timestamp_col (str): Column name of the date/timestamp. Defaults to 'date'.
        past_only_covariate_cols (List[str], optional): Covariate columns whose future
            values are NOT available at inference time (stochastic). Passed only to
            past_covariates.
        future_known_covariate_cols (List[str], optional): Covariate columns whose future
            values ARE known at inference time (deterministic, e.g. calendar features,
            holidays). Passed to both past_covariates (with real values) and
            future_covariates (with None as a declaration).

    Returns:
        List[Dict[str, Any]]: One dict per unique time series, ready for pipeline.fit().
    """
    past_only = past_only_covariate_cols or []
    future_known = future_known_covariate_cols or []
    all_covariates = past_only + future_known

    print(f"[+] Converting Panel DataFrame into Chronos-2 training format...")
    print(f"    Unique series found: {df[item_id_col].nunique()}")
    print(f"    Past-only covariates   : {len(past_only)} columns")
    print(f"    Future-known covariates: {len(future_known)} columns")

    training_data = []

    for item_id, group in df.groupby(item_id_col):
        group = group.sort_values(timestamp_col).reset_index(drop=True)

        ts_dict: Dict[str, Any] = {
            "target": group[target_col].values.astype(np.float32)
        }

        # Bundle all historical covariate values under past_covariates.
        # Both stochastic and deterministic columns are included here because
        # the model needs to observe their historical values during training.
        if all_covariates:
            ts_dict["past_covariates"] = {
                col: group[col].values.astype(np.float32)
                for col in all_covariates
            }

        # Declare deterministic covariates as available at inference time.
        # None signals the API contract: .fit() resolves the actual future
        # slice from the cropped training window — no explicit values needed.
        if future_known:
            ts_dict["future_covariates"] = {
                col: None for col in future_known
            }

        training_data.append(ts_dict)

    print(f"[+] Dataset conversion complete. {len(training_data)} series packaged.\n")
    return training_data