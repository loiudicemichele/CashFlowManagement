# %% [markdown] Imports and Dataset Loading
"""
Imports and Dataset Loading
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset and parse the date column
DS_NAME = 'aggregated_stores_cashflow.csv'
df = pd.read_csv(f'../Datasets/{DS_NAME}')

# %% [markdown] Checking for missing data
"""
Checking for missing data
"""
print(f"Are there any missing data: {df.isna().any(axis=0).any()}")
print(f"Number of missing data: {df.isna().sum(axis=0).sum()}")

# %% [markdown] Encoding the temporal feature with their cyclical relations
"""
Encoding the temporal feature with their cyclical relations and removing the originals
"""
# Month
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

# Day of week
df['weekday_sin'] = np.sin(2 * np.pi * df['week_day'] / 7)
df['weekday_cos'] = np.cos(2 * np.pi * df['week_day'] / 7)

# Day within the month
df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
df.iloc[:, -10:]

# %% [markdown] Showing the variables correlation
"""
Showing the variables correlation
"""
import seaborn as sns
features = [c for c in df.columns if c not in ['date', 'store_id', 'cash_balance']] 
corr = df[features].corr()
plt.figure(figsize=(12,10))
sns.heatmap(corr, annot=False, cmap='RdBu_r', center=0, square=True)
plt.title('Correlazione tra feature (senza target)')
plt.show()


# %% [markdown] Removing unknown future features
"""
Removing unknown future features
"""
# Features categories to keep
index_features = ['date']

time_features = ['day_sin', 'day_cos', 'weekday_sin', 'weekday_cos', 
                 'month_sin', 'month_cos', 'weekend', 'holiday', 'actual_holiday', 
                 ]

external_factors_features = [
    'oil_price', 'euribor', 'consumer_confidence', 'inflation_index',
    'consumer_prices', 'fao', 'pandemic_uncertainty',
]

flow_categories_features = ['daily_nonfood_sales', 'daily_food_sales',
       'daily_total_sales', 'supplier_revenue_monthly', 'cogs_payment',
       'pos_commission_rate', 'waste_rate', 'daily_salary', 'services',
       'logistics', 'marketing', 'it', 'admin', 'other', 'insurance', 'taxes',
       'rent'
]

pred_flow_features = ['pred_outflow', 'pred_inflow']

pred_features = ['cash_balance', 'net_inflow']

# Set of features for each task
# 1. One-step ahead task -> We leverage the external factor information
features_one_step = index_features + time_features + external_factors_features + flow_categories_features + pred_flow_features + pred_features

# 2. Multi-step ahead task -> We can only use the time features, we don't know the external
# factors and expenses during the day-by-day prediction
feature_multi_step = index_features + time_features + pred_features

# 3. Multi-output prediction. We can leverage of today's indices to predict the future values.
clean_features = index_features + time_features + external_factors_features + flow_categories_features + pred_features
df.columns
# %% [markdown] Saving the dataset
output_dir = '../Datasets/data_partitioned/aggregated'
split_date = '2023-12-31'
train_one_step = df[features_one_step].loc[df['date'] <= split_date]
test_one_step = df[features_one_step].loc[df['date'] > split_date]
train_one_step.to_csv(f'{output_dir}/train_one_step.csv', index=False)
test_one_step.to_csv(f'{output_dir}/test_one_step.csv', index=False)

train_multi_step = df[feature_multi_step].loc[df['date'] <= split_date]
test_multi_step = df[feature_multi_step].loc[df['date'] > split_date]
train_multi_step.to_csv(f'{output_dir}/train_multi_step.csv', index=False)
test_multi_step.to_csv(f'{output_dir}/test_multi_step.csv', index=False)

train_clean = df[clean_features].loc[df['date'] <= split_date]
test_clean = df[clean_features].loc[df['date'] > split_date]
train_clean.to_csv(f'{output_dir}/train_multi_output.csv', index=False)
test_clean.to_csv(f'{output_dir}/test_multi_output.csv', index=False)
# %%
