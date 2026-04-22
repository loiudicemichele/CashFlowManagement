# %% Imports & Loading
import pandas as pd
from data_plots import *
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
sns.set_theme(style="whitegrid")

df = pd.read_csv('all_stores_cashflow.csv', parse_dates=['date'])
# Assegna store_id correttamente
n_stores = 10
rows_per_store = len(df) // n_stores
df['store_id'] = [i // rows_per_store for i in range(len(df))]

# %% Plotting
# plot_PU(df)
plot_all_stores_net_inflow(df)
plot_daily_total_sales(df)
plot_cash_balance(df)
# print(df.columns)
# trans_cols = ['daily_nonfood_sales',
#        'daily_food_sales', 'daily_total_sales', 'supplier_revenue_monthly',
#        'cogs_payment', 'pos_commission_rate', 'waste_rate', 'daily_salary',
#        'services', 'logistics', 'marketing', 'it', 'admin', 'other',
#        'insurance', 'taxes', 'rent', 'cash_balance', 'net_inflow',
#        'pred_outflow', 'pred_inflow', 'store_id']
# transactions = df[trans_cols]

# print(transactions.describe())


# # %%
# print(df['cash_balance'] == df['daily_total_sales'])















# # %%
# # Impostiamo lo stile


# # 1. Selezioniamo solo le variabili continue giornaliere
# daily_continuous = ['daily_food_sales', 'daily_nonfood_sales', 'daily_total_sales']

# plt.figure(figsize=(10, 6))
# sns.boxplot(data=df[daily_continuous], orient='h', palette='Set2')
# plt.title('Distribuzione delle Vendite Giornaliere', fontsize=14)
# plt.xlabel('Importo (€)')
# plt.show()

# # 2. Boxplot dei costi continui (pos_commission e waste_rate)
# daily_costs = ['pos_commission_rate', 'waste_rate']
# plt.figure(figsize=(10, 4))
# sns.boxplot(data=df[daily_costs], orient='h', palette='Reds')
# plt.title('Distribuzione dei Costi Operativi Quotidiani (Trattenute)', fontsize=14)
# plt.xlabel('Importo (€) - Valori Negativi')
# plt.show()

# # %%
# # Lista delle spese intermittenti principali
# # intermittent_expenses = ['cogs_payment', 'daily_salary', 'services', 'taxes', 'rent', 'logistics']

# # plt.figure(figsize=(14, 8))

# # # Creiamo un dataframe fittizio dove rimpiazziamo gli 0 con NaN, 
# # # in modo che il boxplot li ignori e plotteri solo i pagamenti reali.
# # df_no_zeros = df[intermittent_expenses].replace(0, np.nan)

# # sns.boxplot(data=df_no_zeros, orient='h', palette='magma')
# # plt.title('Distribuzione degli Importi delle Fatture/Spese (Solo giorni di effettivo pagamento)', fontsize=14)
# # plt.xlabel('Importo Uscita (€)')
# # plt.show()
# # %%
# # # Creiamo un plot a doppia scala per vedere come le vendite influenzano la cassa
# # fig, ax1 = plt.subplots(figsize=(15, 6))

# # color = 'tab:blue'
# # ax1.set_xlabel('Data')
# # ax1.set_ylabel('Vendite Totali (€)', color=color)
# # # Facciamo una media mobile a 7 giorni per pulire il rumore dei weekend
# # ax1.plot(df['date'], df['daily_total_sales'].rolling(7).mean(), color=color, alpha=0.7, label='Vendite (Media Mobile 7g)')
# # ax1.tick_params(axis='y', labelcolor=color)

# # ax2 = ax1.twinx()  
# # color = 'tab:green'
# # ax2.set_ylabel('Cash Balance (€)', color=color)  
# # ax2.plot(df['date'], df['cash_balance'], color=color, linewidth=2, label='Cassa Cumulata')
# # ax2.tick_params(axis='y', labelcolor=color)

# # plt.title('Trend Temporale: Vendite vs Cassa Disponibile', fontsize=16)
# # fig.tight_layout() 
# # plt.show()

# # # %%
# # plt.figure(figsize=(16, 12))

# # # Calcoliamo la correlazione di Pearson
# # corr_matrix = df.corr()

# # # Maschera per nascondere la metà superiore (speculare)
# # mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

# # sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm', 
# #             vmax=1, vmin=-1, center=0, square=True, linewidths=.5)

# # plt.title('Matrice di Correlazione delle Voci Finanziarie', fontsize=18)
# # plt.show()
# # %%

# %%
