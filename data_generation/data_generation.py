# %% Imports
import pandas as pd
import numpy as np
from workalendar.europe import Italy # Calendario italiano delle festività
from data_plots import *
import yfinance as yf

from supermarkets import *

from fredapi import Fred


def get_external_indices():
    API_KEY = '3d0439b03f06c9faf751f4be1697f26a'
    fred = Fred(api_key=API_KEY) 
    # ==================================
    # Storico della serie
    # ==================================

    # Periodo storico della serie.
    start_date = '2019-01-01'
    end_date = '2024-12-31'
    dates = pd.date_range(start=start_date, end=end_date, freq='D')

    # Calendario italiano per le festività
    cal = Italy()
    is_holiday = [1 if cal.is_working_day(d) is False else 0 for d in dates] # Per ogni giornata, se è festività o meno

    # Costruiamo il dataframe base
    df = pd.DataFrame({
        'date': dates,
        'year': dates.year,
        'month': dates.month,
        'day': dates.day,
        'week_day': dates.dayofweek,  # 0=lunedì, 6=domenica
        'holiday': is_holiday,
        'actual_holiday': [1 if cal.is_holiday(d) else 0 for d in dates]
    })

    # Aggiungiamo una colonna 'weekend' (1 se sabato o domenica)
    df['weekend'] = (df['week_day'] >= 5).astype(int)

    # ==================================================================
    # Variabili macroeconomiche: Petrolio, EURIBOR, 
    # ==================================================================

    # ------------------------------------------------------------------
    # Varaibile - Prezzo carburanti.
    # ------------------------------------------------------------------


    brent = yf.download('BZ=F', start=start_date, end=end_date, progress=False)

    # print(brent)
    # La colonna 'Adj Close' è quella che ci interessa (prezzo aggiustato)
    brent_prices = brent['Close']

    # Allinea al calendario giornaliero (riempi i weekend con il valore del venerdì)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    brent_aligned = brent_prices.reindex(dates, method='ffill').bfill()

    # Aggiungi al dataframe principale
    df['oil_price'] = brent_aligned.values

    # **********************************
    # Osservo come il prezzo varia nella fascia. Influenza Excalation guerra in ucraina nel
    # Febbraio 2022 e come il trend si sia spostato verso un punto più alto.
    # **********************************
    # plot_oil_price(df)

    # print(df['prezzo_brent'].isnull().sum())

    # ------------------------------------------------------------------
    # Tasso Euribor 3 mesi (giornaliero) da FRED
    # Tasso di interesse a breve termine: Influenzano il costo del credito al consumo (rate, prestiti) 
    # e quindi la capacità di spesa delle famiglie. Tassi più alti tendono a ridurre i consumi. Possono
    #  anche influire sul costo del denaro per l'azienda (se ha debiti).
    # ------------------------------------------------------------------
    euribor_series = fred.get_series('IR3TIB01EZM156N', observation_start=start_date, observation_end=end_date)
    euribor_daily = euribor_series.reindex(dates)
    euribor_smooth = euribor_daily.interpolate(method='linear').bfill().ffill()
    df['euribor'] = euribor_smooth.values
    #plot_EURIBOR(df)

    # ------------------------------------------------------------------
    # Variabile - Fiducia dei consumatori: Indice che misura l'ottimismo delle famiglie riguardo alla situazione economica 
    # e alla propria capacità di spesa. Una maggiore fiducia si traduce in maggiori acquisti, anche non necessari.
    # ------------------------------------------------------------------
    confidence = fred.get_series('CSCICP03ITM665S', observation_start=start_date, observation_end=end_date)
    confidence_daily = confidence.reindex(dates)
    confidence_smooth = confidence_daily.interpolate(method='linear')
    confidence_smooth = confidence_smooth.bfill().ffill()
    df['consumer_confidence'] = confidence_smooth.values.flatten()
    # plot_Consumer_Confidence(df)

    # ------------------------------------------------------------------
    # Variabile - Inflazione IPCA Italia (mensile, interpolata) [Inflazione]
    # Misura l'aumento generale dei prezzi di beni e servizi. Un'inflazione più alta fa crescere il fatturato a parità di volumi (perché i 
    # prezzi aumentano), ma può anche ridurre i volumi se i consumatori tagliano la spesa. Inoltre, aumenta i costi di acquisto dai fornitori.
    # ------------------------------------------------------------------
    inflation = fred.get_series('ITACPIALLMINMEI',  observation_start=start_date, observation_end=end_date)
    inflation_daily = inflation.reindex(dates)
    inflation_smooth = inflation_daily.interpolate(method='linear')
    inflation_smooth = inflation_smooth.bfill().ffill()
    df['inflation_index'] = inflation_smooth.values.flatten()
    # plot_IPCA(df)



    # ------------------------------------------------------------------
    # Variabile - Harmonized Index of Consumer Prices: Electricity, Gas and Other Fuels for Italy
    # Misura l'aumento generale dei prezzi di energia elettrica e gas. Questo fattore ha influenza sulle spese aziendali
    # Maggiore è il prezzo di energia e gas maggiore saranno le spese legate alle bollette.
    # ------------------------------------------------------------------
    consumer_prices = fred.get_series('CP0450ITM086NEST',  observation_start=start_date, observation_end=end_date)
    consumer_prices = consumer_prices.reindex(dates)
    consumer_prices_smooth = consumer_prices.interpolate(method='linear')
    consumer_prices_smooth = consumer_prices_smooth.bfill().ffill()
    df['consumer_prices'] = consumer_prices_smooth.values.flatten()
    # plot_HICP(df)
    # print(df)
    # print(df.columns)
    # plot_cumulated_indices(df)

    # ------------------------------------------------------------------
    # Variabile - Food price index: Misura l'aumento generale dei prezzi dei beni primari
    # ------------------------------------------------------------------
    fao = fred.get_series('PFOODINDEXM',  observation_start=start_date, observation_end=end_date)
    fao = fao.reindex(dates)
    fao_smooth = fao.interpolate(method='linear')
    fao_smooth = fao_smooth.bfill().ffill()
    df['fao'] = fao_smooth.values.flatten()


    # ------------------------------------------------------------------
    # Variabile - World Pandemic Uncertainty Index
    # ------------------------------------------------------------------
    wupi = fred.get_series('WUPI', observation_start=start_date, observation_end=end_date)
    wupi = wupi.reindex(dates)
    wupi_smooth = wupi.interpolate(method='linear')
    wupi_smooth = wupi_smooth.bfill().ffill()
    df['pandemic_uncertainty'] = wupi_smooth.values.flatten()
    # plot_PU(df)
    return df
    # plot_FAO(df)

# ===================================================================================================
# ===================================================================================================
# ===================================================================================================
# ==============================GENERAZIONE DEI DATI DEL SUPERMERCATO================================
# ===================================================================================================
# ===================================================================================================
# ===================================================================================================

# 1 - ================= PARAMETRI DAILY SALES =================

def generate_store_data(df, params, store_id, store_path = './'):
    
    # =========================================================
    # 1. ESTRAZIONE PARAMETRI
    # =========================================================
    base_food_sales          = params.get("base_food_sales", 0)
    base_nonfood_sales       = params.get("base_nonfood_sales", 0)
    base_sales               = base_food_sales + base_nonfood_sales 
    
    annual_trend             = params.get("annual_trend", 0)
    other_inflow_in          = params.get("other_inflow_in", 0)
    promotional_inflow_prob  = params.get("promotional_inflow_prob", 0)
    supplier_revenue_monthly = params.get("supplier_revenue_monthly", 0)
    service_revenue_rate     = params.get("service_revenue_rate", 0)
    
    cogs_ratio               = params.get("cogs_ratio", 0)
    pos_commission_rate      = params.get("pos_commission_rate", 0)
    waste_rate               = params.get("waste_rate", 0)
    
    base_salary              = params.get("base_salary", 0)
    base_services            = params.get("base_services", 0)
    base_rent                = params.get("base_rent", 0)
    base_logistics           = params.get("base_logistics", 0)
    base_marketing           = params.get("base_marketing", 0)
    base_it                  = params.get("base_it", 0)
    base_admin               = params.get("base_admin", 0)
    base_other               = params.get("base_other", 0)
    
    insurance_annual         = params.get("insurance_annual", 0)
    base_fixed_tax           = params.get("base_fixed_tax", 0)
    tax_rate                 = params.get("tax_rate", 0)

    start_date = df['date'].min()
    end_date   = df['date'].max()

    # =========================================================
    # 2. CALCOLO VENDITE GIORNALIERE (TOTALE, FOOD E NON-FOOD)
    # =========================================================
    # coeff_infl = 0.7      
    # coeff_conf = 0.05     
    # coeff_oil  = -0.15      
    # coeff_eur  = -0.05   
    coeff_infl = 0.3      
    coeff_conf = 0.03     
    coeff_oil  = -0.08      
    coeff_eur  = -0.03  

    days_since_start   = (df['date'] - df['date'].min()).dt.days
    trend_factor       = (1 + annual_trend) ** (days_since_start / 365)
    weekly_factor      = {0:0.90, 1:0.88, 2:0.95, 3:1.00, 4:1.10, 5:1.20, 6:1.15}
    weekly_effect      = df['week_day'].map(weekly_factor)
    month_end_effect   = np.where(df['day'] >= 25, 1.15, 1.00)
    month_effect       = df['month'].map({1:1, 2:1, 3:1, 4:1, 5:1, 6:1, 7:1, 8:0.90, 9:1, 10:1, 11:1, 12:1.20})
    holiday_effect     = 1 - (0.2 * df['holiday'])
    pre_holiday_effect = np.where(df['actual_holiday'].shift(-1) == 1, 1.3, 1.00)

    inflation_norm     = df['inflation_index'] / df['inflation_index'].iloc[0]
    confidence_norm    = df['consumer_confidence'] / 100.0
    oil_norm           = df['oil_price'] / df['oil_price'].iloc[0]
    euribor_mean       = df['euribor'].mean()
    euribor_dev        = df['euribor'] - euribor_mean

    max_pandemic = df['pandemic_uncertainty'].max()
    covid_norm = df['pandemic_uncertainty'] / max_pandemic if max_pandemic > 0 else 0
    
    # 0.15 significa che AL PICCO del COVID le vendite aumenteranno esattamente del 15%
    covid_effect = 1 + (covid_norm * 0.15)

    macro_effect       = (1 + coeff_infl * (inflation_norm - 1)) * (1 + coeff_conf * (confidence_norm - 1)) * (1 + coeff_oil * (oil_norm - 1)) * (1 + coeff_eur * euribor_dev)

    # Vendite Totali + Rumore
    daily_sales = base_sales * trend_factor * weekly_effect * month_end_effect * month_effect * holiday_effect * pre_holiday_effect * macro_effect * covid_effect
    np.random.seed(42)
    daily_sales *= np.random.normal(1, 0.015, len(df))
    
    df['daily_total_sales'] = daily_sales.round(2) * covid_effect
    
    # Divisione Food vs Non-Food
    food_ratio = base_food_sales / base_sales if base_sales > 0 else 0
    df['daily_food_sales'] = (df['daily_total_sales'] * food_ratio).round(2)
    df['daily_nonfood_sales'] = (df['daily_total_sales'] * (1 - food_ratio)).round(2)

    transactions = []  

    # =========================================================
    # 3. GENERAZIONE TRANSAZIONI CON I NOMI ESATTI DELLE COLONNE
    # =========================================================
    for idx, row in df.iterrows():
        sales = row['daily_total_sales']
        date  = row['date']
        
        # Flussi di cassa operativi e trattenute
        transactions.append({'date': date, 'type': 'operating_inflow', 'amount': sales, 'due_date': date})
        
        if pos_commission_rate > 0:
            transactions.append({'date': date, 'type': 'pos_commission_rate', 'amount': -(sales * pos_commission_rate), 'due_date': date})
        if waste_rate > 0:
            transactions.append({'date': date, 'type': 'waste_rate', 'amount': -(sales * waste_rate), 'due_date': date})

    # Fornitori Merce (COGS)        
    payment_dates = set()
    for month_period in df['date'].dt.to_period('M').unique():
        days_in_month  = df[df['date'].dt.to_period('M') == month_period]['date'].tolist()
        num_payments   = np.random.choice([3, 4])
        selected_dates = np.random.choice(days_in_month, size=num_payments, replace=False)
        payment_dates.update(selected_dates)
    payment_dates.add(df['date'].iloc[-1])
    

    #accumulated_order_value = 0 
    accumulated_order_value = base_sales * cogs_ratio * 21
    for idx, row in df.iterrows():
        sales = row['daily_total_sales']
        #daily_order = sales * cogs_ratio * (row['fao'] / df['fao'].iloc[0])
        #fao_effect = 1 + 0.3 * (row['fao'] / df['fao'].iloc[0] - 1)  # impatto ridotto al 30%
        #daily_order = sales * cogs_ratio * fao_effect
        daily_order = sales * cogs_ratio
        accumulated_order_value += daily_order
        
        if row['date'] in payment_dates:
            delay = np.random.choice([30, 60, 90], p=[0.15, 0.7, 0.15])
            due_date = row['date'] + pd.Timedelta(days=delay)
            transactions.append({'date': row['date'], 'type': 'cogs_payment', 'amount': -accumulated_order_value, 'due_date': due_date})
            accumulated_order_value = 0

    # Costi Fissi Mensili
    for month in df['date'].dt.to_period('M').unique():
        last_day  = pd.Timestamp(month.end_time).normalize()
        first_day = pd.Timestamp(month.start_time).normalize()
        
        if last_day not in df['date'].values or first_day not in df['date'].values: continue
            
        inflation_factor = df.loc[df['date'] == last_day, 'inflation_index'].iloc[0] / df['inflation_index'].iloc[0]
        
        # Stipendi (daily_salary come richiesto, anche se pagati a fine mese)
        transactions.append({'date': last_day, 'type': 'daily_salary', 'amount': -(base_salary * inflation_factor), 'due_date': last_day})
        # Entrate Fornitori
        transactions.append({'date': last_day, 'type': 'supplier_revenue_monthly', 'amount': supplier_revenue_monthly, 'due_date': last_day})
        
        # Affitto, IT, Admin (Giorno 5)
        pay_date_early = first_day + pd.Timedelta(days=4) 
        if pay_date_early <= pd.Timestamp(end_date):
            transactions.append({'date': pay_date_early, 'type': 'rent',  'amount': -base_rent,  'due_date': pay_date_early})
            transactions.append({'date': pay_date_early, 'type': 'it',    'amount': -base_it,    'due_date': pay_date_early})
            transactions.append({'date': pay_date_early, 'type': 'admin', 'amount': -base_admin, 'due_date': pay_date_early})

        # Logistica, Marketing, Utenze, Altro (Giorno 15)
        pay_date_mid = first_day + pd.Timedelta(days=14) 
        if pay_date_mid <= pd.Timestamp(end_date):
            energy_idx  = df.loc[df['date'] == pay_date_mid, 'consumer_prices'].iloc[0]
            energy_norm = energy_idx / df['consumer_prices'].iloc[0]
            transactions.append({'date': pay_date_mid, 'type': 'services',  'amount': -(base_services * energy_norm), 'due_date': pay_date_mid})
            transactions.append({'date': pay_date_mid, 'type': 'logistics', 'amount': -(base_logistics * inflation_factor), 'due_date': pay_date_mid})
            transactions.append({'date': pay_date_mid, 'type': 'marketing', 'amount': -base_marketing, 'due_date': pay_date_mid})
            transactions.append({'date': pay_date_mid, 'type': 'other',     'amount': -(base_other * inflation_factor), 'due_date': pay_date_mid})

    # Assicurazione Annuale
    for date in pd.date_range(start_date, end_date, freq='YS'):
        pay_date = date + pd.Timedelta(days=9) 
        if pay_date <= pd.Timestamp(end_date):
            transactions.append({'date': pay_date, 'type': 'insurance', 'amount': -insurance_annual, 'due_date': pay_date})

    # Tasse Trimestrali
    temp_trans_df = pd.DataFrame(transactions)
    temp_trans_df['due_date'] = pd.to_datetime(temp_trans_df['due_date'])
    for q_end in pd.date_range(start_date, end_date, freq='QE'):
        pay_date = q_end + pd.Timedelta(days=20)
        if pay_date <= pd.Timestamp(end_date):
            mask = (temp_trans_df['due_date'].dt.year == q_end.year) & (temp_trans_df['due_date'].dt.quarter == q_end.quarter)
            quarterly_net = temp_trans_df.loc[mask, 'amount'].sum()
            variable_tax = quarterly_net * tax_rate if quarterly_net > 0 else 0
            inflation_factor = df.loc[df['date'] == q_end, 'inflation_index'].iloc[0] / df['inflation_index'].iloc[0]
            transactions.append({'date': q_end, 'type': 'taxes', 'amount': -(base_fixed_tax * inflation_factor + variable_tax), 'due_date': pay_date})


    # =========================================================
    # 4. CREAZIONE DATASET E FILTRO COLONNE FINALI
    # =========================================================
    trans_df = pd.DataFrame(transactions)

    # Raggruppamento Cashflow totale
    daily_net = trans_df.groupby('due_date')['amount'].sum().reset_index()
    daily_net.columns = ['date', 'net_inflow']

    # Pivot delle transazioni in colonne separate
    pivot_trans = trans_df.pivot_table(index='due_date', columns='type', values='amount', aggfunc='sum').reset_index()
    pivot_trans.rename(columns={'due_date': 'date'}, inplace=True)

    # Merge
    final_df = df.merge(daily_net, on='date', how='left')
    final_df = final_df.merge(pivot_trans, on='date', how='left') 

    final_df.fillna(0, inplace=True)
    
    # Saldo di Cassa e Predizioni
    final_df['cash_balance'] = 100000 + final_df['net_inflow'].cumsum()
    future_outflows, future_inflows = [], []
    for i, row in final_df.iterrows():
        curr_date = row['date']
        future = trans_df[(trans_df['due_date'] > curr_date) & (trans_df['due_date'] <= curr_date + pd.Timedelta(days=90))]
        future_outflows.append(future[future['amount'] < 0]['amount'].sum())
        future_inflows.append(future[future['amount'] > 0]['amount'].sum())

    final_df['pred_outflow'] = future_outflows
    final_df['pred_inflow']  = future_inflows

    final_df['store_id'] = store_id

    # ---------------------------------------------------------
    # LA LISTA EXACTA DELLE COLONNE RICHIESTE
    # ---------------------------------------------------------
    expected_cols = [
        'date', 'store_id', 'year', 'month', 'day', 'week_day', 'holiday', 'actual_holiday', 'weekend',
        'oil_price', 'euribor', 'consumer_confidence', 'inflation_index', 'consumer_prices', 'fao', 'pandemic_uncertainty',
        'daily_nonfood_sales', 'daily_food_sales', 'daily_total_sales',
        'supplier_revenue_monthly', 'cogs_payment', 'pos_commission_rate', 'waste_rate',
        'daily_salary', 'services', 'logistics', 'marketing', 'it', 'admin', 'other', 
        'insurance', 'taxes', 'rent', 'cash_balance', 'net_inflow', 'pred_outflow', 'pred_inflow'
    ]

    # Assicuriamoci che tutte le colonne esistano (se in un negozio una spesa non è mai scattata, la crea a 0)
    for col in expected_cols:
        if col not in final_df.columns:
            print(f"La colonna {col} non è stata trovata per {store_id}")
            final_df[col] = 0.0

    # FILTRO FINALE: Conserva solo le colonne richieste
    final_df = final_df[expected_cols]

    # Esportazione
    import os
    os.makedirs(store_path, exist_ok=True)
    filepath = f"{store_path}/supermarket_cashflow_{store_id}.csv"
    final_df.to_csv(filepath, index=False)
    
    print(f"   [+] Generato CSV pulito: {filepath}")
    
    # Restituiamo il dataframe filtrato (così anche il file aggregato avrà questa struttura)
    return final_df

def generate_all_stores_data(saving_path):
    print("Scaricando indici economici da FRED (richiede qualche secondo)...")
    df = get_external_indices()
    print("Indici scaricati. Inizio la generazione per i singoli supermercati...")
    
    all_data = pd.DataFrame()
    for store in all_params:
        ds = generate_store_data(df, store["params"], store["store_id"], store_path = saving_path)
        all_data = pd.concat([all_data, ds], ignore_index=True)
        
    all_data.to_csv(f'{saving_path}/all_stores_cashflow.csv', index=False)
    print("--> Master CSV 'all_stores_cashflow.csv' completato!")

def aggregate_stores(df, output_csv_path):
    
    # Colonne da sommare (tutti i flussi e i saldi)
    sum_cols = [
        'net_inflow', 'cash_balance', 'pred_outflow', 'pred_inflow',
        'daily_nonfood_sales', 'daily_food_sales', 'daily_total_sales',
        'supplier_revenue_monthly', 'cogs_payment', 'pos_commission_rate',
        'waste_rate', 'daily_salary', 'services', 'logistics', 'marketing',
        'it', 'admin', 'other', 'insurance', 'taxes', 'rent'
    ]
    
    # Colonne da mantenere come sono (uguali per tutti gli store nella stessa data)
    keep_cols = ['year', 'month', 'day', 'week_day', 'holiday', 'actual_holiday', 'weekend',
                 'oil_price', 'euribor', 'consumer_confidence', 'inflation_index',
                 'consumer_prices', 'fao', 'pandemic_uncertainty']
    
    # Group by data
    agg_dict = {col: 'sum' for col in sum_cols if col in df.columns}
    for col in keep_cols:
        if col in df.columns:
            agg_dict[col] = 'first'  # oppure 'mean' – tanto sono uguali
    
    df_agg = df.groupby('date', as_index=False).agg(agg_dict)
    
    df_agg.to_csv(output_csv_path, index=False)
    print(f"Dataset aggregato salvato in {output_csv_path}")
    return df_agg

def main():
    saving_path = './Datasets/data_generation'
    print(">>> INIZIO GENERAZIONE <<<")
    generate_all_stores_data(saving_path)
    
    print("Lettura del CSV per l'aggregazione...")
    df = pd.read_csv(f"{saving_path}/all_stores_cashflow.csv")
    aggregate_stores(df, './aggregated_stores_cashflow.csv')
    



if __name__ == "__main__":
    main()