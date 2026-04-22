import matplotlib.pyplot as plt

def plot_PU(df):
    plt.figure(figsize=(12, 5))
    plt.plot(df['date'], df['pandemic_uncertainty'], color='navy', linewidth=1)
    plt.title('Pandemic Uncertainty Index')
    plt.xlabel('Data')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_oil_price(df):
    plt.figure(figsize=(12, 5))
    plt.plot(df['date'], df['oil_price'], color='navy', linewidth=1)
    plt.title('Andamento del prezzo del Brent (BZ=F) [andamento giornaliero interpolato]')
    plt.xlabel('Data')
    plt.ylabel('Prezzo (USD)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_EURIBOR(df):
    plt.figure(figsize=(12, 5))
    plt.plot(df['date'], df['euribor'], color='navy', linewidth=1)
    plt.title('Andamento dell\' EURIBOR [andamento giornaliero interpolato]')
    plt.xlabel('Data')
    plt.ylabel('Tasso di interesse')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_Consumer_Confidence(df):
    plt.figure(figsize=(12, 5))
    plt.plot(df['date'], df['consumer_confidence'], color='navy', linewidth=1)
    plt.title('Andamento dell\' Confidenza consumatore [andamento giornaliero interpolato]')
    plt.xlabel('Data')
    plt.ylabel('Confidenza del consumatore')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_IPCA(df):
    plt.figure(figsize=(12, 5))
    plt.plot(df['date'], df['inflation_index'], color='navy', linewidth=1)
    plt.title('Andamento dell\' IPCA [andamento giornaliero interpolato]')
    plt.xlabel('Data')
    plt.ylabel('Inflazione')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_HICP(df):
    plt.figure(figsize=(12, 5))
    plt.plot(df['date'], df['consumer_prices'], color='navy', linewidth=1)
    plt.title('Andamento dell\' IPCA [andamento giornaliero interpolato]')
    plt.xlabel('Data')
    plt.ylabel('Index')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_FAO(df):
    plt.figure(figsize=(12, 5))
    plt.plot(df['date'], df['fao'], color='navy', linewidth=1)
    plt.title('Andamento dell\' FAO [andamento giornaliero interpolato]')
    plt.xlabel('Data')
    plt.ylabel('Index')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_cumulated_indices(df):
    df_norm = df[['date', 'oil_price', 'consumer_confidence', 'inflation_index', 'consumer_prices']].copy()
    df_norm.set_index('date', inplace=True)
    df_norm = df_norm / df_norm.iloc[0] * 100

    # Plot
    plt.figure(figsize=(12,6))
    for col in df_norm.columns:
        plt.plot(df_norm.index, df_norm[col], label=col, linewidth=1)
    plt.title('Andamento delle variabili macroeconomiche (normalizzate a 100)')
    plt.xlabel('Data')
    plt.ylabel('Indice (base 100)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_data(df):
    df_norm = df[['date', 'net_inflow']].copy()
    df_norm.set_index('date', inplace=True)
    df_norm = df_norm / df_norm.iloc[0] * 100

    # Plot
    plt.figure(figsize=(12,6))
    for col in df_norm.columns:
        plt.plot(df_norm.index, df_norm[col], label=col, linewidth=1)
    plt.title('Andamento delle variabili macroeconomiche (normalizzate a 100)')
    plt.xlabel('Data')
    plt.ylabel('Indice (base 100)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def plot_net_inflow(df):
    # Crea una figura con 3 subplot
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # 1. Net inflow giornaliero
    axes[0].plot(df['date'], df['net_inflow'], color='blue', linewidth=0.8)
    axes[0].axhline(0, color='red', linestyle='--', linewidth=0.8)
    axes[0].set_ylabel('Net Inflow (€)')
    axes[0].set_title('Flusso di cassa giornaliero netto')
    axes[0].grid(True, alpha=0.3)

    # # 2. Saldo di cassa cumulato
    axes[1].plot(df['date'], df['cash_balance'], color='green', linewidth=1)
    axes[1].set_ylabel('Cassa cumulata (€)')
    axes[1].set_title('Evoluzione della liquidità')
    axes[1].grid(True, alpha=0.3)

    # 3. Confronto tra inflazione e fiducia (normalizzate per vedere i trend)
    # Normalizziamo a base 100 al primo giorno per confrontare i pattern
    inflation_norm = df['inflation_index'] / df['inflation_index'].iloc[0] * 100
    confidence_norm = df['consumer_confidence'] / df['consumer_confidence'].iloc[0] * 100
    oil_norm = df['oil_price'] / df['oil_price'].iloc[0]
    axes[2].plot(df['date'], inflation_norm, label='Inflazione (IPCA)', color='purple')
    axes[2].plot(df['date'], confidence_norm, label='Fiducia consumatori', color='orange')
    axes[2].plot(df['date'], oil_norm, label='Oil Price', color='blue')
    axes[2].set_ylabel('Indice (base 100)')
    axes[2].set_title('Variabili macroeconomiche')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    # Imposta etichetta x per tutti
    axes[2].set_xlabel('Data')

    plt.tight_layout()
    plt.show()

def plot_cash_balance(df):
    # 2. Raggruppa per data (assicurati di usare sempre 'd' da qui in poi)
    d = df.groupby('date')['cash_balance'].sum().reset_index()
    
    plt.figure(figsize=(12, 6))
    
    # 3. Aggiunto l'argomento 'label' così plt.legend() sa cosa mostrare
    plt.plot(d['date'], d['cash_balance'], color='navy', linewidth=1.5, label='Total Cash Balance')
    
    plt.title("Cash Balance", fontsize=14)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Cash Flow (€)", fontsize=12)
    
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45) # Inclina le date per una migliore leggibilità
    plt.tight_layout()      # Evita che le etichette vengano tagliate
    
    plt.show()

def plot_all_stores_net_inflow(df):
    plt.figure(figsize=(12, 6))
    
    for store_id, group in df.groupby('store_id'):
        group = group.sort_values('date')  # sicurezza: ordine cronologico
        plt.plot(group['date'], group['cash_balance'], linewidth=0.8, color='navy')
    
    plt.title('Net Cash Inflow')
    plt.xlabel('Date')
    plt.ylabel('Cash Flow')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_daily_total_sales(df):
    df_grouped = df.groupby('date')['daily_total_sales'].sum().reset_index()

    # 4. Creazione del grafico
    plt.figure(figsize=(12, 6))
    plt.plot(df_grouped['date'], df_grouped['daily_total_sales'], color='blue', linewidth=1.5)

    # Formattazione del grafico
    plt.title("Vendite Totali Giornaliere (Aggregato per Data)", fontsize=14)
    plt.xlabel("Data", fontsize=12)
    plt.ylabel("Vendite Totali (€)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()