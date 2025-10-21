import pandas as pd

def load_data(incident_path, weather_path):
    incidentes = pd.read_csv(incident_path, parse_dates=['fecha'])
    clima = pd.read_csv(weather_path, parse_dates=['fecha'])
    df = incidentes.merge(clima, on='fecha', how='left')
    return df

def create_lags(df, lag_days=[3, 7, 14]):
    df = df.sort_values(['colonia', 'fecha'])
    for lag in lag_days:
        df[f'incidentes_lag{lag}'] = df.groupby('colonia')['incidentes'].shift(lag)
    return df

def clean_data(df):
    df = df.dropna()
    # Variable objetivo: 1 si hubo >=3 incidentes
    df['riesgo'] = (df['incidentes'] >= 3).astype(int)  #ARREGLAR COMO SE CALCULA EL RIESGO
    return df

def save_processed(df, path):
    df.to_csv(path, index=False)
    print(f"Dataset guardado en {path}")
