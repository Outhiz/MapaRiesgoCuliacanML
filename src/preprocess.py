# src/preprocess.py (CÓDIGO MODIFICADO)
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
    
    # ----------------------------------------------------
    # 🌟 CORRECCIÓN 1: Renombrar 'incidentes_lag3' a 'lag3' para CBR
    # ----------------------------------------------------
    if 'incidentes_lag3' in df.columns:
        df = df.rename(columns={'incidentes_lag3': 'lag3',
                                'incidentes_lag7': 'lag7',
                                'incidentes_lag14': 'lag14'})

    # ----------------------------------------------------
    # 🌟 CORRECCIÓN 2: Crear 'hora' y 'dia' para CBR
    # (Asumiendo que la columna 'fecha' tiene la hora)
    # Si 'fecha' solo tiene la fecha, 'hora' será 0
    # ----------------------------------------------------
    if 'fecha' in df.columns:
        # Extraer el día de la semana y la hora
        df['dia'] = df['fecha'].dt.dayofweek # Lunes=0, Domingo=6 (Numérico)
        df['hora'] = df['fecha'].dt.hour
        
    return df

def clean_data(df):
    df = df.dropna()
    # Variable objetivo: 1 si hubo >=3 incidentes
    df['riesgo'] = (df['incidentes'] >= 3).astype(int)
    return df

def save_processed(df, path):
    df.to_csv(path, index=False)
    print(f"Dataset guardado en {path}")

# Si la columna 'fecha' en tu archivo incidentes.csv/clima.csv no tiene la hora,
# la columna df['fecha'].dt.hour devolverá 0 para todos, y 'hora' podría no ser útil
# para la similaridad, pero al menos evitará la KeyError.