import pandas as pd
import os 

def load_cases(file_path='data/processed/casos.csv'):
    """
    Carga los casos históricos desde el archivo CSV.

    Retorna:
        pandas.DataFrame: Un DataFrame con todos los casos cargados.
    """
    try:
        # Aseguramos que la ruta sea absoluta o relativa al directorio de trabajo
        # Si el script main.py está en el directorio raíz y data/processed/casos.csv
        # es la ruta correcta desde ese punto, esto debería funcionar.
        cases_df = pd.read_csv(file_path)
        return cases_df
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo en la ruta: {file_path}")
        print("Asegúrate de que la ruta 'data/processed/casos.csv' sea correcta "
              "con respecto a donde ejecutas main.py.")
        return pd.DataFrame()
    except Exception as e:
        print(f"Ocurrió un error al cargar el archivo CSV: {e}")
        return pd.DataFrame()
# Pesos de importancia para cada atributo
weights = {
    'temp_max': 0.25,
    'lluvia_mm': 0.25,
    'lag3': 0.30,
    'hora': 0.10,
    'dia': 0.10
}


def similarity(case1, case2):
    sim = 0

    # normalización básica para cada atributo numérico
    sim += weights['temp_max'] * (1 - abs(case1['temp_max'] - case2['temp_max']) / 45)
    sim += weights['lluvia_mm'] * (1 - abs(case1['lluvia_mm'] - case2['lluvia_mm']) / 100)
    sim += weights['lag3'] * (1 - abs(case1['lag3'] - case2['lag3']) / 20)

    # hora (0–23)
    sim += weights['hora'] * (1 - abs(case1['hora'] - case2['hora']) / 23)

    # día como categórico: igual = 1, distinto = 0
    sim += weights['dia'] * (1 if case1['dia'] == case2['dia'] else 0)

    return sim


def retrieve_similar_cases(cases_df, query_case, k=5):
    sims = []

    for _, c in cases_df.iterrows():
        sims.append((similarity(query_case, c), c))

    sims.sort(reverse=True, key=lambda x: x[0])
    return sims[:k]
