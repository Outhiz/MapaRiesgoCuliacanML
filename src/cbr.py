import numpy as np
import pandas as pd

def load_cases(path="data/raw/casos.csv"):
    df = pd.read_csv(path)
    return df

def similarity(case1, case2, weights=None):
    if weights is None:
        weights = {'temp': 0.3, 'lluvia': 0.2, 'lag3': 0.4, 'hora': 0.1}

    sim = 0
    sim += weights['temp'] * (1 - abs(case1['temp'] - case2['temp']) / 40)
    sim += weights['lluvia'] * (1 - abs(case1['lluvia'] - case2['lluvia']) / 50)
    sim += weights['lag3'] * (1 - abs(case1['incidentes_lag3'] - case2['incidentes_lag3']) / 10)
    sim += weights['hora'] * (1 - abs(case1['hora'] - case2['hora']) / 23)

    return sim

def retrieve_similar_cases(cases, query, k=3):
    sims = []
    for _, c in cases.iterrows():
        sims.append((similarity(query, c), c))
    sims.sort(reverse=True, key=lambda x:x[0])
    return sims[:k]
