import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, average_precision_score, recall_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


# MÉTRICAS PERSONALIZADAS

def recall_at_k(y_true, y_scores, k=0.1):
    """ Calcula Recall@k (proporción de verdaderos positivos dentro del top k%) """
    n = int(len(y_scores) * k)
    idx = np.argsort(y_scores)[::-1][:n]
    top_k_true = y_true.iloc[idx]
    return top_k_true.sum() / y_true.sum()


# VALIDACIÓN TEMPORAL

def temporal_validation(df, model_type='logreg', splits=3):
    tscv = TimeSeriesSplit(n_splits=splits)
    results = []

    X = df[['temp_max', 'lluvia_mm', 'incidentes_lag3']]
    y = df['riesgo']

    for i, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        if model_type == 'logreg':
            model = LogisticRegression(class_weight='balanced', max_iter=500)
        else:
            model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        results.append({
            "Fold": i + 1,
            "F1": f1_score(y_test, y_pred),
            "AUC-PR": average_precision_score(y_test, y_proba),
            "Recall@10%": recall_at_k(y_test, y_proba, k=0.1)
        })

    return pd.DataFrame(results)


# VALIDACIÓN ESPACIAL

def spatial_validation(df, model_type='logreg'):
    colonias = df['colonia'].unique()
    results = []

    for colonia_out in colonias:
        train = df[df['colonia'] != colonia_out]
        test = df[df['colonia'] == colonia_out]

        X_train = train[['temp_max', 'lluvia_mm', 'incidentes_lag3']]
        y_train = train['riesgo']
        X_test = test[['temp_max', 'lluvia_mm', 'incidentes_lag3']]
        y_test = test['riesgo']

        if model_type == 'logreg':
            model = LogisticRegression(class_weight='balanced', max_iter=500)
        else:
            model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        results.append({
            "Zona_excluida": colonia_out,
            "F1": f1_score(y_test, y_pred),
            "AUC-PR": average_precision_score(y_test, y_proba),
            "Recall@10%": recall_at_k(y_test, y_proba, k=0.1)
        })

    return pd.DataFrame(results)
