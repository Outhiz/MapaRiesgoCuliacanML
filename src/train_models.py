import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, average_precision_score
from joblib import dump

def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    probas = model.predict_proba(X_test)[:, 1]
    f1 = f1_score(y_test, preds)
    auc_pr = average_precision_score(y_test, probas)
    return {'F1': round(f1, 3), 'AUC-PR': round(auc_pr, 3)}

def train_and_compare(data_path):
    df = pd.read_csv(data_path)
    X = df[['temp_max', 'lluvia_mm', 'incidentes_lag3', 'incidentes_lag7']]
    y = df['riesgo']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, shuffle=False
    )

    # Modelo base
    logreg = LogisticRegression(class_weight='balanced', max_iter=500)
    logreg.fit(X_train, y_train)
    log_metrics = evaluate(logreg, X_test, y_test)

    # Modelo avanzado
    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    rf.fit(X_train, y_train)
    rf_metrics = evaluate(rf, X_test, y_test)

    dump(logreg, 'models/logreg.pkl')
    dump(rf, 'models/randomforest.pkl')

    print("Resultados:")
    print("Regresión Logística:", log_metrics)
    print("Random Forest:", rf_metrics)
