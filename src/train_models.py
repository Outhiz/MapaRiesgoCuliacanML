import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, average_precision_score
from joblib import dump
from xgboost import XGBClassifier

def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    probas = model.predict_proba(X_test)[:, 1]
    f1 = f1_score(y_test, preds)
    auc_pr = average_precision_score(y_test, probas)
    return {'F1': round(f1, 3), 'AUC-PR': round(auc_pr, 3)}
# Modelo de Gradient Boosting con XGBoost
def train_gbm(X_train, y_train):
    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='aucpr',
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

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
    # Modelo GBM
    gbm = train_gbm(X_train, y_train)
    gbm_metrics = evaluate(gbm, X_test, y_test)

    # Guardar modelos
    dump(logreg, 'models/logreg.pkl')
    dump(rf, 'models/randomforest.pkl')
    dump(gbm, 'models/gbm_xgboost.pkl')

    print("Resultados:")
    print("Regresión Logística:", log_metrics)
    print("Random Forest:", rf_metrics)
    print("GBM:", gbm_metrics)
