import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, average_precision_score
from joblib import dump
from xgboost import XGBClassifier
from src.plots import plot_feature_importance


# ---- Función de evaluación ----
def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    probas = model.predict_proba(X_test)[:, 1]
    f1 = f1_score(y_test, preds)
    auc_pr = average_precision_score(y_test, probas)
    return {'F1': round(f1, 3), 'AUC-PR': round(auc_pr, 3)}


# ---- Modelo GBM (XGBoost) ----
def train_gbm(X_train, y_train):
    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='aucpr',
        random_state=42,
        tree_method="hist"
    )
    model.fit(X_train, y_train)
    return model


# ---- Entrenamiento general ----
def train_and_compare(data_path):
    df = pd.read_csv(data_path)
    X = df[['temp_max', 'lluvia_mm', 'lag3', 'lag7']]
    y = df['riesgo']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, shuffle=False
    )

    # ------------------------
    # MODELO BASE: LOGISTIC
    # ------------------------
    logreg = LogisticRegression(class_weight='balanced', max_iter=500)
    logreg.fit(X_train, y_train)
    log_metrics = evaluate(logreg, X_test, y_test)

    # ------------------------
    # MODELO AVANZADO: RANDOM FOREST
    # ------------------------
    rf = RandomForestClassifier(
        n_estimators=200,
        class_weight='balanced',
        random_state=42,
        max_depth=12
    )
    rf.fit(X_train, y_train)
    rf_metrics = evaluate(rf, X_test, y_test)

    # ------------------------
    # MODELO GBM (XGBOOST)
    # ------------------------
    gbm = train_gbm(X_train, y_train)
    gbm_metrics = evaluate(gbm, X_test, y_test)

    # ------------------------
    # FEATURE IMPORTANCE
    # ------------------------
    plot_feature_importance(
        rf,
        feature_names=X_train.columns,
        save_path="reports/feature_importance.png"
    )

    # ------------------------
    # Guardar modelos
    # ------------------------
    dump(logreg, 'models/logreg.pkl')
    dump(rf, 'models/randomforest.pkl')
    dump(gbm, 'models/gbm_xgboost.pkl')

    print("\n=== Resultados ===")
    print("Regresión Logística:", log_metrics)
    print("Random Forest:", rf_metrics)
    print("XGBoost GBM:", gbm_metrics)
