from src.preprocess import load_data, create_lags, clean_data, save_processed
from src.train_models import train_and_compare
from src.evaluate import temporal_validation, spatial_validation
from src.plots import plot_temporal_results, plot_spatial_heatmap
from src.cbr import load_cases, retrieve_similar_cases
from src.rules import apply_rules
from src.genetic import evolve
from joblib import load
import os
import pandas as pd

def main():
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    # --- Preprocesamiento
    df = load_data('data/raw/incidentes.csv', 'data/raw/clima.csv')
    df = create_lags(df)
    df = clean_data(df)
    save_processed(df, 'data/processed/dataset_final.csv')

    # --- Entrenamiento simple
    train_and_compare('data/processed/dataset_final.csv')

    cases = load_cases()
    example = df.iloc[-1]
    top_cases = retrieve_similar_cases(cases, example)

    rf = load("models/randomforest.pkl")
    y_proba = rf.predict_proba(df[['temp_max','lluvia_mm','lag3','lag7']])[:,1]
    best_threshold = evolve(df['riesgo'], y_proba)

    print("\n--- Integración Final ---")
    print("Casos similares recuperados (CBR):")
    print(top_cases)
    print("Mejor umbral optimizado (GA):", best_threshold)
    print("Reglas activadas:", apply_rules(example))

    # --- Validaciones

    df = pd.read_csv('data/processed/dataset_final.csv')

    print("\nValidación temporal (Rolling Origin):")
    temp_res = temporal_validation(df, model_type='logreg')
    print(temp_res)
    print("\nPromedios:\n", temp_res.mean(numeric_only=True))

    print("\nValidación espacial (Leave-one-colonia-out):")
    spatial_res = spatial_validation(df, model_type='rf')
    print(spatial_res)
    print("\nPromedios:\n", spatial_res.mean(numeric_only=True))

    # Guardar los resultados de validación
    plot_temporal_results(temp_res)
    plot_spatial_heatmap(spatial_res)

    from src.map_generator import generate_risk_map
    generate_risk_map(df)

if __name__ == "__main__":
    main()
