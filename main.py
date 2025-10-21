from src.preprocess import load_data, create_lags, clean_data, save_processed
from src.train_models import train_and_compare
from src.evaluate import temporal_validation, spatial_validation
from src.plots import plot_temporal_results, plot_spatial_heatmap
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

if __name__ == "__main__":
    main()
