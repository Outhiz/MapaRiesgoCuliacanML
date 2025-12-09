import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os


# ============================================================
# FEATURE IMPORTANCE
# ============================================================
def plot_feature_importance(model, feature_names, save_path="reports/feature_importance.png"):
    """Genera un gráfico de importancia de características para RF o XGBoost."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(8, 6))
    plt.title("Importancia de características", fontsize=16)
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), np.array(feature_names)[indices], rotation=45, ha='right')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

    print(f"[OK] Feature importance guardado en: {save_path}")


# ============================================================
# VALIDACIÓN TEMPORAL - Gráfico
# ============================================================
def plot_temporal_results(df, save_path="reports/temporal_metrics.png"):
    plt.figure(figsize=(10, 6))
    sns.lineplot(x="Fold", y="F1", data=df, marker="o", label="F1")
    sns.lineplot(x="Fold", y="AUC-PR", data=df, marker="o", label="AUC-PR")
    sns.lineplot(x="Fold", y="Recall@10%", data=df, marker="o", label="Recall@10%")

    plt.title("Validación temporal (Rolling Origin)")
    plt.xlabel("Fold")
    plt.ylabel("Métrica")
    plt.legend()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

    print(f"[OK] Temporal metrics guardado en: {save_path}")


# ============================================================
# HEATMAP ESPACIAL
# ============================================================
def plot_spatial_heatmap(df, save_path="reports/spatial_heatmap.png"):
    df_long = df.melt(id_vars="Zona_excluida")

    pivot = df_long.pivot_table(
        index="Zona_excluida",
        columns="variable",
        values="value",
        aggfunc="mean"
    )

    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, cmap="YlOrRd", fmt=".3f")

    plt.title("Validación espacial: métricas por colonia excluida")
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

    print(f"[OK] Spatial heatmap guardado en: {save_path}")
