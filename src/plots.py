import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_temporal_results(results_df, output_path="reports/temporal_metrics.png"):
    os.makedirs("reports", exist_ok=True)
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=results_df, x="Fold", y="F1", label="F1", marker="o")
    sns.lineplot(data=results_df, x="Fold", y="AUC-PR", label="AUC-PR", marker="s")
    plt.title("Validación temporal - Métricas por fold")
    plt.xlabel("Fold (división temporal)")
    plt.ylabel("Valor de métrica")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Gráfico guardado en {output_path}")


def plot_spatial_heatmap(results_df, output_path="reports/spatial_heatmap.png"):
    os.makedirs("reports", exist_ok=True)
    plt.figure(figsize=(7, 4))
    df_long = results_df.melt(id_vars="Zona_excluida", value_vars=["F1", "AUC-PR", "Recall@10%"])
    pivot = df_long.pivot(index="Zona_excluida", columns="variable", values="value")

    sns.heatmap(pivot, annot=True, cmap="Blues", fmt=".2f", cbar_kws={"label": "Valor"})
    plt.title("Validación espacial - Métricas por colonia excluida")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Gráfico guardado en {output_path}")
