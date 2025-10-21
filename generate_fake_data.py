import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

# --- Crear carpeta si no existe
os.makedirs("data/raw", exist_ok=True)

# --- Colonias simuladas
colonias = ["Centro", "Las Quintas", "Tierra Blanca", "Lomas del Blvd", "Humaya"]

# --- Fechas de 3 meses (septiembre a noviembre 2025)
fechas = pd.date_range(start="2025-09-01", end="2025-11-30")

# --- Generar datos de incidentes (ruido aleatorio)
data = []
np.random.seed(42)
for colonia in colonias:
    base_incidentes = np.random.randint(0, 6)  # promedio base
    tendencia = np.random.normal(0, 1, len(fechas))
    for i, fecha in enumerate(fechas):
        incidentes = abs(int(base_incidentes + tendencia[i] + np.random.randint(0, 3)))
        data.append([fecha, colonia, incidentes])

df_inc = pd.DataFrame(data, columns=["fecha", "colonia", "incidentes"])
df_inc.to_csv("data/raw/incidentes.csv", index=False)
print("✅ Archivo generado: data/raw/incidentes.csv")

# --- Generar datos de clima
np.random.seed(24)
clima = {
    "fecha": fechas,
    "temp_max": np.random.normal(34, 2, len(fechas)).round(1),
    "lluvia_mm": np.random.choice([0, 0, 1, 2, 3, 5, 10, 15], size=len(fechas))
}
df_clima = pd.DataFrame(clima)
df_clima.to_csv("data/raw/clima.csv", index=False)
print("Archivo generado: data/raw/clima.csv")
