import folium
import pandas as pd
import os

def generate_risk_map(df, output_path="reports/mapa_prediccion.html"):
    os.makedirs("reports", exist_ok=True)
    m = folium.Map(location=[24.8, -107.4], zoom_start=12)

    for _, row in df.iterrows():
        color = 'green'
        if row['riesgo'] == 1:
            color = 'red'
        elif row['incidentes'] >= 2:
            color = 'orange'

        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=7,
            popup=f"{row['colonia']} ({row['riesgo']})",
            color=color,
            fill=True,
            fill_opacity=0.7
        ).add_to(m)

    m.save(output_path)
    print(f"Mapa generado: {output_path}")
