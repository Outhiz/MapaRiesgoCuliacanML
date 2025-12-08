def apply_rules(row):
    rules = []

    if row['incidentes_lag3'] >= 4:
        rules.append("Histórico reciente indica riesgo alto.")

    if row['temp_max'] > 33 and row['lluvia_mm'] < 5:
        rules.append("Condiciones climáticas favorecen incidentes.")

    if row['lluvia_mm'] > 30:
        rules.append("Lluvia intensa → riesgo de accidentes.")

    return rules
