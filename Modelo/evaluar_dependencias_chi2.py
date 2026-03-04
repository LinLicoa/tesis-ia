import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import os

# Rutas
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
RESULTADOS_PATH = os.path.join(ROOT_PATH, "datos")
DATA_PATH = os.path.join(RESULTADOS_PATH, "Datos_entrenamiento.csv")
REPORT_OUTPUT = os.path.join(ROOT_PATH, "reporte_chi_cuadrado.md")

def discretizar_valor(valor):
    if valor < 0.33:
        return 'Bajo'
    elif valor < 0.66:
        return 'Medio'
    else:
        return 'Alto'

def load_and_preprocess():
    print("Cargando y procesando datos...")
    df = pd.read_csv(DATA_PATH)
    
    df_features = df['TFI'].str.split(',', expand=True).astype(float)
    num_features = df_features.shape[1]
    col_names = [f'Feat_{i}' for i in range(num_features)]
    df_features.columns = col_names
    
    df_final = pd.DataFrame()
    
    for col in df_features.columns:
        df_final[col] = df_features[col].apply(discretizar_valor)
        
    targets = ['ansiedad_predicha', 'depresion_predicha', 'estres_predicho']
    for target in targets:
        if df[target].dtype == object or isinstance(df[target].iloc[0], str):
            df_final[target] = df[target]
        else:
            df_final[target] = df[target].apply(discretizar_valor)

    for col in df_final.columns:
        df_final[col] = df_final[col].astype('category')
        
    return df_final

def main():
    if not os.path.exists(DATA_PATH):
        print(f"Error: No se encuentra en {DATA_PATH}")
        return

    df = load_and_preprocess()
    
    # Red Bayesiana Edges correspondientes a entrenamiento_bn.py
    edges = []
    
    # 1. Ansiedad (GAD-7) -> Feat 0 to 20
    for i in range(0, 21):
        edges.append(('ansiedad_predicha', f'Feat_{i}'))
        
    # 2. Depresión (PHQ-9) -> Feat 21 to 47
    for i in range(21, 48):
        edges.append(('depresion_predicha', f'Feat_{i}'))
        
    # 3. Estrés (PSS-10) -> Feat 48 to 77
    for i in range(48, 78):
        edges.append(('estres_predicho', f'Feat_{i}'))

    # Correlaciones entre enfermedades
    edges.append(('ansiedad_predicha', 'depresion_predicha'))
    edges.append(('estres_predicho', 'ansiedad_predicha'))
    edges.append(('estres_predicho', 'depresion_predicha'))

    print(f"Evaluando {len(edges)} aristas (dependencias) usando Prueba Chi-Cuadrado...")
    
    resultados = []
    significativas = 0
    alpha = 0.05
    
    for nodo_origen, nodo_destino in edges:
        # Crear tabla de contingencia
        contingency_table = pd.crosstab(df[nodo_origen], df[nodo_destino])
        
        # Calcular Chi-Cuadrado
        chi2, p_value, dof, _ = chi2_contingency(contingency_table)
        
        es_significativa = p_value < alpha
        if es_significativa:
            significativas += 1
            
        resultados.append({
            'Origen': nodo_origen,
            'Destino': nodo_destino,
            'Chi2': round(chi2, 4),
            'P-Value': p_value, # Mantener precision para ver e-xx
            'Grados Libertad': dof,
            'Significativa (α < 0.05)': 'Sí' if es_significativa else 'No'
        })

    # Mostrar reporte
    total_aristas = len(edges)
    porcentaje = (significativas / total_aristas) * 100
    resumen = f"De un total de {total_aristas} aristas en la Red Bayesiana, {significativas} ({porcentaje:.2f}%) son estadísticamente significativas (p < {alpha})."
    
    print("\n--- RESUMEN CHI-CUADRADO ---")
    print(resumen)
    print("Las dependencias significativas validan la estructura del modelo propuesto.\n")

    # Guardar reporte en CSV
    df_resultados = pd.DataFrame(resultados)
    
    # Generar un markdown para visualizar la tabla 
    reporte_md = f"""# Test de Dependencia Chi-Cuadrado para Red Bayesiana

**Descripción:** Esta prueba valida las dependencias (aristas) establecidas en el Modelo Bayesiano mediante un test estadístico Chi-Cuadrado para variables categóricas.
**Nivel de significancia (α)**: {alpha}

**Resumen:** {resumen}

## Resultados Detallados
"""
    
    with open(REPORT_OUTPUT, "w", encoding="utf-8") as f:
        f.write(reporte_md)
        # Create markdown table manually
        columns = df_resultados.columns.tolist()
        f.write("| " + " | ".join(columns) + " |\n")
        f.write("|" + "|".join(["---"] * len(columns)) + "|\n")
        
        for index, row in df_resultados.iterrows():
            f.write("| " + " | ".join([str(val) for val in row.values]) + " |\n")
        
    print(f"Reporte detallado escrito en {REPORT_OUTPUT}")

if __name__ == "__main__":
    main()
