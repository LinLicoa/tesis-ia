import pandas as pd
import numpy as np
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
import joblib
import os

# Rutas
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
RESULTADOS_PATH = os.path.join(ROOT_PATH, "datos")
DATA_PATH = os.path.join(RESULTADOS_PATH, "Datos_entrenamiento.csv")
MODEL_OUTPUT = os.path.join(ROOT_PATH, "modelo_bn.pkl")

def discretizar_valor(valor):
    """Convierte un valor continuo 0-1 en categorías discretas."""
    if valor < 0.33:
        return 'Bajo'
    elif valor < 0.66:
        return 'Medio'
    else:
        return 'Alto'

def main():
    print("Iniciando entrenamiento OPTIMIZADO de Red Bayesiana...")
    
    if not os.path.exists(DATA_PATH):
        print(f"Error: No se encuentra en {DATA_PATH}")
        return

    print("Cargando datos...")
    df = pd.read_csv(DATA_PATH)
    
    # --- Preprocesamiento ---
    # --- Preprocesamiento ---
    print("Procesando features en batch...")
    # OPTIMIZACION: Usar operaciones vectorizadas de Pandas en lugar de apply fila por fila
    # Esto es mucho más rápido para 1M de registros
    
    # 1. Split string "0.9,0.1,..." into columns
    print("Desglosando scores (split)...")
    # Generamos directamente el dataframe de features
    df_features = df['TFI'].str.split(',', expand=True).astype(float)
    
    # Generar nombres de features
    num_features = df_features.shape[1]
    col_names = [f'Feat_{i}' for i in range(num_features)]
    df_features.columns = col_names
    
    df_final = pd.DataFrame()
    
    # Feature Discretization
    for col in df_features.columns:
        df_final[col] = df_features[col].apply(discretizar_valor)
        
    # Target Processing
    targets = ['ansiedad_predicha', 'depresion_predicha', 'estres_predicho'] # Orden nmemónico
    for target in targets:
        if df[target].dtype == object or isinstance(df[target].iloc[0], str):
             df_final[target] = df[target]
        else:
             df_final[target] = df[target].apply(discretizar_valor)

    print(f"Datos listos. Dimensiones: {df_final.shape}")

    # --- Definición de Estructura (Expert Knowledge) ---
    # En lugar de buscar (lento), definimos las dependencias clínicas conocidas.
    # Latent Variable Model: La enfermedad (Ansiedad) causa los síntomas (Respuestas GAD).
    
    edges = []
    
    # 1. Ansiedad (GAD-7) -> Feat 0 to 20
    # GAD has 7 items. Each item has 3 components (T,F,I). 
    # Indices: 0 to 20
    for i in range(0, 21):
        edges.append(('ansiedad_predicha', f'Feat_{i}'))
        
    # 2. Depresión (PHQ-9) -> Feat 21 to 47
    # 21 + 27 = 48
    for i in range(21, 48):
        edges.append(('depresion_predicha', f'Feat_{i}'))
        
    # 3. Estrés (PSS-10) -> Feat 48 to 77
    # 48 + 30 = 78
    for i in range(48, 78):
        edges.append(('estres_predicho', f'Feat_{i}'))

    # Opcional: Correlaciones entre enfermedades
    edges.append(('ansiedad_predicha', 'depresion_predicha'))
    edges.append(('estres_predicho', 'ansiedad_predicha'))
    edges.append(('estres_predicho', 'depresion_predicha'))

    print(f"Definiendo estructura con {len(edges)} aristas...")
    model = DiscreteBayesianNetwork(edges)
    
    # --- Entrenamiento de Parámetros ---
    print("Entrenando parámetros (Maximum Likelihood)...")
    # State names deben ser consistentes
    model.fit(df_final, estimator=MaximumLikelihoodEstimator)
    
    print("Validando modelo...")
    assert model.check_model()
    
    print(f"Guardando modelo en {MODEL_OUTPUT}...")
    joblib.dump(model, MODEL_OUTPUT)
    print("¡Entrenamiento finalizado y modelo guardado!")

if __name__ == "__main__":
    main()
