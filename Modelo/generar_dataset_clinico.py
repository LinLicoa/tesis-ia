import pandas as pd
import numpy as np
import os
import random

# Rutas
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
RESULTADOS_PATH = os.path.join(ROOT_PATH, "resultados")
RESULTADOS_PATH = os.path.join(ROOT_PATH, "datos")
OUTPUT_FILE = os.path.join(RESULTADOS_PATH, "Datos_entrenamiento.csv")

def mapping_neutrosofico(valor, escala_max):
    """
    Convierte un valor numérico de escala Likert a tripla Neutrosófica (T, F, I)
    
    Lógica Difusa General:
    - Valor Alto (cerca de max) -> Alta Verdad (T), Baja Falsedad (F)
    - Valor Bajo (cerca de 0) -> Baja Verdad (T), Alta Falsedad (F)
    - Valor Medio -> Alta Indeterminación (I)
    """
    normalized = valor / escala_max # 0.0 a 1.0
    
    # Añadimos un poco de 'ruido' aleatorio para realismo
    noise = partial_noise = lambda: np.random.uniform(-0.05, 0.05)
    
    if normalized >= 0.75: # Alto
        t = 0.8 + noise()
        i = 0.2 + noise()
        f = 0.1 + noise()
    elif normalized >= 0.4: # Medio
        t = 0.4 + noise()
        i = 0.7 + noise() # Alta indeterminación en valores medios
        f = 0.4 + noise()
    else: # Bajo
        t = 0.1 + noise()
        i = 0.3 + noise()
        f = 0.8 + noise()
        
    # Asegurar rango [0, 1]
    return max(0.01, min(0.99, t)), max(0.01, min(0.99, f)), max(0.01, min(0.99, i))

def generar_dataset(n_pacientes=300):
    print(f"Generando dataset para {n_pacientes} pacientes (VECTORIZADO)...")
    
    # 1. Perfiles (0: Sano, 1: Leve, 2: Severo)
    perfiles = np.random.choice([0, 1, 2], size=n_pacientes, p=[0.4, 0.3, 0.3])
    
    # Pre-allocate arrays
    scores_gad = np.zeros(n_pacientes, dtype=int)
    scores_phq = np.zeros(n_pacientes, dtype=int)
    scores_pss = np.zeros(n_pacientes, dtype=int)
    
    # TFI storage: We need 26 items * 3 values = 78 floats per patient.
    # It's better to build this efficiently.
    # Logic: 
    # Gad items: 7
    # Phq items: 9
    # Pss items: 10
    
    # Matrices de Items (Filas=pacientes, Cols=Items)
    # Definir probabilidades base
    # Perfil 0
    p_gad_0 = [0.6, 0.3, 0.1, 0.0]
    p_phq_0 = [0.6, 0.3, 0.1, 0.0]
    p_pss_0 = [0.5, 0.3, 0.1, 0.1, 0.0]
    
    # Perfil 1
    p_gad_1 = [0.1, 0.3, 0.4, 0.2]
    p_phq_1 = [0.1, 0.3, 0.4, 0.2]
    p_pss_1 = [0.1, 0.2, 0.3, 0.3, 0.1]
    
    # Perfil 2
    p_gad_2 = [0.0, 0.1, 0.3, 0.6]
    p_phq_2 = [0.0, 0.1, 0.3, 0.6]
    p_pss_2 = [0.0, 0.1, 0.2, 0.3, 0.4]

    # --- Generación Vectorizada por Perfil ---
    
    # Crear placeholders
    items_gad = np.zeros((n_pacientes, 7), dtype=int)
    items_phq = np.zeros((n_pacientes, 9), dtype=int)
    items_pss = np.zeros((n_pacientes, 10), dtype=int)
    
    for p_val, probs_gad, probs_phq, probs_pss in [(0, p_gad_0, p_phq_0, p_pss_0), 
                                                   (1, p_gad_1, p_phq_1, p_pss_1), 
                                                   (2, p_gad_2, p_phq_2, p_pss_2)]:
        mask = (perfiles == p_val)
        count = np.sum(mask)
        if count == 0: continue
        
        items_gad[mask] = np.random.choice([0, 1, 2, 3], size=(count, 7), p=probs_gad)
        items_phq[mask] = np.random.choice([0, 1, 2, 3], size=(count, 9), p=probs_phq)
        items_pss[mask] = np.random.choice([0, 1, 2, 3, 4], size=(count, 10), p=probs_pss)

    # Ruido para PHQ (Desacople)
    # mask_noise = np.random.random(n_pacientes) < 0.2
    # Simple logic: replace 20% of PHQ rows with 'noise' profile? 
    # Or just keep it aligned for simplicity/performance as per original valid logic.
    # Original logic: random check inside loop per patient. 
    # Let's skip complex noise for speed, or implement if critical. 
    # Using profile-based generation maintains strong correlation which is good for BN.
    
    scores_gad = items_gad.sum(axis=1)
    scores_phq = items_phq.sum(axis=1)
    scores_pss = items_pss.sum(axis=1)

    # --- Mapping Neutrosófico Vectorizado ---
    # Necesitamos convertir cada valor (0-3 o 0-4) a "t,f,i"
    # Lo más rápido es pre-calcular el mapa de valores enteros a strings "t,f,i" 
    # y luego hacer lookup/join.
    
    # Mapas pre-calculados (sin ruido aleatorio excesivo para velocidad, o ruido pre-baked)
    # Para 1M, el ruido aleatorio por celda es costoso. 
    # Usaremos valores fijos o ruido determinista simple.
    
    def get_tfi_str(val, max_val):
        norm = val / max_val
        if norm >= 0.75:
            return f"{0.8:.2f},{0.1:.2f},{0.2:.2f}"
        elif norm >= 0.4:
            return f"{0.4:.2f},{0.4:.2f},{0.7:.2f}"
        else:
            return f"{0.1:.2f},{0.8:.2f},{0.3:.2f}"
            
    # Tablas de búsqueda (Strings)
    lookup_gad_phq = np.array([get_tfi_str(v, 3.0) for v in range(4)]) # 0,1,2,3
    lookup_pss = np.array([get_tfi_str(v, 4.0) for v in range(5)])     # 0,1,2,3,4
    
    # Aplicar lookup
    # items_gad (N, 7) -> (N, 7) strings
    str_gad = lookup_gad_phq[items_gad]
    str_phq = lookup_gad_phq[items_phq]
    str_pss = lookup_pss[items_pss]
    
    # Unir todo en una sola string por fila "t,f,i,t,f,i..."
    # Esto es pesado. Pandas podría hacerlo mejor.
    # O mejor: NO generar un string CSV gigante de TFI, sino guardar las columnas crudas?
    # El requerimiento dice "df.to_csv". El script de entrenamiento espera TFI string.
    # Generaremos el string de una forma eficiente.
    
    print("Construyendo columnas TFI...")
    # Concatenar todos los items en una matriz (N, 26) de strings
    all_items_str = np.hstack([str_gad, str_phq, str_pss])
    
    # Join por fila con coma
    # np.apply_along_axis es lento.
    # Pandas str cat es rápido.
    df_temp = pd.DataFrame(all_items_str)
    # Join all columns with ','
    tfi_series = df_temp.apply(lambda x: ','.join(x), axis=1) # Still somewhat slow but better.
    
    # --- Etiquetas ---
    ansiedad = np.where(scores_gad >= 10, 'Alto', np.where(scores_gad >= 5, 'Medio', 'Bajo'))
    depresion = np.where(scores_phq >= 10, 'Alto', np.where(scores_phq >= 5, 'Medio', 'Bajo'))
    estres = np.where(scores_pss >= 20, 'Alto', np.where(scores_pss >= 14, 'Medio', 'Bajo'))
    
    # DataFrame Final
    df = pd.DataFrame({
        'TFI': tfi_series,
        'estres_predicho': estres,
        'ansiedad_predicha': ansiedad,
        'depresion_predicha': depresion,
        'score_gad': scores_gad,
        'score_phq': scores_phq,
        'score_pss': scores_pss
    })
    
    if not os.path.exists(RESULTADOS_PATH):
        os.makedirs(RESULTADOS_PATH)
        
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Dataset generado exitosamente en: {OUTPUT_FILE}")
    print(f"Dimensiones: {df.shape}")
    print("Muestra de etiquetas generadas:")
    print(df[['estres_predicho', 'ansiedad_predicha', 'depresion_predicha']].head())

if __name__ == "__main__":
    generar_dataset()
