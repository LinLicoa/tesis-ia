import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pgmpy.inference import VariableElimination
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelBinarizer

# Rutas
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
RESULTADOS_PATH = os.path.join(ROOT_PATH, "resultados")
DATA_PATH = os.path.join(RESULTADOS_PATH, "Datos_entrenamiento.xlsx")
MODEL_PATH = os.path.join(ROOT_PATH, "modelo_bn.pkl")

def discretizar_valor(valor):
    if valor < 0.33:
        return 'Bajo'
    elif valor < 0.66:
        return 'Medio'
    else:
        return 'Alto'

def map_label_to_int(label):
    if label == 'Bajo': return 0
    if label == 'Medio': return 1
    if label == 'Alto': return 2
    return -1

def main():
    print("Iniciando evaluación de Red Bayesiana...")
    
    # 1. Cargar Modelo y Datos
    if not os.path.exists(MODEL_PATH):
        print("Error: No se encuentra el modelo entrenado.")
        return
        
    model = joblib.load(MODEL_PATH)
    infer = VariableElimination(model)
    
    df = pd.read_excel(DATA_PATH)
    
    # Procesar características igual que en el entrenamiento
    tfi_data = df['TFI'].apply(lambda x: [float(v) for v in x.split(',')]).tolist()
    col_names = [f'Feat_{i}' for i in range(len(tfi_data[0]))]
    df_features = pd.DataFrame(tfi_data, columns=col_names)
    
    # Discretizar entradas
    df_input = df_features.applymap(discretizar_valor)
    
    # 2. Realizar Predicciones
    # Esto puede ser lento fila por fila, pgmpy no es tan rápido como sklearn
    print("Realizando predicciones (esto puede tardar unos momentos)...")
    
    targets = ['estrés_predicho', 'ansiedad_predicha', 'depresion_predicha']
    y_true_all = []
    y_pred_all = []
    
    # Tomamos una muestra para no tardar mucho si el dataset es gigante
    sample_size = min(100, len(df)) 
    print(f"Evaluando en una muestra de {sample_size} registros...")
    
    indices = np.random.choice(len(df), sample_size, replace=False)
    
    for idx in indices:
        # Obtener valores reales
        row_true = {}
        for target in targets:
            val_true = discretizar_valor(df.iloc[idx][target])
            row_true[target] = val_true
            
        # Preparar evidencia (inputs)
        evidence = df_input.iloc[idx].to_dict()
        # Solo usar evidencia que esté en el modelo (nodes)
        evidence = {k: v for k, v in evidence.items() if k in model.nodes()}
        
        try:
            # Predecir para cada target
            # map_query devuelve el estado más probable
            prediction = infer.map_query(variables=targets, evidence=evidence, show_progress=False)
            
            y_true_all.append(row_true)
            y_pred_all.append(prediction)
            
        except Exception as e:
            print(f"Error en índice {idx}: {e}")

    # 3. Métricas y Visualización
    labels = ['Bajo', 'Medio', 'Alto']
    
    for target in targets:
        y_true = [item[target] for item in y_true_all]
        y_pred = [item[target] for item in y_pred_all]
        
        acc = accuracy_score(y_true, y_pred)
        print(f"\n--- {target} ---")
        print(f"Accuracy: {acc:.2f}")
        
        # Matriz de Confusión
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.title(f'Matriz de Confusión - {target} (BN)')
        plt.xlabel('Predicción')
        plt.ylabel('Real')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
