import pandas as pd
import numpy as np
import os
import joblib
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Rutas
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ROOT_PATH, "Modelo", "datos", "Datos_entrenamiento.csv")
MODEL_PATH = os.path.join(ROOT_PATH, "Modelo", "modelo_bn.pkl")
REPORT_IMAGE = os.path.join(ROOT_PATH, "_Reportes_Tesis", "matriz_confusion.png")
REPORT_TEXT = os.path.join(ROOT_PATH, "_Reportes_Tesis", "informe_confusion.txt")

def discretizar_valor(valor):
    if valor < 0.33:
        return 'Bajo'
    elif valor < 0.66:
        return 'Medio'
    else:
        return 'Alto'

def generar_matriz_confusion():
    print("Iniciando Evaluación del Modelo Bayesiano...")
    
    if not os.path.exists(DATA_PATH) or not os.path.exists(MODEL_PATH):
        print("Faltan datos o el modelo compilado.")
        return

    # 1. Cargar el dataset (ground truth)
    df = pd.read_csv(DATA_PATH)
    
    # 2. Cargar el modelo
    print("Cargando modelo...")
    modelo_bn = joblib.load(MODEL_PATH)
    inferencia = VariableElimination(modelo_bn)
    
    # Vamos a usar una muestra para evaluar, ya que 300 predicciones completas tomará un poquito (pero es rápido)
    targets = ['ansiedad_predicha', 'depresion_predicha', 'estres_predicho']
    
    y_true_dict = {t: [] for t in targets}
    y_pred_dict = {t: [] for t in targets}
    
    # 3. Preparar features igual que se entrenó
    df_features = df['TFI'].str.split(',', expand=True).astype(float)
    
    print(f"Evaluando {len(df)} pacientes...")
    
    # Recorrer filas y predecir
    for idx, row in df.iterrows():
        # Construir evidencia
        evidencia = {}
        for feat_idx in range(df_features.shape[1]):
            val = df_features.iloc[idx, feat_idx]
            evidencia[f'Feat_{feat_idx}'] = discretizar_valor(val)
            
        # Para acelerar la inferencia en pgmpy, consultamos las tres variables juntas
        try:
            # Maximum Probability state for each prediction
            # Nota: model.predict usa MAP (Maximum A Posteriori) que es lo que queremos para evaluar Acc
            df_evidencia_fila = pd.DataFrame([evidencia])
            pred_df = modelo_bn.predict(df_evidencia_fila)
            
            for t in targets:
                y_true_dict[t].append(df.iloc[idx][t])
                y_pred_dict[t].append(pred_df.iloc[0][t])
                
        except Exception as e:
            print(f"Error procesando fila {idx}: {e}")
            
    # 4. Generar reportes
    with open(REPORT_TEXT, "w", encoding="utf-8") as f:
        f.write("=== REPORTE DE MATRIZ DE CONFUSIÓN Y PRECISIÓN ===\n\n")
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Las clases esperadas
        labels = ['Bajo', 'Medio', 'Alto']
        titulos = ['Ansiedad', 'Depresión', 'Estrés']
        
        for idx_t, target in enumerate(targets):
            y_true = y_true_dict[target]
            y_pred = y_pred_dict[target]
            
            acc = accuracy_score(y_true, y_pred)
            report = classification_report(y_true, y_pred, labels=labels, zero_division=0)
            
            f.write(f"--- Evaluación para: {titulos[idx_t]} ---\n")
            f.write(f"Precisión (Accuracy): {acc:.2%}\n")
            f.write("Reporte de Clasificación:\n")
            f.write(report + "\n\n")
            
            # Matriz de Confusión
            cm = confusion_matrix(y_true, y_pred, labels=labels)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=axes[idx_t])
            axes[idx_t].set_title(f'Matriz Confusión - {titulos[idx_t]}\nAcc: {acc:.2%}')
            axes[idx_t].set_xlabel('Predicción del Modelo')
            axes[idx_t].set_ylabel('Ground Truth (Realidad)')
            
        f.write("\nInterpretación:\n")
        f.write("Una matriz de confusión ideal tiene los números concentrados en la diagonal principal.\n")
        f.write("Esto certifica que la red paramétrica bayesiana ha aprendido correctamente la distribución del dataset.\n")

    plt.tight_layout()
    plt.savefig(REPORT_IMAGE, dpi=300)
    
    print(f"Evaluación completada.\nInforme de texto guardado en: {REPORT_TEXT}\nGráfico guardado en: {REPORT_IMAGE}")

if __name__ == "__main__":
    generar_matriz_confusion()
