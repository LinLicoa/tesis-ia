import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Rutas
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ROOT_PATH, "Modelo", "datos", "Datos_entrenamiento.csv")
REPORT_IMAGE = os.path.join(ROOT_PATH, "_Reportes_Tesis", "matriz_correlacion.png")
REPORT_TEXT = os.path.join(ROOT_PATH, "_Reportes_Tesis", "informe_correlacion.txt")

def generar_demostracion():
    print(f"Leyendo dataset desde: {DATA_PATH}")
    if not os.path.exists(DATA_PATH):
        print("El Dataset no existe. Ejecuta primero generar_dataset_clinico.py.")
        return

    df = pd.read_csv(DATA_PATH)
    
    # 1. Extraemos solo los puntajes puros de cada cuestionario
    df_scores = df[['score_gad', 'score_phq', 'score_pss']]
    
    # Renombramos para que el gráfico se vea profesional
    df_scores.columns = ['Ansiedad (GAD-7)', 'Depresión (PHQ-9)', 'Estrés (PSS-10)']

    # 2. Calculamos la matriz de correlación de Pearson
    print("Calculando correlación de Pearson...")
    correlation_matrix = df_scores.corr()
    
    # 3. Guardar el Informe de Texto
    with open(REPORT_TEXT, "w", encoding="utf-8") as f:
        f.write("=== REPORTE DE CORRELACIÓN CLÍNICA (Tesis IA) ===\n\n")
        f.write("Matriz de Correlación de Pearson entre Cuestionarios:\n")
        f.write(correlation_matrix.to_string())
        f.write("\n\nInterpretación:\n")
        f.write("- Valores cercanos a 1.0 indican una fuerte correlación positiva.\n")
        f.write("- Esto demuestra que la generación del dataset (perfiles) funciona ")
        f.write("conectando intrínsecamente los padecimientos, tal como se indicó a la Red Bayesiana.\n")

    print(f"Informe de texto guardado en: {REPORT_TEXT}")

    # 4. Generar el Mapa de Calor (Heatmap)
    print("Generando Gráfico (Mapa de Calor)...")
    plt.figure(figsize=(8, 6))
    
    # Configuramos seaborn
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", 
                linewidths=1, vmin=0, vmax=1)
                
    plt.title('Matriz de Correlación entre Cuestionarios Psicológicos', pad=20)
    plt.tight_layout()
    
    # Guardar la imagen
    plt.savefig(REPORT_IMAGE, dpi=300)
    print(f"Gráfico guardado exitosamente en: {REPORT_IMAGE}")

if __name__ == "__main__":
    generar_demostracion()
