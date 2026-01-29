import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, confusion_matrix, roc_curve, auc
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar los datos de entrenamiento
root = r"C:\Users\User\Desktop\Proyecto\Api_emociones\Modelo"
folder = r"C:\Users\User\Desktop\Proyecto\Api_emociones\Modelo\resultados"
df = pd.read_excel(folder + '\\Datos_entrenamiento.xlsx')

# Convertir la columna 'TFI' en una lista de números
df['TFI'] = df['TFI'].apply(lambda x: [float(valor) for valor in x.split(',')])

# Dividir los datos en características (X) y objetivo (y)
X = pd.DataFrame(df['TFI'].tolist())  # Convertir la lista en un DataFrame
y = df[['estrés_predicho', 'ansiedad_predicha', 'depresion_predicha']]  # Definir y

# Entrenar el modelo con los datos cargados
modelo = RandomForestRegressor(random_state=42)

# Realizar validación cruzada (usamos 5 pliegues como ejemplo)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_results = cross_val_score(modelo, X, y, cv=kf, scoring='neg_mean_squared_error')  # Usamos MSE negativo para que sea más fácil de interpretar

# Mostrar los resultados de la validación cruzada
print(f"Resultados de la validación cruzada (MSE negativo): {cv_results}")
print(f"Promedio de la validación cruzada: {cv_results.mean()}")

# Ajustar el modelo con los datos completos
modelo.fit(X, y)

# Obtener las predicciones del modelo
y_pred = modelo.predict(X)  # Definir y_pred

# Convertir las predicciones continuas a clases (por ejemplo, umbral de 0.5)
umbral = 0.5
y_pred_classes = (y_pred >= umbral).astype(int)  # Convertir a 1 para valores >= 0.5 y 0 para < 0.5
y_true_classes = (y.values >= umbral).astype(int)  # Convertir a 1 para valores >= 0.5 y 0 para < 0.5

# Calcular y mostrar la matriz de confusión para cada emoción
for i, emocion in enumerate(y.columns):
    print(f"\nMatriz de Confusión para {emocion}:")
    cm = confusion_matrix(y_true_classes[:, i], y_pred_classes[:, i])
    
    # Graficar la matriz de confusión
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Bajo', 'Alto'], yticklabels=['Bajo', 'Alto'])
    plt.title(f'Matriz de Confusión para {emocion}')
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    plt.show()

# Calcular la Curva ROC y el AUC para cada emoción
for i, emocion in enumerate(y.columns):
    print(f"\nCurva ROC para {emocion}:")
    
    # Obtener las probabilidades predichas para la emoción
    fpr, tpr, thresholds = roc_curve(y_true_classes[:, i], y_pred[:, i])  # Probabilidades vs clases
    auc_value = auc(fpr, tpr)  # Calcular el área bajo la curva (AUC)
    
    # Graficar la curva ROC
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', label=f'AUC = {auc_value:.2f}')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Línea diagonal de aleatoriedad
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title(f'Curva ROC para {emocion}')
    plt.legend(loc='lower right')
    plt.show()

# Realizar una predicción con nuevos datos
nuevos_datos = [[0,1,0,0.6,0.3,0.1,0.4,0.4,0.2,1,0,0,0.3,0.6,0.1,0,1,0,0,1,0,0,1,0,1,0,0,1,0,0,1,0,0,1,0,0]]  # Ejemplo de nuevo dato (estrés, ansiedad, depresión)
predicción = modelo.predict(nuevos_datos)

# Mostrar la predicción
print(f"Predicción para los nuevos datos: {predicción}")

# Guardar el modelo entrenado (opcional)
# joblib.dump(modelo, root + '\\modelo_emocional.pkl')
