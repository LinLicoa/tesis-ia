# Archivo de configuración (opcional)
# Aquí puedes poner configuraciones como parámetros del modelo, rutas de archivos, etc.

CONFIG = {
    'path_datos_entrada': '..\\Modelo\\cuestionarios_respuestas_1000.xlsx',
    'path_resultados': 'resultados_emocionales.xlsx',
    'path_modelo_guardado': 'modelo_emocional.pkl',
    'caracteristicas': ['estrés', 'ansiedad', 'depresión'],  # Las características que usarás para entrenar
    'etiqueta': 'estrés'  # Cambia esto a 'ansiedad' o 'depresión' si deseas predecir una de esas emociones
}
