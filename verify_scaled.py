
import sys
import os
import pandas as pd
sys.path.append(os.getcwd())
from main import predecir_emociones, RespuestaUsuario

def test_scaled_model():
    print("--- Verifying Scaled Model (100k) ---")
    
    respuestas = []
    
    # CASE 1: High Stress Input (Consistent Profile)
    print("\nCase 1: High Stress Input (All High)")
    for i in range(1, 17): # GAD/PHQ High (Consistent with Stress)
        respuestas.append(RespuestaUsuario(idcuestionario=1, pregunta=i, valor=3))
    for i in range(17, 27): # PSS High
        respuestas.append(RespuestaUsuario(idcuestionario=1, pregunta=i, valor=4))
        
    try:
        results = predecir_emociones(respuestas)
        pred = results[0]
        print(f"Stress Prediction: {pred.estres_predicho}%")
        
        if pred.estres_predicho > 70:
            print("PASS: High Stress detected")
        else:
            print(f"FAIL: Stress too low ({pred.estres_predicho}%)")
            
    except Exception as e:
        print(f"ERROR: {e}")

    # CASE 2: High Anxiety (GAD-7 High)
    print("\nCase 2: High Anxiety Input")
    respuestas_anx = []
    for i in range(1, 8): # GAD High
        respuestas_anx.append(RespuestaUsuario(idcuestionario=2, pregunta=i, valor=3))
    for i in range(8, 27): # Others Low
        respuestas_anx.append(RespuestaUsuario(idcuestionario=2, pregunta=i, valor=0))
        
    try:
        results = predecir_emociones(respuestas_anx)
        pred = results[0]
        print(f"Anxiety Prediction: {pred.ansiedad_predicha}%")
        
        if pred.ansiedad_predicha > 70:
            print("PASS: High Anxiety detected")
        else:
            print(f"FAIL: Anxiety too low ({pred.ansiedad_predicha}%)")
            
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    test_scaled_model()
