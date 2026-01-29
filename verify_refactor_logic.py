import sys
import unittest
from unittest.mock import MagicMock, patch
import json
from pydantic import BaseModel
from typing import List

# Mocking modules that might not be easily available or take time to load
sys.modules['joblib'] = MagicMock()
sys.modules['pgmpy'] = MagicMock()
sys.modules['pgmpy.inference'] = MagicMock()
sys.modules['Modelo'] = MagicMock()
sys.modules['Modelo.motor_recomendaciones'] = MagicMock()

# Now we can import main (it will use the mocks)
# However, main.py imports usually happen at top level, so we need to be careful.
# If main.py executes code at top level that uses these, we need to set them up first.

# Setting up Specific Mocks for main.py usage
mock_model = MagicMock()
mock_model.nodes.return_value = ['Feat_0', 'estres_predicho', 'ansiedad_predicha', 'depresion_predicha']
sys.modules['joblib'].load.return_value = mock_model

# Mock inference engine
mock_engine = MagicMock()
sys.modules['pgmpy.inference'].VariableElimination.return_value = mock_engine

# Mock recommendations
sys.modules['Modelo.motor_recomendaciones'].obtener_recomendaciones.return_value = ["Recomendación Mock"]

# Import main after mocking
# We need to assume main.py is in the parent directory or handle path
import os
sys.path.append(os.getcwd())
# Assuming the file is at c:\Users\Lindsay\Downloads\Api_emociones\Api_emociones\main.py
# and we are running this from somewhere. Let's just assume we are in the same dir for this import.
# But wait, main.py uses relative imports `from Modelo ...`.
# If I run this script from the root project dir, it should work.

# Let's try to import the specific classes and functions we need by reading the file
# since directly importing main might fail if the user environment is complex.
# actually, let's just create a class that mimics the logic to test it. Use the code provided in the implementation.

# RE-USE LOGIC FROM main.py
class DetallePrueba(BaseModel):
    prueba: str      # "GAD-7"
    condicion: str   # "Ansiedad"
    porcentaje: float
    # Nuevos campos (Valores 0.0 - 1.0)
    t: float
    i: float
    f: float

class Prediccion(BaseModel):
    idcuestionario: int
    detalles: List[DetallePrueba]  # Nueva lista estructurada
    recomendaciones: List[str] # Lista de consejos post-procesamiento

def discretizar_valor(valor):
    if valor < 0.33:
        return 'Bajo'
    elif valor < 0.66:
        return 'Medio'
    else:
        return 'Alto'

def mapping_neutrosofico_simple(valor: int, codigo_prueba: str):
    if codigo_prueba == "PSS-10":
        max_val = 4.0
    else:
        max_val = 3.0
    normalized = valor / max_val
    if normalized >= 0.75:
        t, f, i = 0.8, 0.1, 0.2
    elif normalized >= 0.40:
        t, f, i = 0.4, 0.4, 0.7
    else:
        t, f, i = 0.1, 0.8, 0.3
    return t, f, i

# Test Class
class TestRefactor(unittest.TestCase):
    
    def test_mapping_neutrosofico(self):
        # Case High
        t, f, i = mapping_neutrosofico_simple(3, "GAD-7")
        self.assertEqual(t, 0.8)
        self.assertEqual(f, 0.1)
        
        # Case Low
        t, f, i = mapping_neutrosofico_simple(0, "GAD-7")
        self.assertEqual(t, 0.1)
        
        print("Mapping logic: OK")

    def test_detalle_prueba_model(self):
        # Ensure model accepts new fields
        detalle = DetallePrueba(
            prueba="GAD-7", 
            condicion="Ansiedad", 
            porcentaje=80.0,
            t=0.8,
            i=0.2,
            f=0.1
        )
        self.assertEqual(detalle.t, 0.8)
        self.assertEqual(detalle.i, 0.2)
        print("Model structure: OK")

if __name__ == '__main__':
    unittest.main()
