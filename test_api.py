import requests
import json

payload = [
  {"idcuestionario": 1, "pregunta": 1, "valor": 3, "codigo_prueba": "GAD-7"},
  {"idcuestionario": 1, "pregunta": 2, "valor": 2, "codigo_prueba": "GAD-7"},
  {"idcuestionario": 1, "pregunta": 3, "valor": 3, "codigo_prueba": "GAD-7"},
  {"idcuestionario": 1, "pregunta": 4, "valor": 2, "codigo_prueba": "GAD-7"},
  {"idcuestionario": 1, "pregunta": 5, "valor": 3, "codigo_prueba": "GAD-7"},
  {"idcuestionario": 1, "pregunta": 6, "valor": 3, "codigo_prueba": "GAD-7"},
  {"idcuestionario": 1, "pregunta": 7, "valor": 1, "codigo_prueba": "GAD-7"},
  {"idcuestionario": 1, "pregunta": 8, "valor": 2, "codigo_prueba": "PHQ-9"},
  {"idcuestionario": 1, "pregunta": 9, "valor": 3, "codigo_prueba": "PHQ-9"},
  {"idcuestionario": 1, "pregunta": 10, "valor": 2, "codigo_prueba": "PHQ-9"},
  {"idcuestionario": 1, "pregunta": 11, "valor": 1, "codigo_prueba": "PHQ-9"},
  {"idcuestionario": 1, "pregunta": 12, "valor": 2, "codigo_prueba": "PHQ-9"},
  {"idcuestionario": 1, "pregunta": 13, "valor": 3, "codigo_prueba": "PHQ-9"},
  {"idcuestionario": 1, "pregunta": 14, "valor": 1, "codigo_prueba": "PHQ-9"},
  {"idcuestionario": 1, "pregunta": 15, "valor": 2, "codigo_prueba": "PHQ-9"},
  {"idcuestionario": 1, "pregunta": 16, "valor": 1, "codigo_prueba": "PHQ-9"},
  {"idcuestionario": 1, "pregunta": 17, "valor": 3, "codigo_prueba": "PSS-10"},
  {"idcuestionario": 1, "pregunta": 18, "valor": 4, "codigo_prueba": "PSS-10"},
  {"idcuestionario": 1, "pregunta": 19, "valor": 2, "codigo_prueba": "PSS-10"},
  {"idcuestionario": 1, "pregunta": 20, "valor": 3, "codigo_prueba": "PSS-10"},
  {"idcuestionario": 1, "pregunta": 21, "valor": 4, "codigo_prueba": "PSS-10"},
  {"idcuestionario": 1, "pregunta": 22, "valor": 1, "codigo_prueba": "PSS-10"},
  {"idcuestionario": 1, "pregunta": 23, "valor": 4, "codigo_prueba": "PSS-10"},
  {"idcuestionario": 1, "pregunta": 24, "valor": 3, "codigo_prueba": "PSS-10"},
  {"idcuestionario": 1, "pregunta": 25, "valor": 2, "codigo_prueba": "PSS-10"},
  {"idcuestionario": 1, "pregunta": 26, "valor": 4, "codigo_prueba": "PSS-10"}
]

try:
    res = requests.post('http://127.0.0.1:8000/predecir', json=payload)
    if res.ok:
        with open('output.json', 'w') as f:
            json.dump(res.json(), f, indent=2)
        print("Guardado ok")
    else:
        print(f"Error {res.status_code}: {res.text}")
except Exception as e:
    print(e)
