from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
import joblib
import os
from dotenv import load_dotenv

# Cargar variables de entorno al iniciar
load_dotenv()

from pgmpy.models import DiscreteBayesianNetwork  # noqa: F401 — needed for joblib deserialization
from pgmpy.inference import VariableElimination
from fastapi.middleware.cors import CORSMiddleware
from Modelo.motor_recomendaciones import obtener_recomendaciones

# --- CONFIGURACIÓN Y CARGA DEL MODELO ---
# Usamos rutas relativas para mayor robustez
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
# Asegúrate de que este nombre coincida con tu archivo generado en entrenamiento_bn.py
# CORRECCIÓN: El modelo está dentro de la carpeta "Modelo"
MODEL_PATH = os.path.join(ROOT_PATH, "Modelo", "modelo_bn.pkl")

# Variables globales para el modelo
modelo_bn = None
inferencia_engine = None

try:
    print(f"Cargando modelo desde: {MODEL_PATH}")
    modelo_bn = joblib.load(MODEL_PATH)
    # Inicializamos el motor de inferencia una sola vez para mejorar rendimiento
    inferencia_engine = VariableElimination(modelo_bn)
    print("¡Modelo Bayesiano cargado y motor de inferencia listo!")
except Exception as e:
    print(f"ERROR CRÍTICO: No se pudo cargar el modelo. {e}")
    # No detenemos la app, pero las predicciones fallarán si no se arregla

# Crear la aplicación FastAPI
app = FastAPI(title="API de Predicción Emocional Neutrosófica", version="2.0")

# CORS: acepta localhost para desarrollo + orígenes de producción vía variable de entorno
allowed_origins = ["http://localhost:4200"]
extra_origins = os.environ.get("CORS_ORIGINS", "")
if extra_origins:
    allowed_origins.extend([o.strip() for o in extra_origins.split(",")])

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": modelo_bn is not None}

# --- MODELOS DE DATOS (Pydantic) ---
# --- MODELOS DE DATOS (Pydantic) ---
class RespuestaUsuario(BaseModel):
    idcuestionario: int
    pregunta: int  # 1-26
    valor: int     # Respuesta del usuario (0-3 o 0-4)
    codigo_prueba: str  # Ejemplo: "GAD-7", "PHQ-9", "PSS-10"

class DetallePrueba(BaseModel):
    prueba: str      # "GAD-7"
    condicion: str   # "Ansiedad"
    porcentaje: float
    # Valores POST Red Bayesiana (0.0 - 1.0)
    t: float
    i: float
    f: float
    # Valores PRE Red Bayesiana (Promedios crudos de las respuestas mapeadas)
    t_bruto: float = None
    i_bruto: float = None
    f_bruto: float = None

class Prediccion(BaseModel):
    idcuestionario: int
    detalles: List[DetallePrueba]  # Nueva lista estructurada
    recomendaciones: List[str] # Lista de consejos post-procesamiento

# --- FUNCIONES AUXILIARES ---
def discretizar_valor(valor):
    """Convierte valor continuo (0-1) a discreto para la Red Bayesiana."""
    if valor < 0.33:
        return 'Bajo'
    elif valor < 0.66:
        return 'Medio'
    else:
        return 'Alto'

def mapping_neutrosofico_simple(valor: int, codigo_prueba: str):
    """
    Simula la lógica 'Smart Backend' convirtiendo valor numérico a tripleta neutrosófica.
    Basado en lógica determinista del dataset de entrenamiento.
    """
    # 1. Determinar escala máxima
    # Si codigo_prueba es "PSS-10" -> max_scale = 4.0
    # Si es "GAD-7" o "PHQ-9" -> max_scale = 3.0
    if codigo_prueba == "PSS-10":
        max_val = 4.0
    else:
        # GAD-7, PHQ-9 (default)
        max_val = 3.0
        
    # 2. Normalizar
    normalized = valor / max_val
    
    # 3. Mapping Determinista (Thresholds de entrenamiento)
    # Lógica: >= 0.75 (Alto), >= 0.40 (Medio), < 0.40 (Bajo)
    if normalized >= 0.75:
        # Alto: Alta Verdad, Baja Falsedad
        t, f, i = 0.8, 0.1, 0.2
    elif normalized >= 0.40:
        # Medio: Verdad Media, Alta Indeterminación
        t, f, i = 0.4, 0.4, 0.7
    else:
        # Bajo: Baja Verdad, Alta Falsedad
        t, f, i = 0.1, 0.8, 0.3
        
    return t, f, i

@app.post("/predecir", response_model=List[Prediccion])
def predecir_emociones(respuestas: List[RespuestaUsuario]):
    """
    Recibe JSON con respuestas simples (pregunta, valor).
    Realiza mapping neutrosófico interno y consulta la Red Bayesiana.
    """
    if not inferencia_engine:
        raise HTTPException(status_code=500, detail="El modelo no está cargado en el servidor.")

    try:
        if not respuestas:
             raise HTTPException(status_code=400, detail="La lista de respuestas está vacía.")

        # 1. Convertir entrada a DataFrame intermedio
        # Necesitamos transformar (valor) -> (t, f, i)
        data_procesada = []
        for r in respuestas:
            t, f, i = mapping_neutrosofico_simple(r.valor, r.codigo_prueba)
            data_procesada.append({
                'idcuestionario': r.idcuestionario,
                'pregunta': r.pregunta,
                'codigo_prueba': r.codigo_prueba,
                't': t,
                'f': f,
                'i': i
            })
            
        df = pd.DataFrame(data_procesada)
        
        resultados_prediccion = []
        
        # 2. Procesar por cada Cuestionario (Paciente)
        grupos = df.groupby('idcuestionario')
        
        for id_cuestionario, grupo in grupos:
            # Validar que tengamos las 26 preguntas
            if len(grupo) != 26:
                print(f"Advertencia: Cuestionario {id_cuestionario} incompleto ({len(grupo)} preguntas). Se omite.")
                continue
                
            # Ordenar por número de pregunta
            grupo = grupo.sort_values('pregunta')
            
            # 3. Aplanar y Discretizar
            evidencia = {}
            flat_values = []
            
            # Extraer T, F, I en orden
            for _, row in grupo.iterrows():
                flat_values.extend([row['t'], row['f'], row['i']])
            
            # Asignar a nombres de nodos (Feat_0 a Feat_35)
            # Nota: El modelo espera Feat_0..Feat_X. Son 26 preguntas * 3 valores = 78 features? 
            # Ojo: En dataset generator vi que aplanaba 26 * 3.
            for i, val in enumerate(flat_values):
                nombre_nodo = f'Feat_{i}'
                if nombre_nodo in modelo_bn.nodes():
                    evidencia[nombre_nodo] = discretizar_valor(val)
            
            # 4. Inferencia Bayesiana
            targets = ['estres_predicho', 'ansiedad_predicha', 'depresion_predicha']
            valid_targets = [t for t in targets if t in modelo_bn.nodes()]
            
            # Diccionario temporal para guardar resultados raw
            predicciones_raw = {
                'estres_predicho': {'val': 0.0, 't': 0.0, 'i': 0.0, 'f': 0.0},
                'ansiedad_predicha': {'val': 0.0, 't': 0.0, 'i': 0.0, 'f': 0.0},
                'depresion_predicha': {'val': 0.0, 't': 0.0, 'i': 0.0, 'f': 0.0}
            }
            
            if evidencia and valid_targets:
                try:
                    resultado = inferencia_engine.query(variables=valid_targets, evidence=evidencia, show_progress=False, joint=False)
                    
                    for target in valid_targets:
                        factor = resultado[target]
                        
                        # Función auxiliar para extraer probabilidad de un estado de forma segura
                        def get_state_prob(factor, state_name):
                            try:
                                return factor.get_value(**{target: state_name})
                            except:
                                if state_name in factor.state_names[target]:
                                    idx = factor.state_names[target].index(state_name)
                                    return factor.values[idx]
                                return 0.0

                        # Extraer T (Alto), I (Medio), F (Bajo)
                        val_t = float(get_state_prob(factor, 'Alto'))
                        val_i = float(get_state_prob(factor, 'Medio'))
                        val_f = float(get_state_prob(factor, 'Bajo'))
                        
                        # Guardamos en el diccionario
                        predicciones_raw[target] = {
                            'val': val_t, # El valor principal para porcentaje sigue siendo T (Alto)
                            't': val_t,
                            'i': val_i,
                            'f': val_f
                        }
                        
                except Exception as e:
                    print(f"Error en inferencia para ID {id_cuestionario}: {e}")
            
            # --- CONSTRUCCION DE DETALLES Y RECOMENDACIONES ---
            
            # 1. Extraer raw values struct
            raw_estres = predicciones_raw.get('estres_predicho')
            raw_ansiedad = predicciones_raw.get('ansiedad_predicha')
            raw_depresion = predicciones_raw.get('depresion_predicha')
            
            # --- Calcular promedios crudos (Antes de la Red Bayesiana) ---
            brutos = {
                "GAD-7": {"t": 0.0, "f": 0.0, "i": 0.0, "count": 0},
                "PHQ-9": {"t": 0.0, "f": 0.0, "i": 0.0, "count": 0},
                "PSS-10": {"t": 0.0, "f": 0.0, "i": 0.0, "count": 0}
            }
            
            for _, row in grupo.iterrows():
                codigo = row['codigo_prueba']
                if codigo in brutos:
                    brutos[codigo]['t'] += row['t']
                    brutos[codigo]['f'] += row['f']
                    brutos[codigo]['i'] += row['i']
                    brutos[codigo]['count'] += 1
            
            for code in brutos:
                count = brutos[code]['count']
                if count > 0:
                    brutos[code]['t'] = round(brutos[code]['t'] / count, 3)
                    brutos[code]['f'] = round(brutos[code]['f'] / count, 3)
                    brutos[code]['i'] = round(brutos[code]['i'] / count, 3)
                    
            # 2. Generar Detalles (Mapping Semántico)
            detalles_lista = []
            
            # Ansiedad -> GAD-7
            detalles_lista.append(DetallePrueba(
                prueba="GAD-7", 
                condicion="Ansiedad", 
                porcentaje=round(raw_ansiedad['t'] * 100, 1),
                t=round(raw_ansiedad['t'], 3),
                i=round(raw_ansiedad['i'], 3),
                f=round(raw_ansiedad['f'], 3),
                t_bruto=brutos["GAD-7"]["t"],
                i_bruto=brutos["GAD-7"]["i"],
                f_bruto=brutos["GAD-7"]["f"]
            ))
            # Depresión -> PHQ-9
            detalles_lista.append(DetallePrueba(
                prueba="PHQ-9", 
                condicion="Depresión", 
                porcentaje=round(raw_depresion['t'] * 100, 1),
                t=round(raw_depresion['t'], 3),
                i=round(raw_depresion['i'], 3),
                f=round(raw_depresion['f'], 3),
                t_bruto=brutos["PHQ-9"]["t"],
                i_bruto=brutos["PHQ-9"]["i"],
                f_bruto=brutos["PHQ-9"]["f"]
            ))
            # Estrés -> PSS-10
            detalles_lista.append(DetallePrueba(
                prueba="PSS-10", 
                condicion="Estrés", 
                porcentaje=round(raw_estres['t'] * 100, 1),
                t=round(raw_estres['t'], 3),
                i=round(raw_estres['i'], 3),
                f=round(raw_estres['f'], 3),
                t_bruto=brutos["PSS-10"]["t"],
                i_bruto=brutos["PSS-10"]["i"],
                f_bruto=brutos["PSS-10"]["f"]
            ))
            
            # 3. Generar Recomendaciones
            try:
                consejos = obtener_recomendaciones(
                    raw_estres['t'], raw_ansiedad['t'], raw_depresion['t'],
                    raw_estres['i'], raw_ansiedad['i'], raw_depresion['i']
                )
            except Exception as e_rec:
                print(f"Error generando recomendaciones para ID {id_cuestionario}: {e_rec}")
                consejos = ["Error generando recomendaciones."]
            
            # 4. Crear Respuesta Final
            prediccion_obj = Prediccion(
                idcuestionario=id_cuestionario,
                detalles=detalles_lista,
                recomendaciones=consejos
            )
            
            resultados_prediccion.append(prediccion_obj)
            
        if not resultados_prediccion:
            raise HTTPException(status_code=400, detail="No se pudieron generar predicciones (datos incompletos o inválidos).")
            
        return resultados_prediccion

    except Exception as e:
        print(f"Error interno: {e}")
        raise HTTPException(status_code=500, detail=f"Error procesando la solicitud: {str(e)}")

# Para ejecutar localmente: uvicorn main:app --reload