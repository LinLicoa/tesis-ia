import random
import os
import json
from typing import List

try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

# --- BANCOS DE CONSEJOS (POOLS) ---

POOL_ESTRES_ALTO = [
    "Practica la técnica de respiración 4-7-8: inhala en 4 seg, retén 7 seg, exhala en 8 seg.",
    "Prioriza tu higiene del sueño: intenta dormir a la misma hora y evita pantallas 1 hora antes.",
    "Realiza una desconexión digital programada de al menos 30 minutos al día.",
    "Dedica 10 minutos a organizar tus tareas por prioridad para reducir la carga mental.",
    "Considera practicar mindfulness o meditación guiada breve para reducir la tensión.",
    "Realiza estiramientos suaves de cuello y hombros para liberar tensión física.",
]

POOL_ANSIEDAD_ALTA = [
    "Utiliza la técnica de 'Grounding' 5-4-3-2-1: nombra 5 cosas que ves, 4 que tocas, etc.",
    "Aplica temperatura fría en tu rostro o manos para reducir la activación fisiológica.",
    "Reduce o evita el consumo de cafeína y estimulantes durante esta etapa.",
    "Escribe tus pensamientos en un diario para externalizar las preocupaciones.",
    "Escucha música relajante o sonidos de la naturaleza para calmar tu mente.",
    "Concéntrate en tu respiración abdominal, sintiendo cómo sube y baja tu diafragma.",
]

POOL_DEPRESION_ALTA = [
    "Intenta la activación conductual: realiza una pequeña tarea que solías disfrutar, aunque sea por 5 minutos.",
    "Busca contacto social breve: envía un mensaje o llama a un amigo o familiar de confianza.",
    "Exponte a la luz solar natural durante al menos 15 minutos al día.",
    "Establece una rutina pequeña y manejable para dar estructura a tu día.",
    "Realiza una caminata corta, el movimiento físico puede ayudar a mejorar el estado de ánimo.",
    "Sé amable contigo mismo/a, reconoce que estás pasando por un momento difícil y valida tus emociones.",
]

# Mensajes fijos y de refuerzo
ADVERTENCIA_ROJA = "IMPORTANTE: Estos indicadores son altos. Te recomendamos consultar con un profesional de la salud mental."
MENSAJE_AMARILLO_ESTRES = "Tu nivel de estrés es moderado. Es buen momento para revisar tus pausas y descansos."
MENSAJE_AMARILLO_ANSIEDAD = "Nivel de ansiedad moderado. Observa qué situaciones te generan inquietud."
MENSAJE_AMARILLO_DEPRESION = "Tu estado de ánimo parece algo bajo. Intenta mantener pequeñas actividades agradables."
MENSAJE_VERDE = "¡Sigue así! Tus indicadores emocionales se encuentran en un rango saludable. Mantén tus hábitos de autocuidado."
CONSEJO_COMORBILIDAD = "Notamos tensión acumulada significativa. Intenta combinar respiración profunda con un paseo relajante para liberar energía."
MENSAJE_ALTA_INDETERMINACION = "Tus respuestas muestran una alta variación o duda en este momento. Te sugerimos realizar nuevamente el cuestionario con más calma o consultar directamente a un profesional para una evaluación precisa."

def generar_prompt_clinico(p_estres: float, p_ansiedad: float, p_depresion: float, 
                             i_estres: float, i_ansiedad: float, i_depresion: float) -> str:
    """Construye el prompt estricto para Gemini con base en probabilidades."""
    
    # Calcular promedios para dar contexto rápido
    i_max = max(i_estres, i_ansiedad, i_depresion)
    
    prompt = f"""
    Actúa como un asistente virtual clínico para un sistema de evaluación psicológica y psicométrica.
    Acabamos de procesar un test psicológico usando un motor de inferencia Bayesiano-Neutrosófico.
    
    Los resultados probabilísticos del paciente son:
    - Riesgo de Estrés Alto: {p_estres:.1%} (Indeterminación/Duda en respuestas: {i_estres:.1%})
    - Riesgo de Ansiedad Alta: {p_ansiedad:.1%} (Indeterminación/Duda en respuestas: {i_ansiedad:.1%})
    - Riesgo de Depresión Alta: {p_depresion:.1%} (Indeterminación/Duda en respuestas: {i_depresion:.1%})
    
    Nivel máximo de duda/indeterminación detectado en toda la prueba: {i_max:.1%}
    
    INSTRUCCIONES ESTRICTAS (DEBES SEGUIRLAS AL PIE DE LA LETRA):
    1. Genera exactamente un arreglo JSON que contenga una lista de cadenas de texto (strings) con las recomendaciones, ordenadas de mayor a menor urgencia. Ejemplo: ["Consejo 1", "Consejo 2", "Mensaje final"].
    2. NUNCA emitas un diagnóstico médico oficial (ej. no digas "tienes depresión"). Di "los indicadores muestran riesgo de...".
    3. NUNCA recetes, recomiendes o menciones nombres de medicamentos, fármacos o suplementos.
    4. Usa un tono empático, respetuoso, calmado y clínico.
    5. NO devuelvas markdown estructurado como ```json ... ```. Devuelve el JSON puro, crudo y válido directamente.

    REGLAS DE DECISIÓN:
    A. Si el "Nivel máximo de duda/indeterminación" es MAYOR a 50% (0.50): La prioridad #1 absoluta de tu respuesta DEBE ser advertir al usuario que sus respuestas al cuestionario fueron altamente erráticas, inconsistentes o dudosas, y que debe repetir la prueba con más calma o asistir a una evaluación profesional presencial para asegurar un resultado preciso. Puedes agregar un consejo de relajación leve adicional.
    B. Si las probabilidades (cualquiera de las 3) superan el 66% (0.66): DEBES incluir una advertencia explícita recomendando buscar ayuda profesional terapéutica de forma pronta. Luego agrega 2 técnicas de contención psicológica inmediata (ej. respiración, grounding, etc.).
    C. Si las probabilidades están entre el 33% y el 66%: Da recomendaciones preventivas de autocuidado, pausas activas o gestión emocional diaria.
    D. Si todas las probabilidades están por debajo del 33%: Da un mensaje de felicitación e incentiva a mantener los buenos hábitos de salud mental actuales.
    """
    return prompt

def _obtener_recomendaciones_gemini(p_estres: float, p_ansiedad: float, p_depresion: float,
                                    ind_estres: float, ind_ansiedad: float, ind_depresion: float) -> List[str]:
    """Intenta obtener recomendaciones dinámicas desde Google Gemini usando el SDK nuevo."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
         raise ValueError("GEMINI_API_KEY no configurada.")
         
    client = genai.Client(api_key=api_key)
    
    prompt = generar_prompt_clinico(p_estres, p_ansiedad, p_depresion, ind_estres, ind_ansiedad, ind_depresion)
    
    # Usando el nuevo modelo gemini-2.5-flash y la estructura types de genai
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=600,
            response_mime_type="application/json",
            response_schema=list[str]
        )
    )
    
    if not response.text:
       raise Exception("Gemini devolvió una respuesta vacía.")
       
    # Limpiamos posibles formatos de markdown
    texto_limpio = response.text.replace('```json', '').replace('```', '').strip()
    
    try:
        # Extraer JSON de la estructura
        parsed = json.loads(texto_limpio)
        
        # Si Gemini devuelve un diccionario con una llave en lugar de la lista directa
        if isinstance(parsed, dict):
            # Buscar la primera llave que contenga una lista
            for value in parsed.values():
                if isinstance(value, list):
                    return value
            # Si no hay lista, devolver keys y values como strings
            return [str(v) for v in parsed.values()]
            
        if not isinstance(parsed, list):
            raise ValueError("El JSON parseado no es una lista ni un formato reconocible.")
            
        return parsed
        
    except json.JSONDecodeError as e:
        print(f"Error parseando JSON de Gemini: {texto_limpio}")
        raise e

def _obtener_recomendaciones_fallback(prob_estres: float, prob_ansiedad: float, prob_depresion: float,
                            ind_estres: float = 0.0, ind_ansiedad: float = 0.0, ind_depresion: float = 0.0) -> List[str]:
    """
    Función base original (Estática / Basada en POOLs). 
    Sirve como respaldo en caso de que Gemini falle.
    """
    ind_max = max(ind_estres if ind_estres is not None else 0.0, 
                  ind_ansiedad if ind_ansiedad is not None else 0.0, 
                  ind_depresion if ind_depresion is not None else 0.0)
    
    if ind_max > 0.5:
        return [MENSAJE_ALTA_INDETERMINACION]

    prioridad_alta = []
    prioridad_media = []
    prioridad_baja = []
    
    p_estres = prob_estres if prob_estres is not None else 0.0
    p_ansiedad = prob_ansiedad if prob_ansiedad is not None else 0.0
    p_depresion = prob_depresion if prob_depresion is not None else 0.0
    
    alerta_roja = False
    alerta_amarilla = False
    
    if p_estres > 0.66:
        prioridad_alta.extend(random.sample(POOL_ESTRES_ALTO, 2))
        alerta_roja = True
    elif p_estres > 0.33:
        prioridad_media.append(MENSAJE_AMARILLO_ESTRES)
        alerta_amarilla = True
        
    if p_ansiedad > 0.66:
        prioridad_alta.extend(random.sample(POOL_ANSIEDAD_ALTA, 2))
        alerta_roja = True
    elif p_ansiedad > 0.33:
        prioridad_media.append(MENSAJE_AMARILLO_ANSIEDAD)
        alerta_amarilla = True
        
    if p_depresion > 0.66:
        prioridad_alta.extend(random.sample(POOL_DEPRESION_ALTA, 2))
        alerta_roja = True
    elif p_depresion > 0.33:
        prioridad_media.append(MENSAJE_AMARILLO_DEPRESION)
        alerta_amarilla = True

    if p_estres > 0.66 and p_ansiedad > 0.66:
        prioridad_alta.insert(0, CONSEJO_COMORBILIDAD)

    if alerta_roja:
        prioridad_alta.append(ADVERTENCIA_ROJA)

    if not alerta_roja and not alerta_amarilla:
        prioridad_baja.append(MENSAJE_VERDE)
        
    return prioridad_alta + prioridad_media + prioridad_baja

def obtener_recomendaciones(prob_estres: float, prob_ansiedad: float, prob_depresion: float,
                            ind_estres: float = 0.0, ind_ansiedad: float = 0.0, ind_depresion: float = 0.0) -> List[str]:
    """
    Intenta generar recomendaciones inteligentes mediante LLM (Gemini).
    Si hay algún error de red, de cuota o falta de configuración, 
    recae de manera segura al motor de POOLs estáticos.
    """
    # 1. Limpieza de valores nulos
    p_e = prob_estres if prob_estres is not None else 0.0
    p_a = prob_ansiedad if prob_ansiedad is not None else 0.0
    p_d = prob_depresion if prob_depresion is not None else 0.0
    i_e = ind_estres if ind_estres is not None else 0.0
    i_a = ind_ansiedad if ind_ansiedad is not None else 0.0
    i_d = ind_depresion if ind_depresion is not None else 0.0
    
    # 2. Intento Generativo
    if GENAI_AVAILABLE and os.environ.get("GEMINI_API_KEY"):
        try:
            return _obtener_recomendaciones_gemini(p_e, p_a, p_d, i_e, i_a, i_d)
        except Exception as e:
            print(f"ADVERTENCIA: Motor Gemini falló ({str(e)}). Activando sistema Fallback estático.")
            # Continuamos al fallback...
            pass
    
    # 3. Fallback Seguro
    return _obtener_recomendaciones_fallback(p_e, p_a, p_d, i_e, i_a, i_d)
