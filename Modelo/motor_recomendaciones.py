import random
from typing import List

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

def obtener_recomendaciones(prob_estres: float, prob_ansiedad: float, prob_depresion: float) -> List[str]:
    """
    Genera una lista de recomendaciones de salud mental basadas en las probabilidades de
    Estrés, Ansiedad y Depresión.
    
    Args:
        prob_estres (float): Probabilidad de estrés alto (0.0 - 1.0).
        prob_ansiedad (float): Probabilidad de ansiedad alta (0.0 - 1.0).
        prob_depresion (float): Probabilidad de depresión alta (0.0 - 1.0).
        
    Returns:
        List[str]: Lista de consejos y mensajes.
    """
    prioridad_alta = []
    prioridad_media = []
    prioridad_baja = []
    
    # Manejo de casos None o inválidos (por robustez se tratan como 0.0)
    p_estres = prob_estres if prob_estres is not None else 0.0
    p_ansiedad = prob_ansiedad if prob_ansiedad is not None else 0.0
    p_depresion = prob_depresion if prob_depresion is not None else 0.0
    
    # Flags para determinar si hay alertas
    alerta_roja = False
    alerta_amarilla = False
    
    # --- LOGICA ESTRÉS ---
    if p_estres > 0.66:
        prioridad_alta.extend(random.sample(POOL_ESTRES_ALTO, 2))
        alerta_roja = True
    elif p_estres > 0.33:
        prioridad_media.append(MENSAJE_AMARILLO_ESTRES)
        alerta_amarilla = True
        
    # --- LOGICA ANSIEDAD ---
    if p_ansiedad > 0.66:
        prioridad_alta.extend(random.sample(POOL_ANSIEDAD_ALTA, 2))
        alerta_roja = True
    elif p_ansiedad > 0.33:
        prioridad_media.append(MENSAJE_AMARILLO_ANSIEDAD)
        alerta_amarilla = True
        
    # --- LOGICA DEPRESIÓN ---
    if p_depresion > 0.66:
        prioridad_alta.extend(random.sample(POOL_DEPRESION_ALTA, 2))
        alerta_roja = True
    elif p_depresion > 0.33:
        prioridad_media.append(MENSAJE_AMARILLO_DEPRESION)
        alerta_amarilla = True

    # --- REGLA DE COMORBILIDAD ---
    # Si Estrés y Ansiedad son altos, inserta la advertencia al inicio de prioridad_alta
    if p_estres > 0.66 and p_ansiedad > 0.66:
        prioridad_alta.insert(0, CONSEJO_COMORBILIDAD)

    # --- ADVERTENCIA GENERAL ---
    # Si hubo alguna alerta roja, añadimos la advertencia al final de la prioridad alta
    if alerta_roja:
        prioridad_alta.append(ADVERTENCIA_ROJA)

    # --- REGLA VERDE (SOLO SI NO HAY ALERTAS) ---
    if not alerta_roja and not alerta_amarilla:
        prioridad_baja.append(MENSAJE_VERDE)
        
    # Retorno: Alta -> Media -> Baja
    return prioridad_alta + prioridad_media + prioridad_baja
