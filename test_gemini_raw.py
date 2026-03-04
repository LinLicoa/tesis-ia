import os
from dotenv import load_dotenv
load_dotenv()
from google import genai
from Modelo.motor_recomendaciones import generar_prompt_clinico

print("Probando generacion cruda...")
try:
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    prompt = generar_prompt_clinico(1.0, 1.0, 1.0, 0.0, 0.0, 0.0)
    
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
    )
    
    print("FINISH REASON:", response.candidates[0].finish_reason if response.candidates else "NO CANDIDATA")
    print("TEXTRAW:\n", response.text)
except Exception as e:
    print(f"Error: {e}")
