import joblib
import os
import sys

# Add root to path so we can import things if needed (though we just need joblib here)
sys.path.append(os.getcwd())

MODEL_PATH = os.path.join("Modelo", "modelo_bn.pkl")

def verify_load():
    print(f"Testing model load from: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        print("ERROR: Model file not found!")
        return
        
    try:
        model = joblib.load(MODEL_PATH)
        print("SUCCESS: Model loaded successfully.")
        print(f"Model Nodes: {len(model.nodes())}")
        print("Verification OK.")
    except Exception as e:
        print(f"CRITICAL ERROR loading model: {e}")

if __name__ == "__main__":
    verify_load()
