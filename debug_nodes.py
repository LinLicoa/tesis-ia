
import os
import joblib

def load_and_print_nodes():
    root = os.getcwd()
    path = os.path.join(root, "Modelo", "modelo_bn.pkl")
    if not os.path.exists(path):
        print("Model not found")
        return

    model = joblib.load(path)
    print("--- Model Nodes ---")
    nodes = sorted(list(model.nodes()))
    for n in nodes:
        print(f"'{n}'")

    print("\n--- Target Names Used in Code ---")
    t1 = 'estrés_predicho'
    print(f"Code uses: '{t1}'")
    
    if t1 in nodes:
        print(f"MATCH: '{t1}' found in nodes.")
    else:
        print(f"FAIL: '{t1}' NOT found in nodes.")

if __name__ == "__main__":
    load_and_print_nodes()
