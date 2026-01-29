
import os
import joblib
from pgmpy.inference import VariableElimination

def debug_bn():
    root = os.getcwd()
    path = os.path.join(root, "Modelo", "modelo_bn.pkl")
    print(f"Loading model from {path}")
    
    if not os.path.exists(path):
        print("Model file not found!")
        return

    model = joblib.load(path)
    print("Model loaded.")
    
    # Check states
    # pgmpy DiscreteBayesianNetwork stores states in .states attribute? 
    # Or we can check via CPDs.
    
    print("\nChecking CPDs for targets:")
    targets = ['estrés_predicho', 'ansiedad_predicha', 'depresion_predicha']
    
    for t in targets:
        if t in model.nodes():
            cpd = model.get_cpds(t)
            print(f"\nTarget: {t}")
            print(f"State Names: {cpd.state_names}")
            print(f"Values shape: {cpd.values.shape}")
        else:
            print(f"\nTarget {t} NOT IN MODEL NODES")

    # Simple inference test
    infer = VariableElimination(model)
    print("\nRunning Simple Inference Test for Stress High...")
    
    # Evidence: PSS High.
    # PSS Features: Feat_48 to Feat_77.
    # Let's set Feat_48='Alto' ... Feat_57='Alto' (Assuming first 10 PSS features are T components of the 10 questions)
    # Wait, mapping is:
    # Q17 -> Feat_48(T), Feat_49(F), Feat_50(I)
    
    evidence = {}
    # Set first 5 PSS questions to High (T=High)
    # Q17: Feat_48. Q18: Feat_51. ...
    for i in range(5):
        idx = 48 + (i * 3)
        evidence[f'Feat_{idx}'] = 'Alto'
        
    print(f"Evidence: {evidence}")
    try:
        res = infer.query(variables=['estrés_predicho'], evidence=evidence, show_progress=False)
        print(f"Result: {res}")
    except Exception as e:
        print(f"Inference Error: {e}")

if __name__ == "__main__":
    debug_bn()
