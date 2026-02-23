import joblib
import pandas as pd
import numpy as np

def test_model():
    print("Loading model...")
    try:
        model = joblib.load('traffic_model.joblib')
        print("Model loaded.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Test cases: load_mbps, delay_ms, queue_length, arrival_rate
    test_data = [
        [10.0, 10.0, 5.0, 1000],   # low
        [60.0, 50.0, 100.0, 5000], # med
        [95.0, 120.0, 400.0, 12000] # high
    ]
    
    df = pd.DataFrame(test_data, columns=['load_mbps', 'delay_ms', 'queue_length', 'arrival_rate'])
    predictions = model.predict(df)
    
    print("\nPredictions:")
    for i, pred in enumerate(predictions):
        print(f"Input {test_data[i]} -> Predicted State: {pred}")

if __name__ == "__main__":
    test_model()
