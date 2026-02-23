import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import joblib
import random

def generate_synthetic_data(samples=1000):
    data = []
    for _ in range(samples):
        # Features: load_mbps, delay_ms, queue_length, arrival_rate
        load = random.uniform(0.01, 120.0)
        delay = random.uniform(5.0, 150.0)
        queue = random.uniform(0.0, 500.0)
        arrival = random.uniform(0, 15000)
        
        # Simple heuristic for labels (matching existing app.py logic roughly)
        # utilization_ratio = load / 100.0
        if load < 40.0:
            state = "low"
        elif load < 75.0:
            state = "med"
        else:
            state = "high"
            
        data.append([load, delay, queue, arrival, state])
    
    return pd.DataFrame(data, columns=['load_mbps', 'delay_ms', 'queue_length', 'arrival_rate', 'state'])

def train():
    print("Generating synthetic data...")
    df = generate_synthetic_data(2000)
    
    X = df[['load_mbps', 'delay_ms', 'queue_length', 'arrival_rate']]
    y = df['state']
    
    print("Training Decision Tree model...")
    clf = DecisionTreeClassifier(max_depth=5)
    clf.fit(X, y)
    
    print(f"Model trained with accuracy: {clf.score(X, y):.2f}")
    
    model_path = 'traffic_model.joblib'
    joblib.dump(clf, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train()
