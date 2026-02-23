import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import joblib

def generate_synthetic_data(samples=2000):
    # Use numpy for more efficient deterministic-like generation if needed, 
    # but keep random for variety in training set. 
    # The KEY is the logic mapping features to labels.
    import random 
    
    data = []
    for _ in range(samples):
        # Features: load_mbps, delay_ms, queue_length, arrival_rate
        # We try to create clusters that make sense
        load = random.uniform(0.01, 110.0)
        
        # In real world, high load correlates with high delay and queue
        if load < 30.0:
            delay = random.uniform(5.0, 15.0)
            queue = random.uniform(0.0, 50.0)
            arrival = load * 120
            state = "low"
        elif load < 70.0:
            delay = random.uniform(15.0, 45.0)
            queue = random.uniform(50.0, 150.0)
            arrival = load * 120
            state = "med"
        else:
            delay = random.uniform(45.0, 150.0)
            queue = random.uniform(150.0, 500.0)
            arrival = load * 120
            state = "high"
            
        data.append([load, delay, queue, arrival, state])
    
    return pd.DataFrame(data, columns=['load_mbps', 'delay_ms', 'queue_length', 'arrival_rate', 'state'])

def train():
    print("Generating refined synthetic data...")
    df = generate_synthetic_data(3000)
    
    X = df[['load_mbps', 'delay_ms', 'queue_length', 'arrival_rate']]
    y = df['state']
    
    print("Training Decision Tree model...")
    # max_depth=None allows for better accuracy on more complex patterns
    clf = DecisionTreeClassifier(max_depth=None)
    clf.fit(X, y)
    
    print(f"Model trained with accuracy: {clf.score(X, y):.2f}")
    
    model_path = 'traffic_model.joblib'
    joblib.dump(clf, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train()
