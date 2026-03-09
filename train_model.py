import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
import joblib

def generate_synthetic_data(samples=5000):
    """
    Generate more detailed synthetic data that captures the relationships 
    found in the ML-driven QoS simulation script.
    """
    import random
    data = []
    
    # Simulate a wide range of network scenarios
    for _ in range(samples):
        # Features: load_mbps (0-100), delay_ms (5-500), queue_length (0-500), arrival_rate (0-15000)
        load = random.uniform(0.1, 100.0)
        
        # Base relationships: High load -> high delay, high queue length
        # Low congestion (< 35 Mbps)
        if load < 35.0:
            delay = 5.0 + random.uniform(0, 5) * (load / 10.0)
            queue = int(load * 0.5 + random.uniform(0, 5))
            arrival = int(load * 120 + random.uniform(-50, 50))
            state = "low"
        # Medium congestion (35 - 75 Mbps)
        elif load < 75.0:
            delay = 15.0 + random.uniform(5, 15) * (load / 20.0)
            queue = int(load * 2.0 + random.uniform(20, 100))
            arrival = int(load * 125 + random.uniform(-100, 100))
            state = "med"
        # High congestion (> 75 Mbps)
        else:
            delay = 50.0 + random.uniform(50, 200) * (load / 50.0)
            queue = int(load * 4.0 + random.uniform(100, 300))
            arrival = int(load * 130 + random.uniform(-200, 200))
            state = "high"
            
        data.append([load, delay, queue, arrival, state])
    
    return pd.DataFrame(data, columns=['load_mbps', 'delay_ms', 'queue_length', 'arrival_rate', 'state'])

def train():
    print("Generating comprehensive synthetic dataset for ML-Driven QoS...")
    df = generate_synthetic_data(10000)
    
    X = df[['load_mbps', 'delay_ms', 'queue_length', 'arrival_rate']]
    y = df['state']
    
    # Split for validation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Starting ML Model Optimization (Grid Search)...")
    
    # Model 1: Optimized Decision Tree (User's preferred algorithm)
    dt_params = {
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    dt_grid = GridSearchCV(
        DecisionTreeClassifier(random_state=42), 
        dt_params, 
        cv=5, 
        n_jobs=-1
    )
    dt_grid.fit(X_train, y_train)
    
    best_dt = dt_grid.best_estimator_
    accuracy = dt_grid.score(X_test, y_test)
    
    print(f"Decision Tree trained. Accuracy: {accuracy:.4f}")
    print(f"Best parameters: {dt_grid.best_params_}")
    
    # Cross-validation for stability
    cv_scores = cross_val_score(best_dt, X, y, cv=5)
    print(f"Cross-validation Mean Accuracy: {cv_scores.mean():.4f}")
    
    # Save the model
    model_path = 'traffic_model.joblib'
    joblib.dump(best_dt, model_path)
    print(f"\n✓ Intelligent ML Model saved to: {model_path}")
    print("-" * 50)
    print("READY FOR REAL-TIME DEPLOYMENT")

if __name__ == "__main__":
    train()

