import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

def train_and_save_model():
    """
    Generates synthetic market/operational data and trains a baseline 
    Random Forest model to predict risk events.
    """
    print("🚂 Training baseline ML risk model...")
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Feature 1: Market Volatility Index (0.0 to 1.0)
    # Feature 2: Internal Liquidity Ratio (0.0 to 2.0)
    # Feature 3: Sentiment Negative Keyword Frequency (0 to 100)
    np.random.seed(42)
    
    # Generate 1000 synthetic historical records
    X = np.column_stack((
        np.random.uniform(0.1, 0.9, 1000), 
        np.random.uniform(0.5, 1.8, 1000),
        np.random.randint(0, 80, 1000)
    ))
    
    # Target: 1 if Risk Event Occurred, 0 if Safe
    # Logic: High volatility + low liquidity + high negative sentiment = Risk Event
    y = ((X[:, 0] > 0.7) & (X[:, 1] < 0.8) & (X[:, 2] > 50)).astype(int)
    
    # Add some noise to make the model work for generalizations
    noise = np.random.choice([0, 1], size=1000, p=[0.95, 0.05])
    y = np.logical_xor(y, noise).astype(int)

    # Train the model
    clf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
    clf.fit(X, y)

    # Save the model artifact to the local Mac M4 filesystem
    model_path = os.path.join(MODEL_DIR, "rf_risk_v1.joblib")
    joblib.dump(clf, model_path)
    
    print(f"✅ Model trained and saved to {model_path}")

if __name__ == "__main__":
    train_and_save_model()