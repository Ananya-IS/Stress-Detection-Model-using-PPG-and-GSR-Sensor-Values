import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# --- Simulated feature data (Replace this with actual feature extraction) ---
# Assume you already have features like below
# Columns must match real extracted feature names
data = {
    'mean_bvp': np.random.rand(100),
    'std_bvp': np.random.rand(100),
    'max_bvp': np.random.rand(100),
    'ent_bvp': np.random.rand(100),
    'mean_eda': np.random.rand(100),
    'std_eda': np.random.rand(100),
    'max_eda': np.random.rand(100),
    'ent_eda': np.random.rand(100),
}
X = pd.DataFrame(data)

# Labels (1: Low stress, 2: Mild, 3: High) — replace with real labels
y = np.random.choice([1, 2, 3], size=(100,))

# --- Save the feature order ---
joblib.dump(X.columns.tolist(), "model/feature_names.pkl")

# --- Scale features ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Train model ---
model = RandomForestClassifier(random_state=42)
model.fit(X_scaled, y)

# --- Save model and scaler ---
joblib.dump(model, "model/trained_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")

print("✅ Model and scaler saved successfully.")




