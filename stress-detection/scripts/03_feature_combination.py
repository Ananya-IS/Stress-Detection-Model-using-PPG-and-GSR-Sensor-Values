import os
import numpy as np
import pandas as pd
from scipy.stats import entropy

# Create 'data' folder if it doesn't exist
os.makedirs("data", exist_ok=True)

# Load filtered dataset
data = np.load("dataset_filtered.npz")
X = data["X"]  # shape: (samples, 2, signal_length)
y = data["y"]

features = []
for i in range(X.shape[0]):
    eda_signal = X[i][0]
    bvp_signal = X[i][1]
    
    # EDA features
    mean_eda = np.mean(eda_signal)
    std_eda = np.std(eda_signal)
    ent_eda = entropy(np.histogram(eda_signal, bins=10)[0] + 1)
    max_eda = np.max(eda_signal)

    # BVP features
    mean_bvp = np.mean(bvp_signal)
    std_bvp = np.std(bvp_signal)
    ent_bvp = entropy(np.histogram(bvp_signal, bins=10)[0] + 1)
    max_bvp = np.max(bvp_signal)

    features.append([
        mean_eda, std_eda, ent_eda, max_eda,
        mean_bvp, std_bvp, ent_bvp, max_bvp
    ])

# Create dataframe
df = pd.DataFrame(features, columns=[
    "mean_eda", "std_eda", "ent_eda", "max_eda",
    "mean_bvp", "std_bvp", "ent_bvp", "max_bvp"
])
df["label"] = y

# Save to CSV
df.to_csv("data/processed_features.csv", index=False)
print("✅ Saved: data/processed_features.csv")

