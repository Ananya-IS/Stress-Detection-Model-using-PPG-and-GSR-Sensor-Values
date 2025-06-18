# 01_load_data.py

import pickle

# Path to data
data_path = r"C:\Users\anany\Documents\stress-detection\WESAD\S2\S2.pkl"

with open(data_path, "rb") as f:
    data = pickle.load(f, encoding='latin1')

print("Top-level keys:", data.keys())
print("Wrist signal keys:", data['signal']['wrist'].keys())
print("Label shape:", data['label'].shape)



