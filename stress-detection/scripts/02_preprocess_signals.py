

import pickle
from scipy.signal import resample
import numpy as np

with open(r"C:\Users\anany\Documents\stress-detection\WESAD\S2\S2.pkl", "rb") as f:
    data = pickle.load(f, encoding='latin1')

eda = data['signal']['wrist']['EDA'].flatten()
bvp = data['signal']['wrist']['BVP'].flatten()
labels = data['label']

# Resample BVP from 64 Hz to 4 Hz
bvp_resampled = resample(bvp, len(eda))

# Resample labels to match EDA
labels_down = resample(labels, len(eda))
labels_down = np.round(labels_down).astype(int)

# Save to .npz for later use
np.savez("preprocessed_signals.npz", eda=eda, bvp=bvp_resampled, labels=labels_down)
