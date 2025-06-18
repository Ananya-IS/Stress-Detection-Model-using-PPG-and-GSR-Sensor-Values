import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load model, scaler, and feature order
model = joblib.load("model/trained_model.pkl")
scaler = joblib.load("model/scaler.pkl")
feature_order = joblib.load("model/feature_names.pkl")

# Collect PPG and GSR input from user
ppg_values = []
gsr_values = []

print("🔹 Enter real-time sensor values. Type 'done' to finish.")

while True:
    ppg_input = input("Enter PPG value (or 'done'): ").strip()
    if ppg_input.lower() == 'done':
        break
    gsr_input = input("Enter GSR value: ").strip()

    try:
        ppg = float(ppg_input)
        gsr = float(gsr_input)
        ppg_values.append(ppg)
        gsr_values.append(gsr)
    except ValueError:
        print("⚠️ Invalid input, please enter numeric values.")

# Check if enough samples
if len(ppg_values) < 3:
    print("⚠️ Not enough samples. Please enter at least 3 readings.")
    exit()

# Feature extraction
features = {
    'mean_bvp': np.mean(ppg_values),
    'std_bvp': np.std(ppg_values),
    'max_bvp': np.max(ppg_values),
    'ent_bvp': -np.sum(np.histogram(ppg_values, bins=10, density=True)[0] * np.log2(np.histogram(ppg_values, bins=10, density=True)[0] + 1e-10)),
    'mean_eda': np.mean(gsr_values),
    'std_eda': np.std(gsr_values),
    'max_eda': np.max(gsr_values),
    'ent_eda': -np.sum(np.histogram(gsr_values, bins=10, density=True)[0] * np.log2(np.histogram(gsr_values, bins=10, density=True)[0] + 1e-10)),
}

features_df = pd.DataFrame([features])
features_df = features_df[feature_order]  # Reorder to match training

# Scale
features_scaled = scaler.transform(features_df)

# Predict
predicted_class = model.predict(features_scaled)[0]

# Stress level interpretation
stress_levels = {
    1: ("😌 No Stress", "You’re doing great! Take deep breaths and keep going! 💪"),
    2: ("😐 Mild Stress", "Consider a 2-minute walk, some water, or a few deep breaths. 🧘"),
    3: ("😣 High Stress", "Take a 5–10 min break. Stretch, breathe deeply, or step outside. 🛑"),
}
stress_text, suggestion = stress_levels.get(predicted_class, ("Unknown", "No suggestion."))

def stress_meter(level):
    bars = {
        1: "[🟩🟩🟩⬜⬜] LOW",
        2: "[🟨🟨🟨🟨⬜] MILD",
        3: "[🟥🟥🟥🟥🟥] HIGH"
    }
    return bars.get(level, "[⬜⬜⬜⬜⬜] UNKNOWN")

def plot_stress_meter(stress_level):
    labels = ['Low', 'Mild', 'High']
    colors = ['green', 'yellow', 'red']
    angle = [0, 90, 180]

    fig, ax = plt.subplots(figsize=(6, 3), subplot_kw={'projection': 'polar'})
    ax.set_theta_offset(np.pi)
    ax.set_theta_direction(-1)

    for i in range(3):
        ax.bar(
            np.radians(np.linspace(angle[i], angle[i] + 60, 30)),
            1, width=np.radians(2),
            color=colors[i], edgecolor='white'
        )

    theta = {1: 30, 2: 90, 3: 150}.get(stress_level, 0)
    ax.plot([np.radians(theta), np.radians(theta)], [0, 1], lw=3, color='black')
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_title("🧠 Stress Meter", fontsize=16, pad=20)
    plt.show()

# Output
print("=" * 40)
print("🧠 STRESS MONITOR REPORT")
print("=" * 40)
print(f"📊 Detected Stress Level: {stress_text}")
print(f"🔧 Stress Meter: {stress_meter(predicted_class)}")
print(f"💡 Suggested Micro Break: {suggestion}")
print("=" * 40)

# Show plot
plot_stress_meter(predicted_class)

