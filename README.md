#  Stress Detection System (Command-Line Interface)

This project is a **command-line based stress detection system** using **PPG (Photoplethysmogram)** and **GSR (Galvanic Skin Response)** sensor inputs. It uses machine learning to classify stress levels based on real-time input from the user.

---

##  Features

-  Real-time stress prediction from user-entered sensor data
-  Uses trained Random Forest model and StandardScaler
-  Extracts basic statistical features from input data
-  No GUI or Streamlit – runs fully in the terminal
-  Easy to test with simulated values

---

##  Project Structure

```
stress-detection/
├── model/
│   ├── trained_model.pkl        # Trained RandomForestClassifier
│   └── scaler.pkl              # StandardScaler for feature normalization
│   └── features_names.pkl
├── scripts/
│   ├──01_load_data
│   ├──02_preprocess_signals
│   ├──03_feature_combination
│   ├──04_train_model.py       
│   └──05_test_model.py        
real-time prediction using CLI
├── data/
│   └── processed_features.csv  # Optional: sensor data for training
├── README.md                   # Project documentation
├── requirements.txt            # Required packages
```

---

## ⚙Setup Instructions

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/stress-detection.git
cd stress-detection
```

### Step 2: Create and Activate a Virtual Environment (Optional but recommended)

```bash
python -m venv venv
# Activate:
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

---
### Step 4: Download the Dataset
Dataset Name: PPG-DaLiA: A Public Dataset for PPG-Based Daily Stress Monitoring
Source: [PhysioNet / IEEE DataPort]

Credits / Citation:

Reiss, Attila, et al. “Deep PPG: Large-scale heart rate estimation with convolutional neural networks.” Sensors 19.14 (2019): 3079.
Dataset available at: https://www.physionet.org/content/ppg-dalia/1.0.0/
##  How to Use

###  Training the Model

Use this if you have CSV data from PPG & GSR sensors.

```bash
python scripts/03_train_model.py
```

This will generate:
- `train_model.pkl`
- `scaler.pkl`

in the `model/` directory.

### 📊 Running the Real-Time Stress Detection

```bash
python scripts/05_test_model.py
```

Then, follow the prompt to enter your sensor values:

Example:
```
🔹 Enter real-time sensor values. Type 'done' to finish.
Enter PPG value (or 'done'): 0.7
Enter GSR value: 1.64
Enter PPG value (or 'done'): 0.6
Enter GSR value: 1.68
Enter PPG value (or 'done'): done
```

Example Output:

```
========================================
🧠 STRESS MONITOR REPORT
========================================
📊 Detected Stress Level: 😌 No Stress
🔧 Stress Meter: [🟩🟩🟩⬜⬜] LOW
💡 Suggested Micro Break: You’re doing great! Take deep breaths and keep going! 💪
========================================
```

---

##  Requirements

List of Python dependencies (`requirements.txt`):

```
numpy
pandas
scikit-learn==1.5.1
joblib
```

You can regenerate this by running:

```bash
pip freeze > requirements.txt
```

---

##  Author

**Ananya**  
Electronics and Communication Engineering Student  
GitHub: [https://github.com/ANANYA-IS](https://github.com/ANANYA-IS)

---

##  License

This project is licensed under the [MIT License](LICENSE)
