# 🧠 EEG Condition Classification Pipeline  
**Signal Processing + Feature Engineering + Classical Machine Learning**

End-to-end machine learning pipeline for classifying experimental conditions from raw EEG recordings.

This project demonstrates:

- Raw signal ingestion and conditioning  
- Preprocessing validation against reference data  
- Feature engineering from nonstationary time-series signals  
- Cross-validated model evaluation  
- Reproducible ML workflow  

---

# 🚀 Objective

Build a robust pipeline that:

1. Converts raw ADC EEG recordings into clean physiological signals  
2. Extracts structured features from time-series data  
3. Classifies experimental conditions using classical ML  

The focus is on **signal-aware ML engineering**, not deep learning shortcuts.

---

# 🏗️ Pipeline Architecture

---

# 🔧 1. Signal Conditioning

Raw EEG requires domain-aware preprocessing before ML.

### Steps

- ADC → microvolt scaling  
- Trim to experiment window  
- Bandpass filter (1–40 Hz)  
- Preserve original reference configuration  

Why this matters:

- Removes non-neural artifacts  
- Ensures physiological interpretability  
- Produces reproducible inputs for modeling  

---

# ✅ 2. Validation Against Reference Pipeline

Before modeling, preprocessing was validated against the dataset’s provided filtered signals.

Validation metrics:

- Pearson correlation  
- Mean absolute difference  
- Channel-wise gain factors  

### Result

- High waveform correlation  
- Constant amplitude scaling factor (~540×)  
- Structural equivalence confirmed  

This step ensures modeling is built on verified signal processing — not guesswork.

---

# ✂️ 3. Epoching (Time-Series Segmentation)

EEG is nonstationary.

Continuous recordings are segmented into:

- 2-second windows  
- 1-second overlap  

Benefits:

- Increases training samples  
- Improves stationarity  
- Enables feature stability  

---

# 📊 4. Feature Engineering

Each epoch is transformed into a structured feature vector.

### Frequency Features

- Absolute band power (delta–gamma)
- Relative band power

### Time-Domain Features

- Hjorth Activity  
- Mobility  
- Complexity  

### Spectral Features

- Spectral entropy  

This transforms raw waveforms into interpretable statistical descriptors suitable for classical ML.

---

# 🤖 5. Modeling

Pipeline:

- `StandardScaler`
- `RandomForestClassifier (n_estimators=200)`
- Stratified 5-fold cross-validation  

### Why Random Forest?

- Handles nonlinear relationships  
- Robust to feature scaling differences  
- Provides feature importance  
- Strong baseline for structured tabular features  

---

# 📈 Evaluation

Model performance is evaluated using:

- Cross-validated accuracy  
- Stratified folds  
- Leakage-aware splitting  

This ensures generalization performance reflects condition separability — not subject memorization.

---

# 🧠 Engineering Highlights

✔ Validated preprocessing against external reference  
✔ Clean separation between signal processing and modeling  
✔ Feature-based approach for interpretability  
✔ Reproducible ML pipeline  
✔ Modular code structure  

---

# 🛠 Tech Stack

- Python  
- NumPy  
- SciPy  
- Pandas  
- scikit-learn  
- Matplotlib  

---

# 📁 Project Structure

---

# 📌 Future Improvements

- Leave-One-Subject-Out validation  
- Feature importance visualization  
- Hyperparameter tuning (`GridSearchCV`)  
- XGBoost comparison  
- CNN-based time-series baseline  

---

# 👤 Author

**Cristopher Delgado **  
Data Scientist | Signal Processing | Biomedical AI