# 🧠 Cross-Subject EEG Classification  
## Eyes Open vs Eyes Closed Using Classical Machine Learning

---

## 📌 Objective

This project evaluates whether spectral EEG features can reliably distinguish between **Eyes Open (EO)** and **Eyes Closed (EC)** conditions across unseen subjects.

The focus is on:

- Proper cross-subject validation  
- Prevention of subject leakage  
- Feature ablation analysis  
- Physiologically interpretable modeling  

Final performance:

> **Cross-subject ROC-AUC: 0.805**

---

## 🧠 Scientific Background

The Eyes Open vs Eyes Closed paradigm is a classical EEG condition contrast.  
Eyes Closed is typically associated with:

- Increased alpha power (8–12 Hz)
- Occipital dominance of alpha rhythms
- Changes in spectral distribution

Alpha reactivity is one of the most robust phenomena in resting-state EEG research  
(Berger, 1929; Barry et al., 2007).

Relevant literature:

- Berger, H. (1929). *Über das Elektrenkephalogramm des Menschen.*
- Barry, R. J., Clarke, A. R., Johnstone, S. J., Magee, C. A., & Rushby, J. A. (2007). EEG differences between eyes-closed and eyes-open resting conditions. *Clinical Neurophysiology.*
- Klimesch, W. (1999). EEG alpha and theta oscillations reflect cognitive and memory performance. *Brain Research Reviews.*

---

## 📊 Dataset

- **20 subjects**
- Each subject performed:
  - Eyes Open (EO)
  - Eyes Closed (EC)
- EEG segmented into epochs
- Band power features extracted per channel

This is explicitly treated as a **cross-subject classification problem**.

---

## ⚙️ Feature Engineering

For each EEG channel:

### Absolute Band Power
- Delta (0.5–4 Hz)
- Theta (4–8 Hz)
- Alpha (8–12 Hz)
- Beta (13–30 Hz)

### Relative Band Power
\[
\text{Relative Power} = \frac{\text{Band Power}}{\text{Total Power}}
\]

Absolute power captures global amplitude changes.  
Relative power captures spectral redistribution.

Both were retained to test complementary contributions.

Band power features are widely used in EEG classification  
(Bashivan et al., 2015; Lotte et al., 2018).

References:

- Bashivan, P., et al. (2015). Learning representations from EEG with deep recurrent-convolutional neural networks. *ICLR Workshop.*
- Lotte, F., et al. (2018). A review of classification algorithms for EEG-based brain–computer interfaces. *Journal of Neural Engineering.*

---

## 🧪 Preprocessing

### Within-Subject Z-Score Normalization

EEG amplitude varies significantly across individuals due to:

- Skull conductivity differences
- Electrode impedance
- Head anatomy
- Baseline neural variability

To reduce inter-subject scaling effects:

\[
Z = \frac{X - \mu_{subject}}{\sigma_{subject}}
\]

Normalization was performed **within subject**, preserving condition differences while reducing baseline amplitude bias.

This is critical for cross-subject generalization  
(Jayaram et al., 2016).

Reference:

- Jayaram, V., et al. (2016). Transfer learning in brain–computer interfaces. *IEEE Computational Intelligence Magazine.*

---

## 🔬 Validation Strategy

### Grouped Cross-Validation

To prevent subject leakage:

- **GroupKFold (5-fold)**
- Subject ID used as grouping variable
- No subject appears in both training and testing sets

This ensures true cross-subject evaluation.

Subject leakage can artificially inflate EEG classification performance  
(Varoquaux et al., 2017).

Reference:

- Varoquaux, G., et al. (2017). Assessing and tuning brain decoders: Cross-validation, caveats, and guidelines. *NeuroImage.*

---

## 🤖 Models Evaluated

### Logistic Regression
- Linear baseline
- L2 regularization
- `class_weight="balanced"`

### Random Forest
- Nonlinear ensemble model
- 400+ trees
- `class_weight="balanced"`

Random Forest was selected due to:

- Robustness to nonlinear interactions
- Suitability for tabular spectral features
- Stability with moderate sample sizes

---

## 📈 Evaluation Metrics

- Sensitivity (Recall for EC)
- Specificity (Recall for EO)
- ROC-AUC

ROC-AUC is threshold-independent and appropriate for balanced binary classification.

---

## 🏆 Results

### Best Model  
**Random Forest + Absolute & Relative Features + Within-Subject Z-score**

| Metric        | Score  |
|--------------|--------|
| Sensitivity  | 0.735  |
| Specificity  | 0.753  |
| ROC-AUC      | **0.805** |

Performance reflects cross-subject generalization to unseen individuals.

---

## 🔎 Feature Ablation Study

To evaluate feature contribution:

| Feature Set        | ROC-AUC |
|--------------------|----------|
| All features       | 0.805    |
| Relative-only      | 0.765    |

Removing absolute power reduced performance (~0.04 AUC drop).

### Interpretation

- Absolute amplitude shifts contribute discriminative signal.
- Relative spectral redistribution also contributes.
- Nonlinear models leverage interactions between both feature types.

This suggests EO vs EC differences include both:

1. Global amplitude changes  
2. Frequency distribution shifts  

---

## 🧠 Technical Insights

- Cross-subject EEG classification is variance-limited with small cohorts.
- Within-subject normalization substantially improves generalization.
- Proper grouped cross-validation is critical.
- Nonlinear models outperform linear baselines.
- Absolute and relative band powers provide complementary information.

With only 20 subjects, an AUC of 0.80 indicates robust condition separability.

---

## ⚠️ Limitations

- Small subject cohort (n=20)
- Limited to band power features
- No connectivity or higher-order spectral metrics
- No domain adaptation techniques applied

---

## 🚀 Future Directions

- Gradient Boosting (e.g., HistGradientBoosting, XGBoost)
- Alpha reactivity indices
- Band ratios (alpha/beta, theta/alpha)
- Spectral entropy
- 1/f slope modeling
- Larger cross-site dataset
- Domain adaptation for subject-invariant representations

---

## 🛠️ Tech Stack

- Python 3.x
- NumPy
- SciPy
- scikit-learn
- Matplotlib

---

## 📌 Conclusion

This project demonstrates that interpretable spectral features combined with rigorous cross-subject validation can achieve reliable EEG condition decoding.

The emphasis was on:

- Methodological rigor  
- Prevention of data leakage  
- Physiological interpretability  
- Robust generalization  

Even with only 20 subjects, a cross-subject ROC-AUC of 0.80 was achieved using classical machine learning methods.


# 👤 Author

**Cristopher Delgado**  
Data Scientist | Signal Processing | Biomedical AI