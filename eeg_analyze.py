# ============================================================
# 0️⃣ Imports
# ============================================================
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import welch
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from sklearn.pipeline import Pipeline

sns.set_theme(style="whitegrid")


# ============================================================
# 1️⃣ PSD Example Function
# ============================================================
def plot_psd_example(eeg_eo, eeg_ec, fs=200, save_path=None):
    f_eo, psd_eo = welch(eeg_eo.mean(axis=0), fs=fs, nperseg=fs*2)
    f_ec, psd_ec = welch(eeg_ec.mean(axis=0), fs=fs, nperseg=fs*2)
    
    plt.figure(figsize=(8,4))
    plt.semilogy(f_eo, psd_eo, label="Eyes Open")
    plt.semilogy(f_ec, psd_ec, label="Eyes Closed")
    plt.xlim(0, 15)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD (µV²/Hz)")
    plt.title("PSD Example — Channel T7")
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


# ============================================================
# 2️⃣ Alpha Power Computation + Plot
# ============================================================
def compute_alpha_power(eeg_epochs, fs=200, alpha_band=(8,12)):
    n_epochs = eeg_epochs.shape[0]
    alpha_power = np.zeros(n_epochs)
    
    for i in range(n_epochs):
        f, psd = welch(eeg_epochs[i], fs=fs, nperseg=fs*2)
        alpha_mask = (f >= alpha_band[0]) & (f <= alpha_band[1])
        alpha_power[i] = psd[alpha_mask].mean()
    return alpha_power

def plot_alpha_distribution(alpha_eo, alpha_ec, save_path=None):
    plt.figure(figsize=(6,4))
    sns.kdeplot(alpha_eo, label="Eyes Open", fill=True)
    sns.kdeplot(alpha_ec, label="Eyes Closed", fill=True)
    plt.xlabel("Alpha Power (µV²)")
    plt.ylabel("Density")
    plt.title("Alpha Power Distribution")
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


# ============================================================
# 3️⃣ Mean ROC Curve Across Folds
# ============================================================
def plot_mean_roc(y_true_folds, y_prob_folds, save_path=None):
    mean_fpr = np.linspace(0,1,100)
    mean_tpr = 0

    plt.figure(figsize=(6,6))
    for y_true, y_prob in zip(y_true_folds, y_prob_folds):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        plt.plot(fpr, tpr, alpha=0.3)
        mean_tpr += np.interp(mean_fpr, fpr, tpr)
    
    mean_tpr /= len(y_true_folds)
    roc_auc = np.trapz(mean_tpr, mean_fpr)
    
    plt.plot(mean_fpr, mean_tpr, color='b', lw=2, label=f"Mean ROC (AUC={roc_auc:.2f})")
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Cross-Validated ROC Curve")
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


# ============================================================
# 4️⃣ Confusion Matrix
# ============================================================
def plot_confusion_matrix(y_true, y_pred, class_names=["Eyes Closed", "Eyes Open"], save_path=None):
    """
    Plot normalized confusion matrix with percentages.
    Handles empty rows gracefully.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    class_names : list of str
        Names for the classes
    save_path : str or None
        Where to save the figure
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    cm_norm = cm.astype('float')
    
    # Normalize each row by its sum; handle zero rows
    row_sums = cm.sum(axis=1)[:, np.newaxis]
    row_sums[row_sums == 0] = 1  # avoid division by zero
    cm_norm /= row_sums
    
    plt.figure(figsize=(5,5))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", cbar=False,
                xticklabels=class_names, yticklabels=class_names)
    
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Normalized Confusion Matrix")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()


# ============================================================
# 5️⃣ Feature Importance (for Logistic Regression or RF)
# ============================================================
def plot_feature_importance_logreg(model, feature_names, n_top=10, save_path=None):
    """
    Plot top n logistic regression coefficients with color based on direction.
    
    Positive → favors class 1 (Eyes Open, blue)
    Negative → favors class 0 (Eyes Closed, red)
    
    Parameters
    ----------
    model : fitted LogisticRegression model
    feature_names : list of str
    n_top : int
    """
    import matplotlib.colors as mcolors
    
    # Coefficients
    coefs = model.coef_.flatten()
    abs_coefs = np.abs(coefs)
    
    # Top n features by absolute value
    indices = np.argsort(abs_coefs)[::-1][:n_top]
    top_features = np.array(feature_names)[indices]
    top_abs = abs_coefs[indices]
    top_sign = np.sign(coefs[indices])
    
    # Colors: blue for positive, red for negative
    colors = ['#1f77b4' if s > 0 else '#d62728' for s in top_sign]
    
    plt.figure(figsize=(6,4))
    sns.barplot(x=top_abs, y=top_features, palette=colors)
    plt.xlabel("Absolute Coefficient")
    plt.ylabel("Feature")
    plt.title(f"Top {n_top} Features (Eyes Open vs Closed)")

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#1f77b4', label='Eyes Open (1)'),
                       Patch(facecolor='#d62728', label='Eyes Closed (0)')]
    plt.legend(handles=legend_elements, loc='lower right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()


# ============================================================
# 6️⃣ Training + Cross-Validation + Metric Collection
# ============================================================
def evaluate_model_cv(X, y, groups, model_cls=LogisticRegression, model_kwargs=None):
    """
    Train and evaluate model using GroupKFold.
    Returns trained model on last fold, metrics, and predictions.
    """
    if model_kwargs is None:
        model_kwargs = {"max_iter":1000}
        
    gkf = GroupKFold(n_splits=5)
    
    auc_scores = []
    sensitivities = []
    specificities = []
    y_true_folds = []
    y_prob_folds = []
    
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", model_cls(**model_kwargs))
    ])
    
    trained_model = None
    
    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        pipeline.fit(X_train, y_train)
        trained_model = pipeline.named_steps['model']  # save last trained model
        
        y_prob = pipeline.predict_proba(X_test)[:,1]
        y_pred = pipeline.predict(X_test)
        
        # Metrics
        auc = roc_auc_score(y_test, y_prob)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        
        auc_scores.append(auc)
        sensitivities.append(sensitivity)
        specificities.append(specificity)
        y_true_folds.append(y_test)
        y_prob_folds.append(y_prob)
        
        print(f"Fold {fold+1} | AUC: {auc:.3f} | Sens: {sensitivity:.3f} | Spec: {specificity:.3f}")
    
    print("\n===== Final Results =====")
    print(f"Mean AUC: {np.mean(auc_scores):.3f} ± {np.std(auc_scores):.3f}")
    print(f"Mean Sensitivity: {np.mean(sensitivities):.3f}")
    print(f"Mean Specificity: {np.mean(specificities):.3f}")
    
    return trained_model, y_true_folds, y_prob_folds, auc_scores, sensitivities, specificities