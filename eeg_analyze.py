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


# # ============================================================
# # 1️⃣ PSD Example Function
# # ============================================================
# def plot_psd_example(eeg_eo, eeg_ec, fs=200, save_path=None):
#     f_eo, psd_eo = welch(eeg_eo.mean(axis=0), fs=fs, nperseg=fs*2)
#     f_ec, psd_ec = welch(eeg_ec.mean(axis=0), fs=fs, nperseg=fs*2)
    
#     plt.figure(figsize=(8,4))
#     plt.semilogy(f_eo, psd_eo, label="Eyes Open")
#     plt.semilogy(f_ec, psd_ec, label="Eyes Closed")
#     plt.xlim(0, 15)
#     plt.xlabel("Frequency (Hz)")
#     plt.ylabel("PSD (µV²/Hz)")
#     plt.title("PSD Example — Channel T7")
#     plt.legend()
#     plt.tight_layout()
#     if save_path:
#         plt.savefig(save_path)
#     plt.show()


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
    plt.figure(figsize=(6,4), facecolor='white')
    
    # Log transform for better visualization
    alpha_eo_log = np.log10(alpha_eo)
    alpha_ec_log = np.log10(alpha_ec)
    # Plot KDEs with fill
    sns.kdeplot(alpha_eo_log, label="Eyes Open", fill=True)
    sns.kdeplot(alpha_ec_log, label="Eyes Closed", fill=True)
    # Define upper x-limit based on 99th percentile of combined data
    upper = np.percentile(
        np.concatenate([alpha_eo, alpha_ec]), 99
    )
    #plt.xlim(0, upper)
    plt.xlabel("Alpha Power log10(µV²)")
    plt.ylabel("Density")
    plt.title("Alpha Power Distribution")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper right')
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


# # ============================================================
# # 4️⃣ Confusion Matrix
# # ============================================================
# def plot_confusion_matrix(y_true, y_pred, class_names=["Eyes Closed", "Eyes Open"], save_path=None):
#     """
#     Plot normalized confusion matrix with percentages.
#     Handles empty rows gracefully.
    
#     Parameters
#     ----------
#     y_true : np.ndarray
#         True labels
#     y_pred : np.ndarray
#         Predicted labels
#     class_names : list of str
#         Names for the classes
#     save_path : str or None
#         Where to save the figure
#     """
#     cm = confusion_matrix(y_true, y_pred, labels=[0,1])
#     cm_norm = cm.astype('float')
    
#     # Normalize each row by its sum; handle zero rows
#     row_sums = cm.sum(axis=1)[:, np.newaxis]
#     row_sums[row_sums == 0] = 1  # avoid division by zero
#     cm_norm /= row_sums
    
#     plt.figure(figsize=(5,5))
#     sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", cbar=False,
#                 xticklabels=class_names, yticklabels=class_names)
    
#     plt.xlabel("Predicted")
#     plt.ylabel("True")
#     plt.title("Normalized Confusion Matrix")
#     plt.tight_layout()
    
#     if save_path:
#         plt.savefig(save_path)
#     plt.show()


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

# ============================================================
# Spectogram Functions
# ============================================================
# --- Spectrogram for individual channel ---
from scipy.signal import spectrogram
from scipy.ndimage import gaussian_filter, median_filter

label = 15
title = 20 
ticks = 12
FS = 200 

# --- Spectrograms for all channels ---
def plot_channel_spectrograms(eeg, channels = ["T7", "F8", "Cz", "P4"], fs=FS, smoothing=None, sigma=1.0, median_size=(3,3), save_fig=False, save_path=None):
    """
    Create a spectrogram for each EEG channel.

    Parameters
    ----------
    eeg : np.ndarray
        Preprocessed EEG data (samples x channels)
    channels : list of str
        List of channel names corresponding to the columns in `eeg`
    fs : int
        Sampling frequency in Hz
    smoothing: str or None
        Type of smoothing to apply to spectrograms ('gaussian', 'median', or None)
    sigma: float
        Standard deviation for Gaussian smoothing (if smoothing='gaussian')
    median_size : tuple
        Size of the kernel for median filtering (if smoothing='median')
    
    Returns
    -------
    Plot of spectrograms for each channel.
    """
    # Check number of channels matches expected
    n_channels = eeg.shape[1]
    assert n_channels == 4, "This layout assumes exactly 4 channels."

    # Create a 2x2 grid of subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 6), sharex=True, sharey=True)
    axes = axes.flatten()

    # First compute all spectrograms so we can normalize color scale
    specs = []
    for ch in range(n_channels):
        f, t, Sxx = spectrogram(eeg[:, ch], fs=fs, nperseg=256, noverlap=128)
        Sxx_db = 10 * np.log10(Sxx + 1e-12)  # Convert to dB
        
        # Apply optional smoothing
        if smoothing == 'gaussian':
            Sxx_db = gaussian_filter(Sxx_db, sigma=sigma)  # Smooth the spectrogram
        elif smoothing == 'median':
            Sxx_db = median_filter(Sxx_db, size=median_size)  # Apply median filter
        elif smoothing is None:
            Sxx_db = Sxx_db  # No smoothing
        
        specs.append((f, t, Sxx_db))

    # Find global min/max for shared color scale
    vmin = min(np.min(Sxx) for _, _, Sxx in specs)
    vmax = max(np.max(Sxx) for _, _, Sxx in specs)

    # Plot each channel's spectrogram
    for ch, ax in enumerate(axes):
        f, t, Sxx_db = specs[ch]
        #Sxx_db = gaussian_filter(Sxx_db, sigma=1.0)  # Smooth the spectrogram
        im = ax.pcolormesh(t, f, Sxx_db, shading='gouraud',
                           vmin=vmin, vmax=vmax)

        ax.set_title(f"{channels[ch]}", fontsize=label)
        ax.set_ylim(0, 60)  # Focus on 0-60 Hz
        
        if ch % 2 == 0:
            ax.set_ylabel("Frequency (Hz)", fontsize=label)
        if ch >= 2:
            ax.set_xlabel("Time (s)", fontsize=label)
    
    
    # Leave space for colorbar on figure 
    fig.subplots_adjust(right=0.88)

    # Shared colorbar
    cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])  # [left, bottom, width, height]

    cbar = fig.colorbar(im, ax=axes, orientation='vertical',
                        fraction=0.02, pad=0.04, cax=cbar_ax)
    cbar.set_label("Power (dB)", fontsize=label)

    # Add Sup title
    plt.suptitle("EEG Spectrograms (0-60 Hz)", fontsize=title)
    plt.tick_params(axis='both', which='major', labelsize=ticks)
    
    # Save figure if requested
    if save_fig and save_path is not None:
        fig.savefig(save_path)
    
    plt.show()

# --- Mean spectrogram across channels ---
def plot_mean_spectrogram(eeg, fs, smoothing='gaussian', sigma=1, save_fig=False, save_path=None):
    """
    Compute and plot the mean spectrogram across all channels.

    Parameters
    ----------
    eeg : np.ndarray
        Preprocessed EEG data (samples x channels)
    fs : int
        Sampling frequency in Hz
    smoothing: str or None
        Type of smoothing to apply to spectrograms ('gaussian', 'median', or None)
    sigma: float
        Standard deviation for Gaussian smoothing (if smoothing='gaussian')
    median_size : tuple
        Size of the kernel for median filtering (if smoothing='median')
    
    """
    channel_spectrograms = []

    # Compute spectrogram per channel
    for ch in range(eeg.shape[1]):
        f, t, Sxx = spectrogram(
            eeg[:, ch],
            fs=fs,
            nperseg=256,
            noverlap=128
        )
        channel_spectrograms.append(Sxx)

        # Apply smoothing per-channel
    if smoothing == 'gaussian':
        Sxx = gaussian_filter(Sxx, sigma=sigma)   # smooth in freq & time
    elif smoothing == 'median':
        Sxx = median_filter(Sxx, size=(3,3))     # adjust kernel as needed
    elif smoothing is None:
        Sxx = Sxx  # No smoothing

        channel_spectrograms.append(Sxx)
    
    # Stack into 3D array: (channels, freqs, times)
    Sxx_stack = np.stack(channel_spectrograms, axis=0)

    # Average in linear power space
    Sxx_mean = np.mean(Sxx_stack, axis=0)

    # Convert to dB AFTER averaging
    Sxx_mean_db = 10 * np.log10(Sxx_mean + 1e-12)

    # Plot
    fig = plt.figure(figsize=(5, 3))
    im = plt.pcolormesh(t, f, Sxx_mean_db,
                        shading='gouraud')

    plt.ylim(0, 50)
    plt.xlabel("Time (s)", fontsize=label)
    plt.ylabel("Frequency (Hz)", fontsize=label)
    plt.title("Mean Spectrogram Across Channels", fontsize=title)
    plt.tick_params(axis='both', which='major', labelsize=ticks)

    cbar = plt.colorbar(im)
    cbar.set_label("Power (dB)", fontsize=label)

    plt.tight_layout()
    
    # Save figure if requested
    if save_fig and save_path is not None:
        fig.savefig(save_path)
    plt.show()
