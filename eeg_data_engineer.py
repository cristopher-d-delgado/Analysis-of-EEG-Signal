import numpy as np
from scipy.signal import welch
from scipy.stats import entropy

def epoch_eeg(
        eeg: np.ndarray, 
        fs:int, 
        epoch_length_sec: float = 2.0, 
        overlap_sec: float= 1.0
        ) -> np.ndarray:
    """
    Segment continuous EEG into overlapping epochs.

    Parameters
    ----------
    eeg : np.ndarray
        Continuous EEG signal of shape (n_samples, n_channels).
    fs : int
        Sampling frequency in Hz.
    epoch_length_sec : float, optional
        Length of each epoch in seconds (default = 2.0).
    overlap_sec : float, optional
        Overlap between consecutive epochs in seconds (default = 1.0).

    Returns
    -------
    epochs : np.ndarray
        Array of shape (n_epochs, epoch_samples, n_channels).
    """

    # Define np.array check 
    if eeg.ndim != 2:
        raise ValueError("EEG must be a 2D array of shape (n_samples, n_channels)")
    
    epoch_samples = int(epoch_length_sec * fs)
    step_samples = int((epoch_length_sec - overlap_sec) * fs)

    # Define check for positive step size
    if step_samples <= 0:
        raise ValueError("Overlap must be less than epoch length to ensure positive step size.")
    
    n_samples, n_channels = eeg.shape
    epochs = []

    for start in range(0, n_samples - epoch_samples + 1, step_samples):
        end = start + epoch_samples
        epochs.append(eeg[start:end, :])
    
    epochs = np.stack(epochs)  # Shape: (n_epochs, epoch_samples, n_channels)
    
    return epochs

def compute_band_powers(signal, fs, bands):
    """
    Compute absolute and relative band power using Welch PSD.

    Parameters
    ----------
    signal : np.ndarray
        1D EEG signal (samples,)
    fs : int
        Sampling frequency
    bands : dict
        Dictionary of frequency bands {name: (low, high)}

    Returns
    -------
    abs_powers : dict
        Absolute band power per band
    rel_powers : dict
        Relative band power per band
    """
    freqs, psd = welch(signal, fs=fs, nperseg=fs*2)

    total_power = np.trapezoid(psd, freqs)

    abs_powers = {}
    rel_powers = {}

    for band_name, (low, high) in bands.items():
        idx = np.logical_and(freqs >= low, freqs <= high)
        band_power = np.trapezoid(psd[idx], freqs[idx])

        abs_powers[band_name] = band_power
        rel_powers[band_name] = band_power / total_power if total_power > 0 else 0

    return abs_powers, rel_powers


def hjorth_parameters(signal):
    """
    Compute Hjorth Activity, Mobility, and Complexity.

    Returns
    -------
    activity : float
    mobility : float
    complexity : float
    """
    first_deriv = np.diff(signal)
    second_deriv = np.diff(first_deriv)

    var_zero = np.var(signal)
    var_d1 = np.var(first_deriv)
    var_d2 = np.var(second_deriv)

    activity = var_zero
    mobility = np.sqrt(var_d1 / var_zero) if var_zero > 0 else 0
    complexity = (
        np.sqrt(var_d2 / var_d1) / mobility
        if var_d1 > 0 and mobility > 0
        else 0
    )

    return activity, mobility, complexity


def spectral_entropy_feature(signal, fs):
    """
    Compute normalized spectral entropy.
    """
    freqs, psd = welch(signal, fs=fs)
    psd_norm = psd / np.sum(psd) if np.sum(psd) > 0 else psd
    return entropy(psd_norm)

def extract_features(epochs, fs):
    """
    Extract features from EEG epochs.

    Parameters
    ----------
    epochs : np.ndarray
        Shape (n_epochs, epoch_samples, n_channels)
    fs : int
        Sampling frequency

    Returns
    -------
    features : np.ndarray
        Shape (n_epochs, n_features)
    """

    bands = {
        "delta": (1, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta": (13, 30),
        "gamma": (30, 60),
    }

    n_epochs, _, n_channels = epochs.shape
    all_features = []

    for ep in range(n_epochs):
        epoch_features = []

        for ch in range(n_channels):
            signal = epochs[ep, :, ch]

            # --- Frequency features ---
            abs_powers, rel_powers = compute_band_powers(signal, fs, bands)

            for band in bands:
                epoch_features.append(abs_powers[band])
            for band in bands:
                epoch_features.append(rel_powers[band])

            # --- Time-domain features ---
            activity, mobility, complexity = hjorth_parameters(signal)
            epoch_features.extend([activity, mobility, complexity])

            # --- Complexity feature ---
            spec_ent = spectral_entropy_feature(signal, fs)
            epoch_features.append(spec_ent)

        all_features.append(epoch_features)

    return np.array(all_features)

def get_feature_names(n_channels):
    """
    Generate feature names in the same order as extract_features().
    """
    bands = ["delta", "theta", "alpha", "beta", "gamma"]
    names = []

    for ch in range(n_channels):
        prefix = f"ch{ch}"

        # Absolute power
        for band in bands:
            names.append(f"{prefix}_abs_{band}")

        # Relative power
        for band in bands:
            names.append(f"{prefix}_rel_{band}")

        # Hjorth
        names.append(f"{prefix}_hjorth_activity")
        names.append(f"{prefix}_hjorth_mobility")
        names.append(f"{prefix}_hjorth_complexity")

        # Entropy
        names.append(f"{prefix}_spectral_entropy")

    return names