import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch
from pathlib import Path
import matplotlib.pyplot as plt

# =========================================
# Constants 
# =========================================
FS = 200  # Sampling rate in Hz
CANONICAL_CHANNELS = ['T7', 'F8', 'Cz', 'P4']  # EEG channels
SCALE_FACTOR = 15686 / 8388607  # µV per LSB (Ganglion ADC)

# =========================================
# Paths to data
# =========================================
raw_dir = Path("data/auditory-evoked-potential-eeg-biometric-dataset-1.0.0/Raw_Data")  # Raw EEG Signal
trim_file = Path("data/auditory-evoked-potential-eeg-biometric-dataset-1.0.0/data_trim.csv")  # Trim points
filtered_dir = Path("data/auditory-evoked-potential-eeg-biometric-dataset-1.0.0/Filtered_Data")  # Reference data

# =========================================
# Trim Utilities
# =========================================
def load_trim_file(trim_file):
    """
    Load the data_trim.csv file containing segmentation indices.

    Parameters
    ----------
    trim_file : str or Path
        Path to data_trim.csv.

    Returns
    -------
    trim_df : pandas.DataFrame
        DataFrame containing trimming information.
    """
    trim_df = pd.read_csv(trim_file)
    expected_cols = {'Subject', 'Experment', 'session', 'From (n)', 'To (n)'}
    missing = expected_cols - set(trim_df.columns)

    if missing:
        raise ValueError(f"Missing columns in data_trim.csv: {missing}")
    
    # Forward fill missing subject/experiment values
    trim_df['Subject'] = trim_df['Subject'].ffill().astype(int)
    trim_df['Experment'] = trim_df['Experment'].ffill().astype(int)
    trim_df['session'] = trim_df['session'].astype(int)
    
    return trim_df

def get_trim_indices(trim_df, subject, experiment, session):
    """
    Retrieve start and end sample indices for segmented EEG.

    Returns
    -------
    start, end : int
        Sample indices for trimming raw EEG.
    """
    row = trim_df[
        (trim_df['Subject'] == subject) &
        (trim_df['Experment'] == experiment) &
        (trim_df['session'] == session)
    ]
    if row.empty:
        raise ValueError(f"No trim info for Subject {subject}, Experiment {experiment}, Session {session}")
    start_sample = int(row['From (n)'].values[0])
    end_sample = int(row['To (n)'].values[0])
    return start_sample, end_sample

# =========================================
# Preprocessing
# =========================================
# def remove_dc_offset(eeg):
#     """Remove DC offset (zero-mean per channel)"""
#     return eeg - np.mean(eeg, axis=0, keepdims=True)

def bandpass_1_40hz(eeg, fs=FS):
    """
    Apply 1st-order Butterworth bandpass filter (1–40 Hz).
    Ensures we focus on the EEG frequency bands relevant to auditory evoked potentials.

    Matches the filtering described in the dataset README.
    """
    low, high = 1, 40
    b, a = butter(N=1, Wn=[low/(fs/2), high/(fs/2)], btype='band')
    return filtfilt(b, a, eeg, axis=0)

def notch_50hz(eeg, fs=FS):
    """
    Apply 50 Hz notch filter (Q=30).

    Removes power line interference as described in the study.
    """
    f0 = 50
    Q = 30
    b, a = iirnotch(f0/(fs/2), Q)
    return filtfilt(b, a, eeg, axis=0)

def preprocess_eeg(eeg):
    """
    Full preprocessing pipeline:
        1. DC offset removal
        2. 1–40 Hz bandpass
        3. 50 Hz notch filter

    Returns
    -------
    eeg : np.ndarray
        Preprocessed EEG (samples × channels)
    """
    eeg = bandpass_1_40hz(eeg)
    eeg = notch_50hz(eeg)    
    return eeg

# def rereference_average(eeg):
#     """Average reference across channels. Removes channel biases."""
#     return eeg - np.mean(eeg, axis=1, keepdims=True)

# =========================================
# LOAD + PREPROCESS
# =========================================
def load_and_preprocess(raw_file, trim_df, subject, experiment, session):
    """
    Load raw EEG file, trim to clean 2-minute segment,
    scale to µV, and apply preprocessing filters.

    Parameters
    ----------
    raw_file : str or Path
        Path to raw EEG CSV file.
    trim_df : pandas.DataFrame
        Trim index table.
    subject : int
    experiment : int
    session : int

    Returns
    -------
    eeg : np.ndarray
        Cleaned EEG signal (samples × 4 channels)
    """
    df = pd.read_csv(raw_file)
    df.columns = df.columns.str.strip()

    raw_map = {
        'EXG Channel 0': 'T7',
        'EXG Channel 1': 'F8',
        'EXG Channel 2': 'Cz',
        'EXG Channel 3': 'P4'
    }
    df = df.rename(columns=raw_map)

    eeg = df[CANONICAL_CHANNELS].to_numpy() * SCALE_FACTOR

    start, end = get_trim_indices(trim_df, subject, experiment, session)
    eeg = eeg[start:end]

    eeg = preprocess_eeg(eeg)

    return eeg

# =========================================
# Gain Matching (Validation Only)
# =========================================
def match_reference_gain(my_eeg, ref_eeg, method="max"):
    """
    Match amplitude scale of processed EEG to reference EEG.

    Parameters
    ----------
    method : str
        'rms' or 'max'

    Returns
    -------
    scaled_eeg : np.ndarray
    gain : np.ndarray
        Channel-wise gain factors applied.
    """
    if method == "rms":
        my_amp = np.sqrt(np.mean(my_eeg**2, axis=0))
        ref_amp = np.sqrt(np.mean(ref_eeg**2, axis=0))
    elif method == "max":
        my_amp = np.max(np.abs(my_eeg), axis=0)
        ref_amp = np.max(np.abs(ref_eeg), axis=0)
    else:
        raise ValueError("method must be 'rms' or 'max'")

    gain = ref_amp / my_amp
    return my_eeg * gain, gain

def validate_against_reference(my_eeg, reference_file, apply_gain=False, method="rms"):
    """
    Compare preprocessed EEG against provided Filtered_Data file.

    Computes:
        - Pearson correlation
        - Mean absolute difference

    Optionally applies gain normalization before comparison.

    Returns
    -------
    validated_eeg : np.ndarray
        Possibly gain-adjusted EEG (for inspection only).
    """
    df_ref = pd.read_csv(reference_file)
    ref = df_ref[CANONICAL_CHANNELS].to_numpy()

    n = min(len(my_eeg), len(ref))
    my_eeg = my_eeg[:n]
    ref = ref[:n]

    if apply_gain:
        my_eeg, gain = match_reference_gain(my_eeg, ref, method=method)
        print(f"\nGain factors applied ({method}): {gain}")

    print("\nValidation Results")
    for ch, name in enumerate(CANONICAL_CHANNELS):
        corr = np.corrcoef(my_eeg[:, ch], ref[:, ch])[0, 1]
        diff = np.mean(np.abs(my_eeg[:, ch] - ref[:, ch]))
        print(f"{name}: Corr={corr:.6f} | MeanAbsDiff={diff:.4f} µV")

    return my_eeg

# # =========================================
# # Load EEG
# # =========================================
# def load_and_preprocess_eeg(
#         raw_file, trim_df, subject, 
#         experiment, session, 
#         reference_file=None, 
#         apply_gain_match=False, 
#         gain_method="rms",
#         apply_avg_ref=False
#     ):
#     """
#     Load raw EEG file and preprocess to match PhysioNet Filtered_Data.
    
#     Returns:
#     - sample_idx: np.ndarray of sample indices
#     - eeg: np.ndarray of preprocessed EEG (n_samples x 4 channels)
#     """
#     # Load raw file
#     df = pd.read_csv(raw_file)
#     df.columns = df.columns.str.strip()

#     # Map ADC channels to canonical names
#     raw_channel_map = {
#         'EXG Channel 0':'T7', 
#         'EXG Channel 1':'F8', 
#         'EXG Channel 2':'Cz', 
#         'EXG Channel 3':'P4'
#     }
#     df = df.rename(columns=raw_channel_map)

#     # Check channels
#     missing = set(CANONICAL_CHANNELS) - set(df.columns)
#     if missing:
#         raise ValueError(f"Missing channels in raw file: {missing}")

#     # Extract sample indices and EEG
#     sample_idx = df.iloc[:, 0].to_numpy()
#     eeg = df[CANONICAL_CHANNELS].to_numpy()

#     # --- Scale ADC → µV ---
#     eeg = eeg * SCALE_FACTOR

#     # --- Remove DC offset and apply Filters---
#     eeg = remove_dc_offset(eeg)
#     eeg = preprocess_eeg(eeg)

#     # Average reference
#     if apply_avg_ref: 
#         eeg = rereference_average(eeg)

#     # --- Trim to clean 2-minute segment ---
#     start, end = get_trim_indices(trim_df, subject, experiment, session)
#     sample_idx = sample_idx[start:end]
#     eeg = eeg[start:end, :]
    
#     # --- Perform Gain Match ---
#     if apply_gain_match: 
#         if reference_file is None:
#             raise ValueError("reference_file must be provided for gain matching.")
        
#         ref = load_filtered_reference(reference_file)
#         ref = ref[:len(eeg)] # align length

#         eeg, gain = match_reference_gain(eeg, ref, method=gain_method)
#     return sample_idx, eeg

# # =========================================
# # Load Filtered Reference
# # =========================================
# def load_filtered_reference(file_path):
#     """Load PhysioNet filtered EEG in canonical channel order."""
#     df = pd.read_csv(file_path)
#     missing = set(CANONICAL_CHANNELS) - set(df.columns)
#     if missing:
#         raise ValueError(f"Missing channels in filtered file: {missing}")
#     return df[CANONICAL_CHANNELS].to_numpy()

# =========================================
# Plotting
# =========================================
def plot_eeg_signal(eeg, channel=0, window=None):
    """Plot a single EEG channel by itself"""
    ch_name = CANONICAL_CHANNELS[channel]
    signal = eeg[:, channel]
    if window is not None:
        signal = signal[:window]
    plt.figure(figsize=(12,4))
    plt.plot(signal, label=ch_name, color='tab:blue')
    plt.title(f"EEG Signal: {ch_name}")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude (µV)")
    plt.legend()
    plt.tight_layout()
    plt.show()

def compare_with_reference(eeg, reference_file, channel=0, window=None, show_diff=True, offset_gap=50, save_fig=False, save_dir=None):
    """
    Compare one EEG channel to PhysioNet filtered reference.

    Parameters
    ----------
    my_eeg : np.ndarray
        Preprocessed EEG (samples x channels)
    reference_file : str or Path
        Path to PhysioNet Filtered_Data CSV
    channel : int
        Channel index (0=T7, 1=F8, 2=Cz, 3=P4)
    window : int or None
        Number of samples to display
    show_diff : bool
        Whether to show difference plot below the signals
    """
    # Define reference channel 
    df_ref = pd.read_csv(reference_file)
    ch_name = CANONICAL_CHANNELS[channel]

    ref = df_ref[ch_name].to_numpy()
    my = eeg[:, channel]

    # Zoom window if requested
    if window is not None:
        my = my[:window]
        ref = ref[:window]

    n = min(len(my), len(ref))
    my = my[:n]
    ref = ref[:n]

    # Apply vertical offset for plotting
    my_offset = my + offset_gap

    if show_diff:
        # Compute difference (my - ref)
        diff = my - ref
        
        # Create figure with 2 subplots: Signals on top, Difference below
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(12,6), facecolor='white')
        
        # Plot Signals on top and Difference below
        ax[0].plot(my_offset, label=f"Pipeline Output {ch_name} + {offset_gap} uV", alpha=0.7)
        ax[0].plot(ref, label=f"Reference {ch_name} uV", alpha=0.7)
        ax[0].set_xlabel("Sample")
        ax[0].set_ylabel("Amplitude (µV)")
        ax[0].set_title(f"{ch_name} Comparison with PhysioNet Reference")
        ax[0].legend(bbox_to_anchor=(1, 1), loc='upper left')

        ax[1].plot(diff)
        ax[1].set_title(f"Difference: {ch_name} (My − Ref)")
        ax[1].set_xlabel("Sample")
        ax[1].set_ylabel("Amplitude Difference (µV)")

        plt.tight_layout()

        if save_fig:
            assert save_dir is not None, "save_dir must be provided if save_fig is True"
            fig.savefig(save_dir)
            plt.show()
        
        if save_fig == False or save_fig: 
            plt.show()
        
        print(f"{ch_name} max diff: {np.max(np.abs(diff)):.5f} µV | mean diff: {np.mean(diff):.5f} µV")
    
    if show_diff == False:
        plt.figure(figsize=(12,4))
        plt.plot(my_offset, label=f"My Ch {ch_name} + {offset_gap} uV", alpha=0.7)
        plt.plot(ref, label=f"Ref Ch {ch_name} uV", alpha=0.7)
        plt.title(f"{ch_name} Comparison")
        plt.xlabel("Sample")
        plt.ylabel("Amplitude (µV)")
        plt.legend()
        plt.tight_layout()
        plt.show()
    
