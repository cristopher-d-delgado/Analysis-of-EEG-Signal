import re
from pathlib import Path
from collections import defaultdict
import numpy as np
from eeg_data_engineer import epoch_eeg, extract_features
from eeg_preprocess import load_and_preprocess

def parse_experiments(raw_path):
    pattern_with_session = re.compile(r"s(\d+)_ex(\d+)_s(\d+)")
    pattern_no_session = re.compile(r"s(\d+)_ex(\d+)")

    experiments = defaultdict(list)

    raw = Path(raw_path)

    for file in raw.glob("*.txt"):
        filename = file.stem

        match = pattern_with_session.search(filename)

        if match:
            subject, experiment, session = match.groups()
        else:
            match = pattern_no_session.search(filename)
            if match:
                subject, experiment = match.groups()
                session = None
            else:
                continue

        ex_key = f"ex{int(experiment):02d}"

        experiments[ex_key].append({
            "subject": int(subject),
            "session": int(session) if session is not None else None,
            "path": file
        })

    return experiments

def build_dataset(
    experiments,
    experiment_to_label,
    trim_df,
    fs,
):
    """
    Build feature matrix, labels, and subject groups from experiment files.
    """

    X_all = []
    y_all = []
    groups_all = []

    for ex_key, label in experiment_to_label.items():

        if ex_key not in experiments:
            continue

        for meta in experiments[ex_key]:

            subject = meta["subject"]
            session = meta["session"]
            file_path = meta["path"]

            # ---- Load + preprocess ----
            eeg_proc = load_and_preprocess(
                file_path,
                trim_df,
                subject=subject,
                experiment=int(ex_key[2:]),  # Extract numeric part from "ex01"
                session=session,
            )

            # ---- Epoch ----
            epochs = epoch_eeg(eeg_proc, fs=fs)

            # ---- Feature extraction ----
            X = extract_features(epochs, fs=fs)

            # ---- Labels ----
            y = np.full(len(X), label)

            # ---- Subject groups ----
            groups = np.full(len(X), subject)

            X_all.append(X)
            y_all.append(y)
            groups_all.append(groups)

    X_all = np.vstack(X_all)
    y_all = np.concatenate(y_all)
    groups_all = np.concatenate(groups_all)

    return X_all, y_all, groups_all