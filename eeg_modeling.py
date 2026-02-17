import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    confusion_matrix
)
from sklearn.pipeline import Pipeline


def evaluate_model_groupkfold(model, X, y, groups, n_splits=5, scale=True):
    """
    Evaluate a classifier using GroupKFold cross-validation.

    Parameters
    ----------
    model : sklearn classifier (must support predict_proba)
    X : feature matrix
    y : labels
    groups : subject IDs
    n_splits : number of folds
    scale : whether to apply StandardScaler

    Returns
    -------
    results : dict containing metrics and ROC data
    """
    # Define Kfold with groups
    gkf = GroupKFold(n_splits=n_splits)

    # Create empty lists to store results for each fold then compute mean at the end
    auc_scores = []
    sensitivities = []
    specificities = []

    mean_fpr = np.linspace(0, 1, 100)
    tprs = []

    # Optional Scaling via Pipeline
    if scale:
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", model)
        ])
    else:
        pipeline = model

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        # Define indices for train/test split based on groups (subjects)
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # ---- Scale and Train Model ----
        pipeline.fit(X_train, y_train)

        # ---- Predict ----
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        y_pred = pipeline.predict(X_test)

        # ---- AUC ----
        auc = roc_auc_score(y_test, y_prob)
        auc_scores.append(auc)
        
        # ---- ROC curve interpolation ----
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        tpr_interp = np.interp(mean_fpr, fpr, tpr)
        tpr_interp[0] = 0.0
        tprs.append(tpr_interp)


        # ---- ROC Curve ----
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        tpr_interp = np.interp(mean_fpr, fpr, tpr)
        tpr_interp[0] = 0.0
        tprs.append(tpr_interp)

        # ---- Confusion Matrix ----
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)

        sensitivities.append(sensitivity)
        specificities.append(specificity)

        print(f"Fold {fold+1} AUC: {auc:.3f}")
    

    # ---- Final Aggregation ----
    mean_auc = np.mean(auc_scores)
    std_auc = np.std(auc_scores)

    mean_sens = np.mean(sensitivities)
    mean_spec = np.mean(specificities)

    print("\n===== Final Results =====")
    print(f"Mean AUC: {mean_auc:.3f} ± {std_auc:.3f}")
    print(f"Mean Sensitivity: {mean_sens:.3f}")
    print(f"Mean Specificity: {mean_spec:.3f}")

    results = {
        "auc_scores": auc_scores,
        "mean_auc": mean_auc,
        "std_auc": std_auc,
        "sensitivities": sensitivities,
        "mean_sensitivity": mean_sens,
        "specificities": specificities,
        "mean_specificity": mean_spec,
        "mean_fpr": mean_fpr,
        "tprs": tprs
    }

    return results

###########################################################################
def plot_mean_roc(results):
    """
    Plot the mean ROC curve from cross-validation results produced by evaluate_model_groupkfold.
    """

    mean_tpr = np.mean(results["tprs"], axis=0)
    mean_tpr[-1] = 1.0

    mean_auc = results["mean_auc"]

    plt.figure()
    plt.plot(results["mean_fpr"], mean_tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Mean ROC Curve (AUC = {mean_auc:.3f})")

    plt.show()