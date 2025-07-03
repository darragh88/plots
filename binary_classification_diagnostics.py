
"""
binary_classification_diagnostics.py
------------------------------------

Utility functions (and command‑line entry point) for evaluating binary
classification outputs produced by your GNN model.

• Accepts a *logit* DataFrame (shape: [timestamps × nodes]) and a matching
  ground‑truth label DataFrame (0/1).

• Computes:
    – ROC‑AUC
    – Average Precision (area under PR curve)
    – Confusion matrix, classification report
    – Plots ROC curve, PR curve, and confusion matrix heat‑map

Example
-------
>>> import pandas as pd
>>> from binary_classification_diagnostics import diagnostics_from_frames
>>> logits = pd.read_pickle("logits.pkl")   # or CSV
>>> y_true = pd.read_pickle("y_true.pkl")
>>> diagnostics_from_frames(logits, y_true, threshold=0.5)

Command‑line
------------
$ python binary_classification_diagnostics.py logits.csv y_true.csv
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    average_precision_score, precision_recall_curve,
    confusion_matrix, classification_report
)

def _flatten(df: pd.DataFrame):
    "Return values flattened to 1‑D numpy array."
    return df.values.ravel()

def diagnostics_from_frames(
    df_logits: pd.DataFrame,
    df_true:   pd.DataFrame,
    threshold: float = 0.5,
    show_plot: bool = True
) -> dict:
    """
    Compute metrics + plots from DataFrames that share index / columns.

    Returns a dict with keys: 'auc', 'ap', 'cm', 'report'.
    """
    assert df_logits.shape == df_true.shape, "Shape mismatch logits vs labels"

    y_logit = _flatten(df_logits)
    y_true  = _flatten(df_true).astype(int)

    y_proba = 1 / (1 + np.exp(-y_logit))
    y_pred  = (y_proba > threshold).astype(int)

    auc  = roc_auc_score(y_true, y_proba)
    ap   = average_precision_score(y_true, y_proba)
    cm   = confusion_matrix(y_true, y_pred)
    rep  = classification_report(y_true, y_pred, digits=4, output_dict=False)

    print(f"AUC  : {auc:.4f}")
    print(f"AP   : {ap:.4f}")
    print(rep)

    if show_plot:
        fig, ax = plt.subplots(1, 3, figsize=(18, 4))

        # ROC
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        ax[0].plot(fpr, tpr, label=f"AUC = {auc:.3f}")
        ax[0].plot([0,1],[0,1],"--",color="gray")
        ax[0].set(xlabel="False Positive Rate", ylabel="True Positive Rate",
                  title="ROC curve")
        ax[0].legend()

        # PR
        prec, rec, _ = precision_recall_curve(y_true, y_proba)
        ax[1].plot(rec, prec, label=f"AP = {ap:.3f}")
        ax[1].set(xlabel="Recall", ylabel="Precision",
                  title="Precision‑Recall curve")
        ax[1].legend()

        # Confusion matrix
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    cbar=False, ax=ax[2])
        ax[2].set(xlabel="Predicted", ylabel="True",
                  title=f"Confusion matrix (thr={threshold})")

        plt.tight_layout()
        plt.show()

    return {"auc": auc, "ap": ap, "cm": cm, "report": rep}

def _cli():
    if len(sys.argv) != 3:
        print("Usage: python binary_classification_diagnostics.py logits.csv y_true.csv")
        sys.exit(1)
    logits_path, y_path = map(Path, sys.argv[1:])
    df_logits = pd.read_csv(logits_path, index_col=0)
    df_true   = pd.read_csv(y_path,    index_col=0)
    diagnostics_from_frames(df_logits, df_true)

if __name__ == "__main__":
    _cli()
