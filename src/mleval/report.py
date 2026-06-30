"""Assemble metrics into a portable report dict and render markdown."""
from __future__ import annotations
import numpy as np
from . import metrics


def evaluate(y_true, y_score, threshold=0.5):
    y_pred = (np.asarray(y_score, dtype=float) >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "confusion": metrics.confusion_matrix(y_true, y_pred),
        "precision": metrics.precision(y_true, y_pred),
        "recall": metrics.recall(y_true, y_pred),
        "f1": metrics.f1(y_true, y_pred),
        "accuracy": metrics.accuracy(y_true, y_pred),
        "roc_auc": metrics.roc_auc(y_true, y_score),
        "brier_score": metrics.brier_score(y_true, y_score),
        "ece": metrics.ece(y_true, y_score),
    }


def to_markdown(result):
    c = result["confusion"]
    return "\n".join([
        "# Evaluation Report", "",
        f"- Threshold: {result['threshold']:.3f}",
        f"- ROC AUC: {result['roc_auc']:.4f}",
        f"- Precision: {result['precision']:.4f}",
        f"- Recall: {result['recall']:.4f}",
        f"- F1: {result['f1']:.4f}",
        f"- Accuracy: {result['accuracy']:.4f}", "",
        "| | Pred + | Pred - |", "|---|---|---|",
        f"| **Actual +** | {c['tp']} | {c['fn']} |",
        f"| **Actual -** | {c['fp']} | {c['tn']} |", "",
    ])
