"""Core metric primitives. NumPy in, plain Python out."""
from __future__ import annotations
import numpy as np


def _counts(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    return tp, fp, tn, fn


def _safe(a, b):
    return a / b if b else 0.0


def confusion_matrix(y_true, y_pred):
    tp, fp, tn, fn = _counts(y_true, y_pred)
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def precision(y_true, y_pred):
    tp, fp, _, _ = _counts(y_true, y_pred)
    return _safe(tp, tp + fp)


def recall(y_true, y_pred):
    tp, _, _, fn = _counts(y_true, y_pred)
    return _safe(tp, tp + fn)


def f1(y_true, y_pred):
    p, r = precision(y_true, y_pred), recall(y_true, y_pred)
    return _safe(2 * p * r, p + r)


def accuracy(y_true, y_pred):
    tp, fp, tn, fn = _counts(y_true, y_pred)
    return _safe(tp + tn, tp + fp + tn + fn)


def roc_curve(y_true, y_score):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    order = np.argsort(-y_score)
    yt = y_true[order]
    P, N = int(yt.sum()), int(len(yt) - yt.sum())
    tps = np.cumsum(yt)
    fps = np.cumsum(1 - yt)
    tpr = tps / P if P else np.zeros_like(tps, dtype=float)
    fpr = fps / N if N else np.zeros_like(fps, dtype=float)
    tpr = np.concatenate([[0.0], tpr])
    fpr = np.concatenate([[0.0], fpr])
    return fpr, tpr


def pr_curve(y_true, y_score):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    order = np.argsort(-y_score)
    yt = y_true[order]
    tps = np.cumsum(yt)
    fps = np.cumsum(1 - yt)
    P = int(yt.sum())
    prec = tps / np.maximum(tps + fps, 1)
    rec = tps / P if P else np.zeros_like(tps, dtype=float)
    return rec, prec


def auc(x, y):
    return float(np.trapz(np.asarray(y, float), np.asarray(x, float)))


def roc_auc(y_true, y_score):
    fpr, tpr = roc_curve(y_true, y_score)
    return auc(fpr, tpr)


def threshold_sweep(y_true, y_score, thresholds=None):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    if thresholds is None:
        thresholds = np.unique(y_score)
    rows = []
    for t in thresholds:
        y_pred = (y_score >= t).astype(int)
        rows.append({"threshold": float(t), "precision": precision(y_true, y_pred),
                     "recall": recall(y_true, y_pred), "f1": f1(y_true, y_pred)})
    return rows


def calibration_bins(y_true, y_score, n_bins=10):
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (y_score >= lo) & (y_score < hi) if i < n_bins - 1 else (y_score >= lo) & (y_score <= hi)
        if not mask.any():
            continue
        out.append({"bin_lo": float(lo), "bin_hi": float(hi), "count": int(mask.sum()),
                    "mean_pred": float(y_score[mask].mean()), "frac_pos": float(y_true[mask].mean())})
    return out
