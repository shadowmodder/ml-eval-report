"""Core metric primitives. NumPy in, plain Python out."""
from __future__ import annotations
import numpy as np

# np.trapz was removed in NumPy 2.0; np.trapezoid is the replacement
try:
    _trapezoid = np.trapezoid
except AttributeError:
    _trapezoid = np.trapz  # type: ignore[attr-defined]


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
    return float(_trapezoid(np.asarray(y, float), np.asarray(x, float)))


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


def brier_score(y_true, y_prob):
    """Mean squared error between predicted probabilities and binary outcomes."""
    return float(np.mean((np.asarray(y_prob, dtype=float) - np.asarray(y_true, dtype=float)) ** 2))


def ece(y_true, y_prob, n_bins=10):
    """Expected Calibration Error: bin-size-weighted mean |confidence - accuracy|."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    total = len(y_true)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    err = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi) if i < n_bins - 1 else (y_prob >= lo) & (y_prob <= hi)
        if not mask.any():
            continue
        conf = float(y_prob[mask].mean())
        acc = float(y_true[mask].mean())
        err += (int(mask.sum()) / total) * abs(conf - acc)
    return float(err)


def macro_f1(y_true, y_pred):
    """Macro-averaged F1 across all integer classes present in y_true."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    classes = np.unique(y_true)
    scores = []
    for c in classes:
        tp = int(np.sum((y_pred == c) & (y_true == c)))
        fp = int(np.sum((y_pred == c) & (y_true != c)))
        fn = int(np.sum((y_pred != c) & (y_true == c)))
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        scores.append(2 * p * r / (p + r) if (p + r) > 0 else 0.0)
    return float(np.mean(scores)) if scores else 0.0


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
