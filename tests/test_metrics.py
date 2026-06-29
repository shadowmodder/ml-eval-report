import numpy as np
from mleval import (confusion_matrix, precision, recall, f1, accuracy,
                    roc_auc, threshold_sweep, evaluate, to_markdown)


def test_perfect_separation():
    y = [0, 0, 1, 1]
    s = [0.1, 0.2, 0.8, 0.9]
    assert abs(roc_auc(y, s) - 1.0) < 1e-9


def test_confusion_and_rates():
    yt = [1, 1, 0, 0]
    yp = [1, 0, 0, 0]
    c = confusion_matrix(yt, yp)
    assert c == {"tp": 1, "fp": 0, "tn": 2, "fn": 1}
    assert precision(yt, yp) == 1.0
    assert recall(yt, yp) == 0.5
    assert abs(f1(yt, yp) - (2 / 3)) < 1e-9
    assert accuracy(yt, yp) == 0.75


def test_sweep_and_report():
    yt = [0, 1, 1, 0, 1]
    ys = [0.2, 0.6, 0.9, 0.4, 0.55]
    rows = threshold_sweep(yt, ys)
    assert all(set(r) == {"threshold", "precision", "recall", "f1"} for r in rows)
    md = to_markdown(evaluate(yt, ys, 0.5))
    assert "Evaluation Report" in md and "ROC AUC" in md
