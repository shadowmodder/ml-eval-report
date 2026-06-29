"""mleval: lightweight, dependency-light evaluation reports for binary classifiers."""
from .metrics import (confusion_matrix, precision, recall, f1, accuracy,
                      roc_curve, pr_curve, auc, roc_auc, threshold_sweep,
                      calibration_bins)
from .report import evaluate, to_markdown

__all__ = ["confusion_matrix", "precision", "recall", "f1", "accuracy",
           "roc_curve", "pr_curve", "auc", "roc_auc", "threshold_sweep",
           "calibration_bins", "evaluate", "to_markdown"]
__version__ = "0.1.0"
