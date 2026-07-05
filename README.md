[![CI](https://github.com/shadowmodder/ml-eval-report/actions/workflows/ci.yml/badge.svg)](https://github.com/shadowmodder/ml-eval-report/actions/workflows/ci.yml)

# ml-eval-report

Lightweight, dependency-light evaluation for **binary classifiers**. NumPy in, plain-Python and markdown out — no heavyweight plotting or framework lock-in. Useful as a drop-in report step in any training or CI pipeline.

## Install
```bash
pip install -e .
```

## Library
```python
from mleval import evaluate, to_markdown, roc_auc, threshold_sweep

y_true  = [0, 0, 1, 1]
y_score = [0.1, 0.3, 0.7, 0.9]

print(to_markdown(evaluate(y_true, y_score, threshold=0.5)))
print("AUC:", roc_auc(y_true, y_score))
for row in threshold_sweep(y_true, y_score):
    print(row)
```

## CLI
```bash
mleval predictions.csv -t 0.5 -o REPORT.md   # CSV needs columns: y_true,y_score
```

## What's inside
- Confusion matrix, precision / recall / F1 / accuracy
- ROC and PR curves + trapezoidal AUC
- Threshold sweep table for choosing an operating point
- Calibration bins (mean predicted vs. empirical positive rate)

MIT licensed.
