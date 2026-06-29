"""CLI: turn a CSV of y_true,y_score into a markdown eval report."""
from __future__ import annotations
import argparse
import csv
import sys
from .report import evaluate, to_markdown


def main(argv=None):
    p = argparse.ArgumentParser(prog="mleval", description="Binary-classifier eval report from a CSV (columns: y_true,y_score).")
    p.add_argument("csv", help="CSV file with a header row containing y_true and y_score")
    p.add_argument("-t", "--threshold", type=float, default=0.5)
    p.add_argument("-o", "--out", default=None, help="write markdown here instead of stdout")
    a = p.parse_args(argv)
    yt, ys = [], []
    with open(a.csv, newline="") as f:
        for row in csv.DictReader(f):
            yt.append(int(float(row["y_true"])))
            ys.append(float(row["y_score"]))
    md = to_markdown(evaluate(yt, ys, a.threshold))
    if a.out:
        with open(a.out, "w") as f:
            f.write(md)
    else:
        sys.stdout.write(md + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
