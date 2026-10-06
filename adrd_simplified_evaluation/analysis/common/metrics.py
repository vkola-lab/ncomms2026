"""
Vectorised metrics. Every function accepts 1D inputs (point estimate) or
2D inputs of shape (n_resamples, n_rows) and reduces over the last axis.
"""

import numpy as np
import pandas as pd


# Label encoding

def build_label_encoding(group: pd.DataFrame):
    """
    Integer encoding for ground-truth and predicted option texts in `group`.
    Returns (all_cats, cat_dtype, invalid_code).
    """
    gt_cats = group["ground_truth_text"].astype("category").cat.categories.tolist()
    pred_cats = group["prediction_text"].astype("category").cat.categories.tolist()
    all_cats = sorted(set(gt_cats + pred_cats))
    if "invalid" not in all_cats:
        all_cats.append("invalid")

    cat_dtype = pd.CategoricalDtype(categories=all_cats, ordered=False)
    return all_cats, cat_dtype, all_cats.index("invalid")


def encode(series: pd.Series, cat_dtype) -> pd.Series:
    return series.astype(cat_dtype).cat.codes.astype(np.int16)


def gt_label_codes(y_true: np.ndarray, invalid_code: int) -> np.ndarray:
    """Classes present in the ground truth, excluding 'invalid'."""
    codes = np.unique(y_true)
    return codes[codes != invalid_code]


# Per-class metrics

def metric_calc(y_true, y_pred, label_code, metric: str):
    """Precision, recall or F1 for one class. Zero division gives 0."""
    tp = np.sum((y_pred == label_code) & (y_true == label_code), axis=-1)

    if metric == "precision":
        den = np.sum(y_pred == label_code, axis=-1)
    elif metric == "recall":
        den = np.sum(y_true == label_code, axis=-1)
    elif metric == "f1":
        precision_den = np.sum(y_pred == label_code, axis=-1)
        recall_den = np.sum(y_true == label_code, axis=-1)
        precision = np.divide(tp, precision_den, out=np.zeros_like(tp, dtype=float),
                              where=precision_den != 0)
        recall = np.divide(tp, recall_den, out=np.zeros_like(tp, dtype=float),
                           where=recall_den != 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.divide(2 * precision * recall, precision + recall,
                             out=np.zeros_like(tp, dtype=float),
                             where=(precision + recall) != 0)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return np.divide(tp, den, out=np.zeros_like(tp, dtype=float), where=den != 0)


def macro_metric_calc(y_true, y_pred, label_codes, metric: str):
    """
    Macro average over `label_codes`. A class with no true rows in a resample
    is left out of that resample's average instead of counting as 0.
    """
    label_codes = np.asarray(label_codes, dtype=np.int16)
    if label_codes.size == 0:
        return np.zeros(y_true.shape[0], dtype=float) if y_true.ndim > 1 else 0.0

    per_label = []
    for lbl in label_codes:
        v = metric_calc(y_true, y_pred, int(lbl), metric).astype(float)
        present = np.sum(y_true == lbl, axis=-1) > 0
        per_label.append(np.where(present, v, np.nan))

    return np.nanmean(np.stack(per_label, axis=-1), axis=-1)


def macro_accuracy(correct, benchmark_codes, benchmark_ids):
    """
    Accuracy macro-averaged over benchmarks.
    correct: (n,) or (B, n); benchmark_codes: (n,) or (B, n).
    """
    correct = np.asarray(correct)
    benchmark_codes = np.asarray(benchmark_codes)
    benchmark_ids = np.asarray(benchmark_ids)

    if correct.ndim == 1:
        bench_accs = []
        for b in benchmark_ids:
            m = benchmark_codes == b
            if m.any():
                bench_accs.append(correct[m].mean())
        return np.mean(bench_accs)

    if correct.ndim == 2:
        B, n = correct.shape
        if benchmark_codes.ndim == 1:
            benchmark_codes = np.broadcast_to(benchmark_codes, (B, n))
        bench_accs = []
        for b in benchmark_ids:
            m = benchmark_codes == b
            counts = m.sum(axis=1)
            sums = (correct * m).sum(axis=1)
            bench_accs.append(
                np.divide(sums, counts, out=np.full(B, np.nan, dtype=float), where=counts > 0)
            )
        return np.nanmean(np.stack(bench_accs, axis=1), axis=1)

    raise ValueError("correct must be 1D or 2D")


# Statistics in "values first" form, for the permutation test
# (the swapped array is the first argument; everything else is fixed).

def per_class_stat(y_pred, y_true, label_code, metric):
    return metric_calc(y_true, y_pred, label_code, metric)


def macro_stat(y_pred, y_true, label_codes, metric):
    return macro_metric_calc(y_true, y_pred, label_codes, metric)


def mean_stat(values):
    return values.mean(axis=-1)
