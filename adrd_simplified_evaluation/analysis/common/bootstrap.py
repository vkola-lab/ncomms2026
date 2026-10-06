"""
Row-level bootstrap CIs (percentile, 2.5/97.5). The point estimate is always
the statistic on the original data.
"""

import re

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .config import N_BOOT
from .metrics import (
    build_label_encoding, encode, gt_label_codes, metric_calc, macro_metric_calc,
    macro_accuracy,
)


def stratified_indices(rng, n_boot: int, strata) -> np.ndarray:
    """
    Bootstrap row indices, shape (n_boot, n_rows). Rows are resampled within
    each stratum and concatenated, so the stratum sizes are fixed. With a
    single stratum this is an ordinary row-level bootstrap (identical draws).
    """
    strata = np.asarray(strata)
    parts = []
    for s in np.unique(strata):
        idx_s = np.flatnonzero(strata == s)
        parts.append(idx_s[rng.integers(0, len(idx_s), size=(n_boot, len(idx_s)))])
    return np.concatenate(parts, axis=1)


def _encoded_group(group: pd.DataFrame):
    group = group.sort_values(["ID", "trial"]).reset_index(drop=True)
    all_cats, cat_dtype, invalid_code = build_label_encoding(group)
    y_true = encode(group["ground_truth_text"], cat_dtype).to_numpy()
    y_pred = encode(group["prediction_text"], cat_dtype).to_numpy()
    return group, all_cats, invalid_code, y_true, y_pred


def _strata(group: pd.DataFrame, stratify_col: str | None):
    if stratify_col is None:
        return np.zeros(len(group), dtype=int)
    return group[stratify_col].astype(str).to_numpy()


_COLS = ["ID", "trial", "dataset", "dataset_raw", "benchmark", "model",
         "ground_truth_text", "prediction_text"]


# Per-class (COG, NP)

def _per_class_task(group_info, y_true_b, y_pred_b, lbl_code, m_type, point):
    boot_values = metric_calc(y_true_b, y_pred_b, lbl_code, m_type).astype(float)

    # Drop resamples where this class has no true rows (metric undefined there)
    present = np.sum(y_true_b == lbl_code, axis=-1) > 0
    boot_values = boot_values[present]

    low, med, high = np.percentile(boot_values, [2.5, 50, 97.5])
    return {
        **group_info,
        "class_code": lbl_code,
        "metric": m_type,
        "low": float(low),
        "high": float(high),
        "point": float(point),
        "n_boot_used": int(present.sum()),
    }


def bootstrap_per_class(
    df: pd.DataFrame,
    n_boot: int,
    seed: int = 42,
    n_jobs: int = -1,
    stratify_col: str | None = None,
) -> pd.DataFrame:
    """
    Bootstrap CIs for per-class precision/recall/F1 per (dataset, model).
    `stratify_col` (e.g. "dataset_raw") keeps pooled cohorts at their original sizes.
    """
    groups = list(df[_COLS].copy().groupby(["dataset", "model"], observed=True))
    main_rng = np.random.default_rng(seed)
    all_tasks = []
    int_to_label: dict[tuple[str, str], dict[int, str]] = {}

    print(f"Preparing bootstrap data for {len(groups)} groups...")

    for g_id, group in groups:
        group, all_cats, invalid_code, y_true, y_pred = _encoded_group(group)
        labels_to_process = [int(c) for c in gt_label_codes(y_true, invalid_code)]
        int_to_label[(g_id[0], g_id[1])] = dict(enumerate(all_cats))

        rng = np.random.default_rng(int(main_rng.integers(0, 2**32)))
        indices = stratified_indices(rng, n_boot, _strata(group, stratify_col))
        y_true_b, y_pred_b = y_true[indices], y_pred[indices]

        for lbl_code in labels_to_process:
            for m_type in ["precision", "recall", "f1"]:
                point = metric_calc(y_true, y_pred, lbl_code, m_type)
                all_tasks.append(({"dataset": g_id[0], "model": g_id[1]},
                                  y_true_b, y_pred_b, lbl_code, m_type, point))

    print(f"Executing bootstrap on {len(all_tasks)} tasks across {n_jobs} cores...")
    results = Parallel(n_jobs=n_jobs)(delayed(_per_class_task)(*t) for t in all_tasks)

    res_df = pd.DataFrame(results)
    res_df["class"] = res_df.apply(
        lambda row: int_to_label[(row["dataset"], row["model"])][row["class_code"]], axis=1
    )

    dropped = (
        res_df.loc[res_df["n_boot_used"] < n_boot, ["dataset", "model", "class", "n_boot_used"]]
        .drop_duplicates()
    )
    if len(dropped) > 0:
        print(f"WARNING: resamples dropped for {len(dropped)} (dataset, model, class) groups "
              f"because the class was absent:")
        print(dropped.to_string(index=False))
    else:
        print("Bootstrap: no resamples dropped.")

    return res_df.drop(columns=["class_code"])


# Macro (ETPR, biomarker)

def _macro_task(group_info, y_true_b, y_pred_b, label_codes, m_type, point):
    boot_values = macro_metric_calc(y_true_b, y_pred_b, label_codes, m_type)
    low, med, high = np.quantile(boot_values, [0.025, 0.5, 0.975])

    # Resamples where at least one class had no true rows
    counts = np.stack([np.sum(y_true_b == l, axis=-1) for l in label_codes], axis=-1)
    n_affected = int((counts == 0).any(axis=-1).sum())

    return {
        **group_info,
        "metric": f"macro_{m_type}",
        "point": float(point),
        "low": float(low),
        "high": float(high),
        "n_boot_affected": n_affected,
    }


def bootstrap_macro(
    df: pd.DataFrame,
    n_boot: int,
    seed: int = 42,
    n_jobs: int = -1,
    stratify_col: str | None = None,
) -> pd.DataFrame:
    """Bootstrap CIs for macro precision/recall/F1 per (dataset, model)."""
    groups = list(df[_COLS].copy().groupby(["dataset", "model"], observed=True))
    main_rng = np.random.default_rng(seed)
    all_tasks = []

    print(f"Preparing bootstrap data for {len(groups) * 3} macro tasks...")

    for g_id, group in groups:
        group, _, invalid_code, y_true, y_pred = _encoded_group(group)

        label_codes_group = gt_label_codes(y_true, invalid_code)
        if label_codes_group.size == 0:
            raise ValueError(f"[{g_id[0]}] {g_id[1]}: no valid ground-truth classes")

        rng = np.random.default_rng(int(main_rng.integers(0, 2**32)))
        indices = stratified_indices(rng, n_boot, _strata(group, stratify_col))
        y_true_b, y_pred_b = y_true[indices], y_pred[indices]

        group_info = {"dataset": g_id[0], "model": g_id[1]}
        for m_type in ["precision", "recall", "f1"]:
            point = macro_metric_calc(y_true, y_pred, label_codes_group, m_type)
            all_tasks.append((group_info, y_true_b, y_pred_b, label_codes_group, m_type, point))

    print(f"Executing bootstrap on {len(all_tasks)} tasks across {n_jobs} cores...")
    results = Parallel(n_jobs=n_jobs)(delayed(_macro_task)(*t) for t in all_tasks)
    res_df = pd.DataFrame(results)

    affected = (
        res_df.loc[res_df["n_boot_affected"] > 0, ["dataset", "model", "n_boot_affected"]]
        .drop_duplicates()
    )
    if len(affected) > 0:
        print(f"WARNING: some resamples were missing a class in {len(affected)} (dataset, model) "
              f"groups (averaged over remaining classes):")
        print(affected.to_string(index=False))
    else:
        print("Bootstrap: no resamples missing a class.")

    return res_df


# Macro accuracy over benchmarks (standard benchmarks)

def _macro_accuracy_task(group_info, correct_b, bench_b, benchmark_ids, point):
    boot_values = macro_accuracy(correct_b, bench_b, benchmark_ids)
    low, med, high = np.quantile(boot_values, [0.025, 0.5, 0.975])
    return {**group_info, "metric": "accuracy", "point": float(point),
            "low": float(low), "high": float(high)}


def bootstrap_macro_accuracy(df: pd.DataFrame, n_boot: int = N_BOOT, seed: int = 42,
                             n_jobs: int = -1) -> pd.DataFrame:
    """Per model: rows resampled within each benchmark, then accuracy macro-averaged."""
    groups = list(df.groupby("model", observed=True))
    main_rng = np.random.default_rng(seed)
    all_tasks = []

    print(f"Preparing bootstrap data for {len(groups)} model groups...")
    for model, group in groups:
        group = group.reset_index(drop=True)
        benchmark_codes = group["benchmark"].astype("category").cat.codes.to_numpy()
        benchmark_ids = np.unique(benchmark_codes)
        correct = group["correct"].to_numpy()

        point = macro_accuracy(correct, benchmark_codes, benchmark_ids)

        rng = np.random.default_rng(int(main_rng.integers(0, 2**32)))
        indices = stratified_indices(rng, n_boot, benchmark_codes)
        all_tasks.append(({"model": model}, correct[indices], benchmark_codes[indices],
                          benchmark_ids, point))

    print(f"Executing bootstrap on {len(all_tasks)} tasks across {n_jobs} cores...")
    return pd.DataFrame(Parallel(n_jobs=n_jobs)(delayed(_macro_accuracy_task)(*t) for t in all_tasks))


# Training curves: macro accuracy over (cohort, benchmark) cells

INTERNAL = "Internal validation\n(NACC)"
EXTERNAL = "External validation\n(All other cohorts)"
INTERNAL_COHORT = "nacc_test_updated"


def _macro_of_cell_means(correct_b: np.ndarray, block_starts: np.ndarray,
                         block_sizes: np.ndarray) -> np.ndarray:
    """
    correct_b: (k, n_rows), columns grouped by cell in contiguous blocks
    (as returned by stratified_indices). Macro average of cell means per resample.
    """
    cell_means = np.add.reduceat(correct_b, block_starts, axis=1) / block_sizes
    return cell_means.mean(axis=1)


def macro_ci_over_training_steps(
    df: pd.DataFrame,
    model_pattern: str = "NACC",
    n_boot: int = N_BOOT,
    seed: int = 0,
    chunk: int = 200,
) -> pd.DataFrame:
    """
    Macro-averaged accuracy per (training_steps, in_distribution).
    Cells = (cohort, benchmark); rows resampled within each cell; point estimate =
    macro-average of the original cell means. Resamples drawn in chunks of `chunk`
    (keep it fixed: the CIs depend on it through the random stream).
    """
    d = df.loc[df["model"].astype(str).str.contains(model_pattern, regex=True, na=False)].copy()
    if d.empty:
        return pd.DataFrame(columns=["training_steps", "in_distribution", "point", "low", "high"])

    d["in_distribution"] = np.where(d["cohort"] == INTERNAL_COHORT, INTERNAL, EXTERNAL)

    rng = np.random.default_rng(seed)
    out_rows = []

    steps_groups = (
        d[["training_steps", "in_distribution"]].drop_duplicates()
        .sort_values(["training_steps", "in_distribution"])
        .itertuples(index=False, name=None)
    )
    for step, grp in steps_groups:
        slice_df = d.loc[(d["training_steps"] == step) & (d["in_distribution"] == grp)]
        if slice_df.empty:
            continue

        cells = slice_df[["cohort", "benchmark"]].drop_duplicates().reset_index(drop=True)
        cells["cell_id"] = np.arange(len(cells), dtype=int)
        s2 = slice_df.merge(cells, on=["cohort", "benchmark"], how="inner")

        cell_id = s2["cell_id"].to_numpy()
        correct = s2["correct"].to_numpy(dtype=float)

        block_sizes = np.bincount(cell_id, minlength=len(cells))
        cell_means = np.bincount(cell_id, weights=correct, minlength=len(cells)) / block_sizes
        point = float(cell_means.mean())

        block_starts = np.concatenate([[0], np.cumsum(block_sizes)[:-1]])
        bs = np.empty(n_boot, dtype=float)
        for start in range(0, n_boot, chunk):
            k = min(chunk, n_boot - start)
            indices = stratified_indices(rng, k, cell_id)
            bs[start:start + k] = _macro_of_cell_means(correct[indices], block_starts, block_sizes)

        out_rows.append((int(step), str(grp), point,
                         float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))))

    return pd.DataFrame(out_rows, columns=["training_steps", "in_distribution", "point", "low", "high"])


def split_baseline(df: pd.DataFrame, model_name: str, n_boot: int = N_BOOT,
                   seed: int = 0) -> pd.DataFrame:
    """Baseline model's macro accuracy, internal and external, same method as the curves."""
    out = macro_ci_over_training_steps(df, model_pattern=rf"^{re.escape(model_name)}$",
                                       n_boot=n_boot, seed=seed)
    if out.empty:
        raise ValueError(f"{model_name}: no baseline rows found")
    if out["in_distribution"].duplicated().any():
        raise ValueError(f"{model_name}: baseline found at multiple training_steps "
                         f"{sorted(out['training_steps'].unique())}")
    return out


def baselines_per_benchmark(df: pd.DataFrame, baselines: dict[str, str],
                            n_boot: int = N_BOOT) -> dict[str, dict[str, pd.DataFrame]]:
    """{benchmark: {size: split baseline}} for each baseline model, e.g. {"3B": "Qwen2.5-3B-Instruct"}."""
    out: dict[str, dict[str, pd.DataFrame]] = {}
    for size, model_name in baselines.items():
        benches = df.loc[df["model"] == model_name, "benchmark"].dropna().unique().tolist()
        for bench in benches:
            out.setdefault(bench, {})[size] = split_baseline(
                df.loc[df["benchmark"] == bench], model_name, n_boot=n_boot)
    return out
