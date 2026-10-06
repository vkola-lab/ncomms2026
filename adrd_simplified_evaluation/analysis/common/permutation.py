"""
Paired ID-level swap permutation tests.

Under the null the two models are exchangeable, so for every ID (question /
patient) their values are swapped with probability 1/2. All rows of an ID are
swapped together. p = (#|null diff| >= |observed diff| + 1) / (n_perms + 1).
"""

from functools import partial
from itertools import combinations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from statsmodels.stats.multitest import multipletests

from .config import PERM_CHUNK
from .metrics import (
    build_label_encoding, encode, gt_label_codes, per_class_stat, macro_stat,
)


def swap_permutation_test(stat_fn, v1, v2, id_array, n_perms: int, seed: int,
                          chunk: int = PERM_CHUNK) -> dict:
    """
    stat_fn(values) -> statistic; must accept (n,) and (k, n) arrays.
    v1, v2: aligned per-row values of the two models (predictions, correctness, ...).
    """
    rng = np.random.default_rng(seed)

    obs1 = stat_fn(v1)
    obs2 = stat_fn(v2)
    obs_diff = obs1 - obs2

    unique_ids, id_indices = np.unique(id_array, return_inverse=True)
    n_ids = len(unique_ids)

    extreme = 0
    for start in range(0, n_perms, chunk):
        k = min(chunk, n_perms - start)
        swap_ids = rng.integers(0, 2, size=(k, n_ids), dtype=bool)
        swap = swap_ids[:, id_indices]  # (k, n_rows)

        p1 = np.where(swap, v2, v1)
        p2 = np.where(swap, v1, v2)

        null_diff = stat_fn(p1) - stat_fn(p2)
        extreme += np.sum(np.abs(null_diff) >= np.abs(obs_diff) - 1e-12)

    return {
        "p_value": float((extreme + 1) / (n_perms + 1)),
        "observed_diff": float(obs_diff),
        "obs1": float(obs1),
        "obs2": float(obs2),
    }


def check_aligned(d1: pd.DataFrame, d2: pd.DataFrame, m1: str, m2: str,
                  id_col: str = "perm_id", same_cols=(), prefix: str = "") -> None:
    """Raise if two sorted per-model frames don't cover the same IDs row by row."""
    ids1, ids2 = set(d1[id_col].unique()), set(d2[id_col].unique())
    if ids1 != ids2:
        raise ValueError(
            f"{prefix}{m1} vs {m2}: ID sets differ "
            f"(only in {m1}: {len(ids1 - ids2)}, only in {m2}: {len(ids2 - ids1)})"
        )
    if len(d1) != len(d2) or not np.array_equal(d1[id_col].to_numpy(), d2[id_col].to_numpy()):
        raise ValueError(f"{prefix}{m1} vs {m2} are not aligned")
    for col in same_cols:
        if not np.array_equal(d1[col].to_numpy(), d2[col].to_numpy()):
            raise ValueError(f"{prefix}{m1} vs {m2}: {col} differs for matched IDs")


def add_bh(res_df: pd.DataFrame, group_cols: list[str] | None = None) -> pd.DataFrame:
    """Benjamini-Hochberg within each group (or across all rows)."""
    res_df = res_df.copy()
    if len(res_df) == 0:
        return res_df
    if not group_cols:
        _, res_df["p_value_bh"], _, _ = multipletests(res_df["p_value"], method="fdr_bh")
    else:
        res_df["p_value_bh"] = np.nan
        for _, sub_idx in res_df.groupby(group_cols).groups.items():
            _, p_bh, _, _ = multipletests(res_df.loc[sub_idx, "p_value"].to_numpy(),
                                          method="fdr_bh")
            res_df.loc[sub_idx, "p_value_bh"] = p_bh
    res_df["Significant_bh"] = res_df["p_value_bh"] < 0.05
    return res_df


def run_tasks(tasks: list[dict], n_perms: int, seed: int, n_jobs: int,
              chunk: int = PERM_CHUNK) -> pd.DataFrame:
    """
    Each task: {"stat_fn", "v1", "v2", "id_array", **labels}. Labels are kept
    in the output; arrays and the function are dropped. One seed per task.
    """
    seeds = np.random.default_rng(seed).integers(0, 2**32, size=len(tasks))
    results = Parallel(n_jobs=n_jobs)(
        delayed(_worker)(tasks[i], n_perms, int(seeds[i]), chunk) for i in range(len(tasks))
    )
    return pd.DataFrame(results)


def _worker(task: dict, n_perms: int, seed: int, chunk: int) -> dict:
    out = swap_permutation_test(task["stat_fn"], task["v1"], task["v2"],
                                task["id_array"], n_perms, seed, chunk)
    labels = {k: v for k, v in task.items() if k not in ("stat_fn", "v1", "v2", "id_array")}
    return labels | out


# Clinical benchmarks (COG, ETPR, NP, biomarker)

def _encoded_pairs(df: pd.DataFrame):
    """
    Yield (ds, int_to_label, invalid_code, m1, m2, yt, yp1, yp2, id_array) for every
    model pair within each dataset group, after strict alignment checks.
    """
    df_grouped = df[["ID", "trial", "dataset", "dataset_raw", "benchmark", "model",
                     "ground_truth_text", "prediction_text"]].copy()
    df_grouped["perm_id"] = df_grouped["dataset_raw"].astype(str) + "__" + df_grouped["ID"].astype(str)

    for ds, group in df_grouped.groupby("dataset", observed=True):
        all_cats, cat_dtype, invalid_code = build_label_encoding(group)
        group_int = pd.DataFrame({
            "perm_id": group["perm_id"],
            "trial": group["trial"],
            "model": group["model"],
            "y_true": encode(group["ground_truth_text"], cat_dtype),
            "y_pred": encode(group["prediction_text"], cat_dtype),
        })

        models = sorted(group_int["model"].unique())
        for m1, m2 in combinations(models, 2):
            d1 = group_int[group_int["model"] == m1].sort_values(["perm_id", "trial"]).reset_index(drop=True)
            d2 = group_int[group_int["model"] == m2].sort_values(["perm_id", "trial"]).reset_index(drop=True)
            check_aligned(d1, d2, m1, m2, same_cols=("y_true",), prefix=f"[{ds}] ")

            yield (ds, dict(enumerate(all_cats)), invalid_code, m1, m2,
                   d1["y_true"].to_numpy(), d1["y_pred"].to_numpy(),
                   d2["y_pred"].to_numpy(), d1["perm_id"].to_numpy())


def pairwise_per_class(df: pd.DataFrame, n_permutations: int, seed: int = 42,
                       n_jobs: int = -1) -> pd.DataFrame:
    """Per-class precision/recall/F1 tests; BH within (dataset, class, metric)."""
    tasks = []
    for ds, int_to_label, invalid_code, m1, m2, yt, yp1, yp2, id_array in _encoded_pairs(df):
        for lbl_code in [int(c) for c in gt_label_codes(yt, invalid_code)]:
            for m_type in ["precision", "recall", "f1"]:
                tasks.append({
                    "dataset": ds, "model1": m1, "model2": m2,
                    "class": int_to_label[lbl_code], "metric": m_type,
                    "stat_fn": partial(per_class_stat, y_true=yt, label_code=lbl_code, metric=m_type),
                    "v1": yp1, "v2": yp2, "id_array": id_array,
                })

    print(f"Executing permutation tests on {len(tasks)} tasks...")
    res_df = run_tasks(tasks, n_permutations, seed, n_jobs)
    return add_bh(res_df, ["dataset", "class", "metric"])


def pairwise_macro(df: pd.DataFrame, n_permutations: int, seed: int = 42,
                   n_jobs: int = -1) -> pd.DataFrame:
    """Macro precision/recall/F1 tests; BH within (dataset, metric)."""
    tasks = []
    for ds, _, invalid_code, m1, m2, yt, yp1, yp2, id_array in _encoded_pairs(df):
        label_codes = gt_label_codes(yt, invalid_code)
        if label_codes.size == 0:
            raise ValueError(f"[{ds}] {m1} vs {m2}: no valid ground-truth classes")
        for m_type in ["precision", "recall", "f1"]:
            tasks.append({
                "dataset": ds, "model1": m1, "model2": m2, "metric": f"macro_{m_type}",
                "stat_fn": partial(macro_stat, y_true=yt, label_codes=label_codes, metric=m_type),
                "v1": yp1, "v2": yp2, "id_array": id_array,
            })

    print(f"Executing permutation tests on {len(tasks)} macro tasks for {n_permutations} permuations...")
    res_df = run_tasks(tasks, n_permutations, seed, n_jobs)
    return add_bh(res_df, ["dataset", "metric"])


# Standard benchmarks: accuracy macro-averaged over benchmarks

def pairwise_macro_accuracy(df: pd.DataFrame, n_permutations: int, seed: int = 42,
                            n_jobs: int = -1) -> pd.DataFrame:
    """ID = benchmark__ID; BH across all model pairs."""
    from .metrics import macro_accuracy

    d = df[["ID", "model", "benchmark", "ground_truth", "correct"]].copy()
    d["perm_id"] = d["benchmark"].astype(str) + "__" + d["ID"].astype(str)

    bench_cat = pd.CategoricalDtype(categories=sorted(d["benchmark"].unique()))
    d["bench_code"] = d["benchmark"].astype(bench_cat).cat.codes
    benchmark_ids = np.unique(d["bench_code"].to_numpy())

    tasks = []
    for m1, m2 in combinations(sorted(d["model"].unique()), 2):
        d1 = d[d["model"] == m1].sort_values(["perm_id", "benchmark"]).reset_index(drop=True)
        d2 = d[d["model"] == m2].sort_values(["perm_id", "benchmark"]).reset_index(drop=True)
        check_aligned(d1, d2, m1, m2, same_cols=("ground_truth",))

        tasks.append({
            "model1": m1, "model2": m2, "metric": "accuracy",
            "stat_fn": partial(macro_accuracy, benchmark_codes=d1["bench_code"].to_numpy(),
                               benchmark_ids=benchmark_ids),
            "v1": d1["correct"].to_numpy(), "v2": d2["correct"].to_numpy(),
            "id_array": d1["perm_id"].to_numpy(),
        })

    print(f"Executing permutation tests on {len(tasks)} model pairs...")
    return add_bh(run_tasks(tasks, n_permutations, seed, n_jobs))


# Continuous value per item (output length, entropy)

def pairwise_mean_tests(df: pd.DataFrame, id_col: str, model_col: str, value_col: str,
                        model_order, n_perms: int, seed: int = 42,
                        chunk: int = PERM_CHUNK, n_jobs: int = 1) -> pd.DataFrame:
    """
    Paired test of the difference in mean `value_col` for every pair in `model_order`.
    Repeats of an item are averaged first (one value per item per model), then
    items are swapped between the two models. BH across all pairs.
    """
    from .metrics import mean_stat

    per_item = df.groupby([id_col, model_col], observed=True)[value_col].mean().reset_index()

    tasks = []
    for i, m1 in enumerate(model_order):
        for m2 in model_order[i + 1:]:
            d1 = per_item[per_item[model_col] == m1].sort_values(id_col).reset_index(drop=True)
            d2 = per_item[per_item[model_col] == m2].sort_values(id_col).reset_index(drop=True)
            check_aligned(d1, d2, m1, m2, id_col=id_col)
            tasks.append({
                "model1": m1, "model2": m2, "n_paired": len(d1),
                "stat_fn": mean_stat,
                "v1": d1[value_col].to_numpy(), "v2": d2[value_col].to_numpy(),
                "id_array": d1[id_col].to_numpy(),
            })

    return add_bh(run_tasks(tasks, n_perms, seed, n_jobs, chunk=chunk))
