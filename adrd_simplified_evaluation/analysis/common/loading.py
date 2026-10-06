"""Data loading: clinical and standard benchmarks, training runs, token counts,
entropy summaries and W&B training logs."""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


def option_string_to_dict(options: str) -> dict[str, str]:
    """Parse 'A. ...\\nB. ...' into {letter: option text}."""
    pattern = r"([A-Z])\. ([^\n]+)"
    return {key: value for key, value in re.findall(pattern, options)}


def model_from_path(fpath: Path) -> str:
    return fpath.parent.name.split("-", 3)[-1]


def benchmark_from_path(fpath: Path) -> str:
    return fpath.parent.parent.name.split("_", 1)[-1].upper()


def load_answers(dir_path: Path, dataset_name: str, model_keys) -> pd.DataFrame:
    """
    Load all parquet files in a dataset/benchmark directory into a tall DataFrame,
    keeping only the models in `model_keys` (raw result-directory names).
    """
    fpaths = list(Path(dir_path).rglob("*.parquet"))
    cols_to_read = ["ID", "ground_truth", "prediction", "ground_truth_text", "options"]
    dfs: list[pd.DataFrame] = []

    for fpath in tqdm(fpaths, desc=f"Loading {dataset_name}"):
        df = pd.read_parquet(fpath, columns=cols_to_read)
        df = df.assign(model=model_from_path(fpath), benchmark=benchmark_from_path(fpath))
        df["correct"] = (df["ground_truth"] == df["prediction"]).astype(int)
        df["prediction_text"] = df.apply(
            lambda row: option_string_to_dict(row["options"]).get(row["prediction"], "invalid"),
            axis=1,
        )
        dfs.append(df)

    if not dfs:
        return pd.DataFrame(
            columns=cols_to_read + ["model", "benchmark", "correct", "prediction_text", "dataset"]
        )

    df_all = pd.concat(dfs, ignore_index=True)
    df_all["dataset"] = dataset_name

    for col in ["dataset", "benchmark", "model", "ground_truth_text", "prediction_text"]:
        df_all[col] = pd.Categorical(df_all[col])

    return df_all[df_all["model"].isin(list(model_keys))]


def print_sample_sizes(df: pd.DataFrame, title: str = "") -> None:
    """
    Unique participants and rows per cohort and model, from a build_results_df
    table. Pooled groups (e.g. "Other") get one line per cohort and a total.
    Participants are counted per cohort (cohort + ID), since IDs can repeat across cohorts.
    """
    d = df.assign(pid=df["dataset_raw"].astype(str) + "__" + df["ID"].astype(str))
 
    def _counts(group_cols):
        out = (d.groupby(group_cols, observed=True)
                .agg(participants=("pid", "nunique"), rows=("pid", "size")))
        out["rows_per_participant"] = (out["rows"] / out["participants"]).round(2)
        return out
 
    per_cohort = _counts(["dataset", "dataset_raw", "model"])
    print(f"\nSample sizes{f' ({title})' if title else ''}:")
    print(per_cohort.to_string())
 
    pooled = d.loc[d["dataset"] != d["dataset_raw"], "dataset"].unique()
    if len(pooled) > 0:
        print("\nPooled totals:")
        print(_counts(["dataset", "model"]).loc[list(pooled)].to_string())
    print()



def build_results_df(
    dataset_paths: dict[str, Path],
    model_keys,
    pool_as_other: list[str] | None = None,
) -> pd.DataFrame:
    """
    Load and concatenate several datasets.
    `dataset_raw` keeps the original cohort; `dataset` optionally pools cohorts
    in `pool_as_other` into "Other". `trial` is a running row index.
    """
    frames = [load_answers(path, dataset_name=name, model_keys=model_keys)
              for name, path in dataset_paths.items()]
    results_df = pd.concat(frames, ignore_index=True)

    results_df["dataset_raw"] = results_df["dataset"]
    if pool_as_other:
        results_df["dataset"] = results_df["dataset"].replace(
            {name: "Other" for name in pool_as_other}
        )
    results_df["trial"] = results_df.index
    return results_df


# Standard medical benchmarks

def load_benchmark_answers(dir_path, benchmark_names: dict[str, str],
                           model_map: dict[str, str], model_order) -> pd.DataFrame:
    """Standard benchmarks: one parquet per (benchmark, model); `correct` per row."""
    dfs = []
    for fpath in tqdm(list(Path(dir_path).rglob("*.parquet")), desc="Loading parquet files"):
        df = pd.read_parquet(fpath, columns=["ID", "question", "options", "ground_truth", "prediction"])
        df = df.assign(model=model_from_path(fpath), benchmark=fpath.parent.parent.name)
        df["correct"] = (df["ground_truth"] == df["prediction"]).astype(int)
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    df["benchmark"] = df["benchmark"].replace(benchmark_names)
    df["model"] = df["model"].replace(model_map)
    df = df[df["model"].isin(model_order)].reset_index(drop=True)

    for col in ["benchmark", "model", "prediction", "ground_truth"]:
        df[col] = pd.Categorical(df[col])
    return df


def print_id_coverage(df: pd.DataFrame) -> None:
    for benchmark in df["benchmark"].unique():
        print(f"\n=== {benchmark} ===")
        bench_data = df[df["benchmark"] == benchmark]
        for model in bench_data["model"].unique():
            n_ids = bench_data[bench_data["model"] == model]["ID"].nunique()
            print(f"{model}: {n_ids} unique IDs")


# Training curves

def load_training_runs(res_path, include_model: str = "SFT") -> pd.DataFrame:
    """
    Load all parquet files whose path contains `include_model`, with training
    step and run name from the config.yml next to each file. No benchmark
    filtering here; callers filter explicitly.
    """
    from yaml import safe_load

    cols = ["ID", "ground_truth", "prediction", "ground_truth_text", "options"]
    dfs, n_ignored, skipped = [], 0, set()

    for fpath in tqdm(Path(res_path).rglob("*.parquet"), desc="Loading parquet files"):
        if "eeg" in str(fpath):
            n_ignored += 1
            continue
        if include_model not in str(fpath):
            n_ignored += 1
            continue
        try:
            with (fpath.parent / "config.yml").open() as f:
                config = safe_load(f)
        except Exception:
            skipped.add(fpath)
            continue
        
        # print(fpath)
        # print(f"benchmark: {fpath.parent.parent.parent.name}")
        # print(f"cohort: {fpath.parent.parent.parent.parent.name}")
        # raise ValueError

        df = pd.read_parquet(fpath, columns=cols)
        df["benchmark"] = fpath.parent.parent.parent.name
        df["cohort"] = fpath.parent.parent.parent.parent.name
        df["correct"] = (df["ground_truth"] == df["prediction"]).astype("int8")
        df["training_steps"] = int(config["training_steps"])
        df["model"] = str(config["run_readable_name"])
        dfs.append(df)

    print(f"Ignored: {n_ignored} files")
    print(f"Skipped: {len(skipped)} files")
    if not dfs:
        return pd.DataFrame(columns=cols + ["benchmark", "cohort", "correct", "training_steps", "model"])

    merged = pd.concat(dfs, ignore_index=True)
    # pick one model out of merged['model']
    # If there is more than one, arbitrarily pick the first one (by sorting the names alphabetically)
    sub_df = merged[merged["model"] == sorted(merged["model"].unique())[0]]
    print(f"Unique benchmarks {include_model}: {dict(sub_df['benchmark'].value_counts())}")
    print(f"Unique cohort {include_model}: {dict(sub_df['cohort'].value_counts())}")
    print(f"Unique model {include_model}: {dict(merged['model'].value_counts())}")
    print(sub_df.head())
    return merged


# Output length (token counts)

_TOKENIZER = None


def get_tokenizer(name: str = "Qwen/Qwen2.5-3B-Instruct"):
    """Load the tokenizer only when token counts are computed."""
    global _TOKENIZER
    if _TOKENIZER is None:
        import os
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
        from transformers import AutoTokenizer
        _TOKENIZER = AutoTokenizer.from_pretrained(name)
    return _TOKENIZER


def count_tokens_fast(texts: list[str], batch_size: int = 1000) -> list[int]:
    tokenizer = get_tokenizer()
    counts: list[int] = []
    for i in range(0, len(texts), batch_size):
        encoded = tokenizer(texts[i: i + batch_size], add_special_tokens=False,
                            truncation=False, padding=False, return_attention_mask=False)
        counts.extend(len(ids) for ids in encoded["input_ids"])
    return counts


TOKEN_CSV_COLS = ["dataset", "benchmark", "model", "ID", "finish_reason", "correct", "token_count"]


def load_token_counts(dir_path, dataset_name: str, model_map: dict[str, str], model_order,
                      skip_benchmarks=()) -> pd.DataFrame:
    """Tokenize each file as it is read and drop the text right away (memory)."""
    cols_to_read = ["ID", "ground_truth", "prediction", "generated_text", "finish_reason"]
    dfs = []

    for fpath in tqdm(list(Path(dir_path).rglob("*.parquet")), desc=f"Loading {dataset_name}"):
        model = model_map.get(model_from_path(fpath), model_from_path(fpath))
        benchmark = benchmark_from_path(fpath)
        # print(str(fpath), model, benchmark)
        # raise ValueError
        if benchmark in skip_benchmarks or model not in model_order:
            print(f"Skipped path {str(fpath)}")
            continue

        df = pd.read_parquet(fpath, columns=cols_to_read)
        df["token_count"] = count_tokens_fast(df["generated_text"].fillna("").tolist())
        df = df.drop(columns=["generated_text"])

        df = df.assign(model=model, benchmark=benchmark, dataset=dataset_name)
        df["correct"] = (df["ground_truth"] == df["prediction"]).astype(int)
        dfs.append(df)

    if not dfs:
        return pd.DataFrame(columns=TOKEN_CSV_COLS)
    return pd.concat(dfs, ignore_index=True)[TOKEN_CSV_COLS]


# Entropy

def load_entropy_summaries(data_dir, key_names: dict[str, str], key_order) -> pd.DataFrame:
    """Per-question mean entropies keyed by question index (prefers *_summary.json)."""
    import json
    import os

    files = sorted(f for f in os.listdir(data_dir) if f.endswith(".json"))
    has_summary = {f[: -len("_summary.json")] for f in files if f.endswith("_summary.json")}

    rows = []
    for fname in files:
        if fname.endswith("_summary.json"):
            key = fname[: -len("_summary.json")]
        else:
            key = fname[: -len(".json")]
            if key in has_summary:
                continue

        with open(os.path.join(data_dir, fname)) as f:
            values = json.load(f)
        if "idx" not in values:
            raise ValueError(f"{fname} has no 'idx'; regenerate it with the updated script")

        method = key_names.get(key, key)
        rows += [{"method": method, "q_idx": i, "mean_entropy": m}
                 for i, m in zip(values["idx"], values["mean"])]

    df = pd.DataFrame(rows)
    missing = set(key_order) - set(df["method"])
    if missing:
        raise ValueError(f"No entropy data for: {missing}")

    print("Questions per model:\n", df.groupby("method")["q_idx"].nunique().to_string())
    print(f"Max mean entropy: {df['mean_entropy'].max():.3f}")
    return df


# Weights & Biases training logs (training-dynamics figure)

def load_wandb_metric(path, metric_key: str, run_names: dict[str, str],
                      merge_runs: dict[str, str] | None = None,
                      step_col: str = "train/global_step") -> pd.DataFrame:
    """
    Load a W&B export CSV: keep the step column and the `metric_key` columns
    (not __MIN/__MAX), name each column by its run, fill gaps in a run from the
    run it continues (`merge_runs` = {continued run: original run}), then rename
    runs to display names (`run_names`).
    """
    df = pd.read_csv(path)
    df[step_col] = pd.to_numeric(df[step_col], errors="coerce")

    metric_cols = [c for c in df.columns
                   if metric_key in c and "__MIN" not in c and "__MAX" not in c]
    if not metric_cols:
        raise ValueError(f"No columns found for metric '{metric_key}' in {path}")

    df = df[[step_col] + metric_cols]
    df = df.rename(columns={c: c.split("-")[0].strip() for c in metric_cols})

    for cont_col, base_col in (merge_runs or {}).items():
        df[cont_col] = df[cont_col].combine_first(df[base_col])
        df = df.drop(columns=[base_col])

    return df.rename(columns=run_names)


def wandb_to_long(df: pd.DataFrame, value_name: str, order,
                  step_col: str = "train/global_step") -> pd.DataFrame:
    return df.melt(id_vars=step_col, value_vars=list(order), var_name="Variant",
                   value_name=value_name)


def ema_smooth_long(df_long: pd.DataFrame, value_col: str, alpha: float,
                    step_col: str = "train/global_step") -> pd.DataFrame:
    """
    Exponential moving average per variant, starting at the first logged value.
    Steps without a value stay missing (so each curve stops where its run ended).
    """
    df_long = df_long.sort_values(["Variant", step_col]).copy()

    def ema_one_variant(g: pd.DataFrame) -> pd.Series:
        values = g[value_col].to_numpy()
        out = np.full(len(values), np.nan, dtype=float)
        if not np.any(~np.isnan(values)):
            return pd.Series(out, index=g.index)

        start = np.argmax(~np.isnan(values))
        ema = values[start]
        out[start] = ema
        for i in range(start + 1, len(values)):
            if np.isnan(values[i]):
                continue
            ema = (1.0 - alpha) * ema + alpha * values[i]
            out[i] = ema
        return pd.Series(out, index=g.index)

    df_long[value_col] = df_long.groupby("Variant", group_keys=False).apply(ema_one_variant)
    return df_long
