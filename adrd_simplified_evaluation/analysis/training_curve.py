"""Training curves: macro-averaged accuracy over training steps, with Q3B/Q7B baselines."""

import warnings
from pathlib import Path
import pandas as pd
from common.bootstrap import baselines_per_benchmark, macro_ci_over_training_steps, split_baseline
from common.config import BASE_RESULTS, N_BOOT, setup_style
from common.loading import load_training_runs
from common.plots_training import make_ablation_figure, make_main_figure, make_per_task_figure

warnings.filterwarnings("ignore")
setup_style(hatch_linewidth=0.5)

# Configuration

RES_PATH = BASE_RESULTS / "training_curve"

EXCLUDED_BENCHMARKS = ["test_mci", "test_np_mixed", "test_np", "test_ftld"]
EXCLUDED_BENCHMARKS_PER_TASK = ["test_mci", "test_ftld", "test_np"]  # keeps test_np_mixed

BASELINES = {"3B": "Qwen2.5-3B-Instruct", "7B": "Qwen2.5-7B-Instruct"}

# Path fragment identifying each run -> panel title in the ablation figure
ABLATION_RUNS = {
    "/NACC-3B/": "LUNAR-OS-SCe",
    "/NACC-3B-SCE/": "LUNAR-OS",
    "/NACC-3B-OS/": "LUNAR-SCe",
}

PER_TASK_BENCHMARKS = {
    "test_cog": "COG",
    "test_etpr": "ETPR",
    "test_pet": "PET",
    "test_csf": "CSF",
    "test_dat": "DAT",
    "test_np_one": "NP_ONE",
    "test_np_mixed": "NP_MIXED",
}

OUT_MAIN = "../figures_main/fig2_test_perf_over_training_trial_bootstrap1.pdf"
OUT_ABLATION = "../figures_main/sup_test_perf_over_training_trial_bootstrap_ablations1.pdf"
OUT_PER_TASK = "../figures_main/sup_test_perf_over_training_trial_bootstrap_sep_benchmarks1.pdf"


def load_filtered(include_model: str, file_path: str, excluded=EXCLUDED_BENCHMARKS):
    if not Path(file_path).exists():
        print("Loading original data files...")
        df = load_training_runs(RES_PATH, include_model=include_model)
        df.to_parquet(file_path)
    else:
        print(f"Loading from {file_path}...")
        df = pd.read_parquet(file_path)
    return df, df.loc[~df["benchmark"].isin(excluded)].copy()

def save(fig, path):
    fig.savefig(path, dpi=200, format="pdf", bbox_inches="tight")

def main() -> None:
    # SFT and LUNAR (SCE) curves
    print("Loading SFT data...")
    _, df_sft = load_filtered("NACC-3B-OS-SFT-ES", file_path="intermediate_files/training_curve_NACC-3B-OS-SFT-ES.parquet")
    print("Loading SCE data...")
    df_sce_all, df_sce = load_filtered("NACC-3B-OS-SCE", file_path="intermediate_files/training_curve_NACC-3B-OS-SCE.parquet")

    # Baselines, split internal/external (same method as the curves)
    base_sft = {size: split_baseline(df_sft, name, n_boot=N_BOOT) for size, name in BASELINES.items()}
    base_sce = {size: split_baseline(df_sce, name, n_boot=N_BOOT) for size, name in BASELINES.items()}
    for label, bases in [("SFT", base_sft), ("SCE", base_sce)]:
        for size, b in bases.items():
            print(f"\nQ{size} {label} baseline:")
            print(b[["in_distribution", "point", "low", "high"]].to_string(index=False))

    ci_sft = macro_ci_over_training_steps(df_sft, n_boot=N_BOOT, seed=0)
    ci_sce = macro_ci_over_training_steps(df_sce, n_boot=N_BOOT, seed=0)
    save(make_main_figure(ci_sft, ci_sce, base_sft, base_sce), OUT_MAIN)

    # Ablations (+ full LUNAR as the last panel)
    print("Loading ablation data...")
    ci_ablation = [macro_ci_over_training_steps(load_filtered(run, file_path=f"intermediate_files/training_curve_{run.replace('/', '')}.parquet")[1], n_boot=N_BOOT, seed=0)
                   for run in ABLATION_RUNS]
    save(make_ablation_figure(ci_ablation + [ci_sce], base_sce,
                              list(ABLATION_RUNS.values()) + ["LUNAR"]), OUT_ABLATION)

    # Per-task curves (SCE only)
    df_tasks = df_sce_all.loc[~df_sce_all["benchmark"].isin(EXCLUDED_BENCHMARKS_PER_TASK)].copy()
    base_by_bench = baselines_per_benchmark(df_tasks, BASELINES, n_boot=N_BOOT)
    ci_by_bench = {
        b: macro_ci_over_training_steps(df_tasks.loc[df_tasks["benchmark"] == b],
                                        model_pattern="NACC", n_boot=N_BOOT, seed=0)
        for b in PER_TASK_BENCHMARKS
    }
    save(make_per_task_figure(ci_by_bench, base_by_bench, PER_TASK_BENCHMARKS), OUT_PER_TASK)


if __name__ == "__main__":
    main()
