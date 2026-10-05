"""Standard medical benchmarks: accuracy macro-averaged over benchmarks."""

from common.bootstrap import bootstrap_macro_accuracy
from common.config import BASE_RESULTS, N_BOOT, N_PERM, model_subset, setup_style
from common.loading import load_benchmark_answers, print_id_coverage
from common.permutation import pairwise_macro_accuracy
from common.plots import plot_accuracy_bars
from common.reporting import letter_pmatrix
from pathlib import Path
import pandas as pd

setup_style(hatch_linewidth=0.5)

# Configuration
BENCH_PATH = BASE_RESULTS / "standard_benchmarks"
BENCHMARK_NAMES = {
    "medmcqa": "MedMCQA",
    "medqa_test": "MedQA",
    "clinical_knowledge": "MMLU - clinical knowledge",
    "professional_medicine": "MMLU - professional medicine",
    "anatomy": "MMLU - anatomy",
    "medexpqa": "MedExpQA",
}
EXCLUDED_BENCHMARKS = ["MMLU - professional medicine"]

MODEL_MAP = model_subset(
    "Qwen2.5-3B-Instruct", "NACC-3B-OS-SFT", "NACC-3B", "NACC-3B-OS", "NACC-3B-SCE",
    "NACC-3B-OS-SCE", "Qwen2.5-7B-Instruct",
)
MODEL_ORDER = list(MODEL_MAP.values())  # Q3B, SFT, LUNAR-OS-SCe, LUNAR-SCe, LUNAR-OS, LUNAR, Q7B
COLOR_ORDER = ["Q3B", "LUNAR-OS-SCe", "LUNAR-SCe", "LUNAR-OS", "LUNAR", "Q7B", "random", "SFT"]

OUT_PATH = "../figures_main/fig2_benchmarks_macro_perm_id.pdf"

LOAD_FROM_ORIGINAL = False
INTERMEDIATE_FILE_PATH = "intermediate_files/benchmark_answers_original.parquet"

def main() -> None:
    if LOAD_FROM_ORIGINAL or not Path(INTERMEDIATE_FILE_PATH).exists():
        print("Loading benchmark answers from original dataset...")
        ans = load_benchmark_answers(BENCH_PATH, BENCHMARK_NAMES, MODEL_MAP, MODEL_ORDER)
        ans = ans[~ans["benchmark"].isin(EXCLUDED_BENCHMARKS)].reset_index(drop=True)
        ans["ID"] = ans["ID"].astype(str)
        ans.to_parquet(INTERMEDIATE_FILE_PATH)
    else:
        print("Loading saved benchmark answers...")
        ans = pd.read_parquet(INTERMEDIATE_FILE_PATH)
    
    print(ans.head())
    print(ans.columns)
    print_id_coverage(ans)

    print("Computing macro-averaged bootstrap accuracy CIs...")
    results = bootstrap_macro_accuracy(ans, n_boot=N_BOOT, seed=42, n_jobs=-1)

    print("Computing permutation tests (macro-averaged)...")
    perm_results = pairwise_macro_accuracy(ans, n_permutations=N_PERM, seed=42, n_jobs=-1)

    print("Plotting figure...")
    model_to_letter = {m: chr(65 + i) for i, m in enumerate(MODEL_ORDER)}
    fig, _ = plot_accuracy_bars(results, letter_pmatrix(perm_results, MODEL_ORDER, model_to_letter),
                                MODEL_ORDER, COLOR_ORDER)
    fig.savefig(OUT_PATH, bbox_inches="tight", dpi=300)
    print(f"Saved figure to {OUT_PATH}")


if __name__ == "__main__":
    main()
