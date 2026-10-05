"""COG: per-class precision / recall / F1, NACC vs all other cohorts pooled."""

import pandas as pd
from pathlib import Path
from common.bootstrap import bootstrap_per_class
from common.config import BASE_RESULTS, FONTSIZE, N_BOOT, N_PERM, model_subset, setup_style
from common.loading import build_results_df, print_sample_sizes
from common.permutation import pairwise_per_class
from common.plots import plot_classwise_bars
from common.reporting import latex_table_per_class, save_text

setup_style()

# Configuration

MODEL_MAP = model_subset(
    "Qwen2.5-3B-Instruct",
    # "NACC-3B",
    # "NACC-3B-SCE",
    # "NACC-3B-OS",
    "NACC-3B-OS-SCE",
    "Qwen2.5-7B-Instruct",
)

CLASS_MAP = {
    "Normal Cognition (NC)": "NC",
    "Mild Cognitive Impairment (MCI)": "MCI",
    "Dementia (DE)": "DE",
}
CLASS_ORDER = ["NC", "MCI", "DE"]
DATASET_ORDER = ["NACC", "Other"]

# Order matters for the row index (`trial`); keep as is for reproducibility
DATASET_PATHS = {
    "NACC": BASE_RESULTS / "NACC/test_cog",
    "ADNI": BASE_RESULTS / "ADNI/test_cog",
    "BrainLat": BASE_RESULTS / "brainlat/test_cog",
    "NIFD": BASE_RESULTS / "NIFD/test_cog",
    "PPMI": BASE_RESULTS / "PPMI/test_cog",
}
POOL_AS_OTHER = ["ADNI", "BrainLat", "NIFD", "PPMI"]
DATASET_TITLES = {"NACC": "NACC (Internal testing)", "Other": "All other cohorts (External testing)"}

FIGSIZE_PR = (4.5, 4)
FIGSIZE_F1 = (4.0, 2.5)

N_JOBS_BOOT = 1
N_JOBS_PERM = 10

OUTPUT_FIG_DIR = "../figures_main"
OUTPUT_FIGNAME_PR = "fig3_cog_stat_id_level"
OUTPUT_FIGNAME_F1 = "fig3_cog_stat_id_level_f1"

LOAD_FROM_ORIGINAL = False
INTERMEDIATE_FILE_PATH = "intermediate_files/cog_results_original.parquet"
SAVE_LATEX = False
LATEX_OUTPUT_PATH = "../figures_main/tables/fig3_cog_table.tex"
TABLE_ONLY = False


# Main

def main() -> None:
    if LOAD_FROM_ORIGINAL or not Path(INTERMEDIATE_FILE_PATH).exists():
        print("Loading COG results from original dataset...")
        results_df = build_results_df(DATASET_PATHS, MODEL_MAP.keys(), pool_as_other=POOL_AS_OTHER)
        results_df.to_parquet(INTERMEDIATE_FILE_PATH)
    else:
        print("Loading saved results...")
        results_df = pd.read_parquet(INTERMEDIATE_FILE_PATH)
    
    print(results_df.head())
    print(results_df.columns)
    print_sample_sizes(results_df, title="COG")
    
    print("Running bootstrap...")
    all_metrics = bootstrap_per_class(results_df, n_boot=N_BOOT, seed=42, n_jobs=N_JOBS_BOOT,
                                      stratify_col="dataset_raw")

    if SAVE_LATEX:
        save_text(latex_table_per_class(all_metrics, MODEL_MAP, CLASS_MAP, CLASS_ORDER),
                  LATEX_OUTPUT_PATH)

    if TABLE_ONLY:
        return

    print("Running permutation tests...")
    pairwise = pairwise_per_class(results_df, n_permutations=N_PERM, seed=42, n_jobs=N_JOBS_PERM)

    plot_args = dict(
        all_metrics=all_metrics, pairwise_pvalues=pairwise, model_map=MODEL_MAP,
        class_map=CLASS_MAP, class_order=CLASS_ORDER, dataset_order=DATASET_ORDER,
        output_dir=OUTPUT_FIG_DIR, fontsize=FONTSIZE, dataset_titles=DATASET_TITLES,
    )
    print("Plotting F1...")
    plot_classwise_bars(**plot_args, metrics=["f1"], figname=OUTPUT_FIGNAME_F1, figsize=FIGSIZE_F1)
    # print("Plotting precision and recall...")
    # plot_classwise_bars(**plot_args, metrics=["precision", "recall"],
    #                     figname=OUTPUT_FIGNAME_PR, figsize=FIGSIZE_PR)


if __name__ == "__main__":
    main()
