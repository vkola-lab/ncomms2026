"""NP-one (NACC only): per-class precision / recall / F1."""

from common.bootstrap import bootstrap_per_class
from common.config import BASE_RESULTS, FONTSIZE, N_BOOT, N_PERM, model_subset, setup_style
from common.loading import build_results_df, print_sample_sizes
from common.permutation import pairwise_per_class
from common.plots import plot_classwise_points
from common.reporting import latex_table_per_class, save_text
from pathlib import Path
import pandas as pd
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
    "Alzheimer's disease pathology (AD)": "NP-AD",
    "Frontotemporal Lobar Degeneration with tau pathology or TDP-43 pathology (FTLD)": "NP-FTLD",
    "Lewy body pathology (LBD)": "NP-LBD",
    "No listed option is correct": "None",
}
CLASS_ORDER = ["None", "NP-AD", "NP-LBD", "NP-FTLD"]
DATASET_ORDER = ["NACC"]

DATASET_PATHS = {"NACC": BASE_RESULTS / "NACC/test_np_one"}

FIGSIZE_PR = (4.0, 4.0)
FIGSIZE_F1 = (3.0, 2.3)

N_JOBS_BOOT = 20
N_JOBS_PERM = 20

OUTPUT_FIG_DIR = "../figures_main"
OUTPUT_FIGNAME_PR = "fig3_np_one_prec_rec"
OUTPUT_FIGNAME_F1 = "fig3_np_one_f1"

LOAD_FROM_ORIGINAL = False
SAVE_LATEX = False
LATEX_OUTPUT_PATH = "../figures_main/tables/fig3_np_one_table.tex"
TABLE_ONLY = False
INTERMEDIATE_FILE_PATH = "intermediate_files/np_one_results_original.parquet"


def main() -> None:
    if LOAD_FROM_ORIGINAL or not Path(INTERMEDIATE_FILE_PATH).exists():
        print("Loading NP-one results from original dataset...")
        results_df = build_results_df(DATASET_PATHS, MODEL_MAP.keys())
        results_df.to_parquet(INTERMEDIATE_FILE_PATH)
    else:
        print("Loading saved results...")
        results_df = pd.read_parquet(INTERMEDIATE_FILE_PATH)
        
    print(results_df.head())
    print(results_df.columns)
    print_sample_sizes(results_df, title="NP-one")
    
    print("Running bootstrap...")
    all_metrics = bootstrap_per_class(results_df, n_boot=N_BOOT, seed=42, n_jobs=N_JOBS_BOOT)

    if SAVE_LATEX:
        save_text(latex_table_per_class(all_metrics, MODEL_MAP, CLASS_MAP, CLASS_ORDER,
                                        by_dataset=False),
                  LATEX_OUTPUT_PATH)

    if TABLE_ONLY:
        return

    print("Running permutation tests...")
    pairwise = pairwise_per_class(results_df, n_permutations=N_PERM, seed=42, n_jobs=N_JOBS_PERM)

    common_args = dict(
        all_metrics=all_metrics, pairwise_pvalues=pairwise, model_map=MODEL_MAP,
        class_map=CLASS_MAP, class_order=CLASS_ORDER, dataset_order=DATASET_ORDER,
        output_dir=OUTPUT_FIG_DIR, show_all_comparisons=True, fontsize=FONTSIZE,
    )
    print("Plotting F1...")
    plot_classwise_points(**common_args, metrics=["f1"], figname=OUTPUT_FIGNAME_F1,
                          figsize=FIGSIZE_F1)
    print("Plotting precision and recall...")
    plot_classwise_points(**common_args, metrics=["precision", "recall"],
                          figname=OUTPUT_FIGNAME_PR, figsize=FIGSIZE_PR)


if __name__ == "__main__":
    main()
