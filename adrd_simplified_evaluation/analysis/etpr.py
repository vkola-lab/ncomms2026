"""ETPR: macro precision / recall / F1 per cohort, circular bar plot."""

from common.bootstrap import bootstrap_macro
from common.config import BASE_RESULTS, FONTSIZE, N_BOOT, N_PERM, model_subset, setup_style
from common.loading import build_results_df, print_sample_sizes
from common.permutation import pairwise_macro
from common.plots import plot_macro_circular
from common.reporting import latex_table_macro, save_text
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
MODEL_ORDER = tuple(MODEL_MAP.values())

# Order matters for the row index (`trial`); keep as is for reproducibility
DATASET_PATHS = {
    "ADNI": BASE_RESULTS / "ADNI/test_etpr",
    "BrainLat": BASE_RESULTS / "brainlat/test_etpr",
    "NIFD": BASE_RESULTS / "NIFD/test_etpr",
    "NACC": BASE_RESULTS / "NACC/test_etpr",
    "PPMI": BASE_RESULTS / "PPMI/test_etpr",
}
DATASET_ORDER = ["NACC", "NIFD", "PPMI", "ADNI", "BrainLat"]

FIGSIZE_PR = (6, 3.2)
FIGSIZE_F1 = (3.2, 3.2)

N_JOBS = 40

OUTPUT_FIG_DIR = "../figures_main"
OUTPUT_FIGNAME_PR = "fig3_macro_circular_bar_plot_sidebyside_id_level"
OUTPUT_FIGNAME_F1 = "fig3_macro_circular_bar_plot_sidebyside_id_level_f1"

LOAD_FROM_ORIGINAL = False
SAVE_LATEX = False
LATEX_OUTPUT_PATH = "../figures_main/tables/fig3_etpr_macro_table.tex"
TABLE_ONLY = False
INTERMEDIATE_FILE_PATH = "intermediate_files/etpr_results_original.parquet"

# Hand-tuned label positions for the circular plot
TABLE_POSITIONS = {
    "macro_precision": {"NIFD": (-0.25, 1.07), "NACC": (-0.5, 0.9), "BrainLat": (0.34, 0.98),
                        "ADNI": (0.3, 1.2), "PPMI": (-0.15, 0.92), "Other": (0.15, 1.08)},
    "macro_recall": {"NIFD": (-0.25, 0.9), "NACC": (-0.5, 0.9), "BrainLat": (0.34, 0.98),
                     "ADNI": (0.3, 1.1), "PPMI": (-0.15, 0.92), "Other": (0.15, 1.08)},
    "macro_f1": {"NIFD": (-0.25, 0.9), "NACC": (-0.5, 0.9), "BrainLat": (0.34, 0.98),
                 "ADNI": (0.3, 1.1), "PPMI": (-0.15, 0.92), "Other": (0.15, 1.08)},
}
VALUE_LABEL_OFFSETS = {
    "NIFD": {"r_offset": 0.06, "theta_offset": 0.0},
    "NACC": {"r_offset": 0.12, "theta_offset": -0.2},
    "BrainLat": {"r_offset": 0.15, "theta_offset": 0.02},
    "ADNI": {"r_offset": 0.14, "theta_offset": -0.03},
    "PPMI": {"r_offset": 0.14, "theta_offset": 0.1},
}
MAX_TABLE_LINES = 6  # comparisons shown per cohort in the annotation box

# Main
def main() -> None:
    print("Loading ETPR results...")
    if LOAD_FROM_ORIGINAL or not Path(INTERMEDIATE_FILE_PATH).exists():
        print("Loading ETPR results from original dataset...")
        results_df = build_results_df(DATASET_PATHS, MODEL_MAP.keys())
        results_df.to_parquet(INTERMEDIATE_FILE_PATH)
    else:
        print("Loading saved results...")
        results_df = pd.read_parquet(INTERMEDIATE_FILE_PATH)
        
    print(results_df.head())
    print(results_df.columns)
    print_sample_sizes(results_df, title="ETPR")
    
    print("Running bootstrap...")
    all_metrics = bootstrap_macro(results_df, n_boot=N_BOOT, seed=42, n_jobs=N_JOBS
                                  ).sort_values(["dataset", "model", "metric"])

    if SAVE_LATEX:
        save_text(latex_table_macro(all_metrics, MODEL_MAP), LATEX_OUTPUT_PATH)

    if TABLE_ONLY:
        return

    print("Running permutation tests...")
    pairwise = pairwise_macro(results_df, n_permutations=N_PERM, seed=42, n_jobs=N_JOBS
                              ).sort_values(["dataset", "metric", "model1", "model2"])

    plot_args = dict(
        all_metrics=all_metrics, pairwise_pvalues=pairwise, model_map=MODEL_MAP,
        model_order=MODEL_ORDER, dataset_order=DATASET_ORDER, output_dir=OUTPUT_FIG_DIR,
        fontsize=FONTSIZE, table_positions=TABLE_POSITIONS,
        value_label_offsets=VALUE_LABEL_OFFSETS, max_table_lines=MAX_TABLE_LINES,
    )
    print("Plotting macro precision + recall...")
    plot_macro_circular(**plot_args, metrics=("macro_precision", "macro_recall"),
                        filename=OUTPUT_FIGNAME_PR, figsize=FIGSIZE_PR)
    print("Plotting macro F1...")
    plot_macro_circular(**plot_args, metrics=("macro_f1",), filename=OUTPUT_FIGNAME_F1,
                        figsize=FIGSIZE_F1)


if __name__ == "__main__":
    main()
