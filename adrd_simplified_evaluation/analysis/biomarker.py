"""Biomarker (amyloid PET, amyloid CSF, DAT): macro precision / recall / F1.

Bootstrap, permutation tests and LaTeX tables for every model in MODEL_MAP.
Forest plot (NACC on top, external cohort below) for the models in PLOT_MODELS.
"""

import os

import matplotlib.pyplot as plt
import pandas as pd

from common.bootstrap import bootstrap_macro
from common.config import BASE_RESULTS, N_BOOT, N_PERM, model_subset, setup_style
from common.loading import build_results_df, print_sample_sizes 
from common.permutation import pairwise_macro
from common.plots import forest_plot_biomarker
from common.reporting import latex_table_macro, save_text
from pathlib import Path
import pandas as pd
setup_style()

# Configuration

# Bootstrap, permutation tests (BH over all pairs) and LaTeX tables
MODEL_MAP = model_subset(
    "Qwen2.5-3B-Instruct",
    # "NACC-3B",
    # "NACC-3B-SCE",
    # "NACC-3B-OS",
    "NACC-3B-OS-SCE",
    "Qwen2.5-7B-Instruct",
)

# Models drawn in the forest plot (display names; order = left to right)
PLOT_MODELS = {"Q3B": "#0173B2", "LUNAR": "#DE8F05", "Q7B": "#029E73"}  # name -> colour
PLOT_MARKERS = {"Q3B": "o", "LUNAR": "s", "Q7B": "^"}

MODALITY_PATHS = {
    "PET": {"NACC": BASE_RESULTS / "NACC/test_pet", "ADNI": BASE_RESULTS / "ADNI/test_pet"},
    "CSF": {"NACC": BASE_RESULTS / "NACC/test_csf", "ADNI": BASE_RESULTS / "ADNI/test_csf"},
    "DAT": {"NACC": BASE_RESULTS / "NACC/test_dat", "PPMI": BASE_RESULTS / "PPMI/test_dat"},
}
INTERNAL_COHORT = "NACC"
EXTERNAL_COHORT = {"PET": "ADNI", "CSF": "ADNI", "DAT": "PPMI"}
BIOMARKER_LABELS = {"PET": "Amyloid PET", "CSF": "Amyloid CSF", "DAT": "DAT"}

PLOT_METRICS = ["macro_f1", "macro_precision", "macro_recall"]
FIGSIZE = (4, 4)
USE_FDR = True  # brackets show BH-adjusted p-values

N_JOBS_BOOT = 20
N_JOBS_PERM = 1

OUTPUT_FIG_DIR = "../figures_main"
OUTPUT_FIGNAME = {m: f"forest_biomarkers_{m}_idlevel" for m in PLOT_METRICS}

LOAD_FROM_ORIGINAL = False
SAVE_LATEX = False
LATEX_OUTPUT_PATH = {m: f"../figures_main/tables/fig3_biomarker_{m.lower()}_table.tex" for m in MODALITY_PATHS}
TABLE_ONLY = False
INTERMEDIATE_FILE_PATH = {m: f"intermediate_files/biomarker_{m}_results_original.parquet" for m in MODALITY_PATHS}

def to_display_names(df: pd.DataFrame, cols) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        out[col] = out[col].map(MODEL_MAP).fillna(out[col])
    return out


def main() -> None:
    nacc_boot, nacc_perm, ext_boot, ext_perm = {}, {}, {}, {}

    for modality, paths in MODALITY_PATHS.items():
        if LOAD_FROM_ORIGINAL or not Path(INTERMEDIATE_FILE_PATH[modality]).exists():
            print(f"Loading {modality} results from original dataset...")
            results_df = build_results_df(paths, MODEL_MAP.keys())
            results_df.to_parquet(INTERMEDIATE_FILE_PATH[modality])
        else:
            print(f"Loading saved {modality} results...")
            results_df = pd.read_parquet(INTERMEDIATE_FILE_PATH[modality])
            
        print(results_df.head())
        print(results_df.columns)
        print_sample_sizes(results_df, title=f"{modality}")
        
        print(f"Running bootstrap ({modality})...")
        boot = bootstrap_macro(results_df, n_boot=N_BOOT, seed=42, n_jobs=N_JOBS_BOOT)

        if SAVE_LATEX:
            save_text(latex_table_macro(boot, MODEL_MAP), LATEX_OUTPUT_PATH[modality])

        if TABLE_ONLY:
            continue

        print(f"Running permutation tests ({modality}, all models)...")
        perm = pairwise_macro(results_df, n_permutations=N_PERM, seed=42, n_jobs=N_JOBS_PERM)

        boot = to_display_names(boot, ["model"])
        perm = to_display_names(perm, ["model1", "model2"])
        external = EXTERNAL_COHORT[modality]
        nacc_boot[modality] = boot[boot["dataset"] == INTERNAL_COHORT]
        nacc_perm[modality] = perm[perm["dataset"] == INTERNAL_COHORT]
        ext_boot[modality] = boot[boot["dataset"] == external]
        ext_perm[modality] = perm[perm["dataset"] == external]

    if TABLE_ONLY:
        return

    os.makedirs(OUTPUT_FIG_DIR, exist_ok=True)
    for metric in PLOT_METRICS:
        fig, _ = forest_plot_biomarker(
            nacc_boot, nacc_perm, ext_boot, ext_perm, metric=metric,
            biomarker_order=list(MODALITY_PATHS), biomarker_labels=BIOMARKER_LABELS,
            model_colors=PLOT_MODELS, model_markers=PLOT_MARKERS,
            figsize=FIGSIZE, use_fdr=USE_FDR,
        )
        path = os.path.join(OUTPUT_FIG_DIR, f"{OUTPUT_FIGNAME[metric]}.pdf")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {path}")


if __name__ == "__main__":
    main()