"""COG: per-class precision / recall / F1, NACC vs all other cohorts pooled."""

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
    "NACC-3B",
    "NACC-3B-SCE",
    "NACC-3B-OS",
    "NACC-3B-OS-SCE",
    "Qwen2.5-7B-Instruct",
)

CLASS_MAP = {
    "Not applicable (no cognitive impairment)": "NC",
    "Alzheimer's disease (AD)": "AD",
    "Frontotemporal lobar degeneration and its variants, including primary progressive aphasia, corticobasal degeneration and progressive supranuclear palsy, and with or without amyotrophic lateral sclerosis (FTLD)": "FTLD",
    "Lewy body disease (LBD)": "LBD",
    "Vascular brain injury or vascular dementia including stroke (VD)": "VD",
    "Idiopathic Parkinson's Disease": "Idiopathic PD",
    "Systemic and environmental factors including infectious diseases (HIV included), metabolic, substance abuse / alcohol, medications, systemic disease and delirium (SEF)": "SEF",
    "Psychiatric conditions including schizophrenia, depression, bipolar disorder, anxiety and posttraumatic stress disorder (PSY)": "PSY",
    "Other (Multiple system atrophy, Essential tremor, Down syndrome, Huntington's disease, Prion disease, Traumatic brain injury, Normal-pressure hydrocephalus, Epilepsy, CNS neoplasm, etc)": "Other",
    "Prodromal Parkinson's Disease": "Prodromal PD",
    "No PD nor other neurological disorder": "No PD/ND",
    "Other neurological disorder(s)": "Other ND"
}
CLASS_ORDER = list(CLASS_MAP.values())

# Order matters for the row index (`trial`); keep as is for reproducibility

DATASET_PATHS = {
    "NACC": {"NACC": BASE_RESULTS / "NACC/test_etpr"},
    "ADNI": {"ADNI": BASE_RESULTS / "ADNI/test_etpr"},
    "BrainLat": {"BrainLat": BASE_RESULTS / "brainlat/test_etpr"},
    "NIFD": {"NIFD": BASE_RESULTS / "NIFD/test_etpr"},
    "PPMI": {"PPMI": BASE_RESULTS / "PPMI/test_etpr"},
}

LATEX_OUTPUT_PATHS = {
    "NACC": "../figures_main/tables/fig3_etpr_nacc_table.tex",
    "ADNI": "../figures_main/tables/fig3_etpr_adni_table.tex",
    "BrainLat": "../figures_main/tables/fig3_etpr_brainlat_table.tex",
    "NIFD": "../figures_main/tables/fig3_etpr_nifd_table.tex",
    "PPMI": "../figures_main/tables/fig3_etpr_ppmi_table.tex",
}
POOL_AS_OTHER = []
N_JOBS_BOOT = 4


# Main

def main() -> None:
    for k, v in DATASET_PATHS.items():
        print(f"Loading {k} results...")
        results_df = build_results_df(v, MODEL_MAP.keys(), pool_as_other=POOL_AS_OTHER)

        print("Running bootstrap...")
        all_metrics = bootstrap_per_class(results_df, n_boot=N_BOOT, seed=42, n_jobs=N_JOBS_BOOT,
                                        stratify_col="dataset_raw")

        save_text(latex_table_per_class(all_metrics, MODEL_MAP, CLASS_MAP, CLASS_ORDER),
                    LATEX_OUTPUT_PATHS[k])


if __name__ == "__main__":
    main()
