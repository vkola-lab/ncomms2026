"""Output length (tokens) per model: half-violin + box on a log scale, paired permutation tests."""

import numpy as np
import pandas as pd
from pathlib import Path
from common.config import ALL_MODELS, BASE_RESULTS, N_PERM, setup_style
from common.loading import TOKEN_CSV_COLS, load_token_counts
from common.permutation import pairwise_mean_tests
from common.plots import plot_half_violins
from common.reporting import letter_pmatrix

setup_style(hatch_linewidth=0.5)

# Configuration

COMPUTE_TOKEN_COUNTS = False  # True = tokenize parquet files and save; False = load CSV
INTERMEDIATE_FILE_PATH = "intermediate_files/token_counts_original.parquet"

DATASET_PATHS = {
    "NIFD": BASE_RESULTS / "NIFD",
    "ADNI": BASE_RESULTS / "ADNI",
    "NACC": BASE_RESULTS / "NACC",
    "PPMI": BASE_RESULTS / "PPMI",
    "BrainLat": BASE_RESULTS / "brainlat", 
}
SKIP_BENCHMARKS = ["MCI", "NP", "NP_MIXED", "FTLD"]

MODEL_ORDER = ["Q3B", "LUNAR-OS-SCe", "LUNAR-SCe", "LUNAR-OS", "LUNAR", "Q7B"]

PERM_CHUNK_TOKENS = 200  # ~73k questions per model; keeps each chunk ~100 MB
YLIM = (10, 6e4)
OUT_PATH = "../figures_main/fig2_output_length_all.pdf"


def main() -> None:
    if COMPUTE_TOKEN_COUNTS or not Path(INTERMEDIATE_FILE_PATH).exists():
        print("Computing token counts from parquet files...")
        df = pd.concat([load_token_counts(p, name, ALL_MODELS, MODEL_ORDER, SKIP_BENCHMARKS)
                        for name, p in DATASET_PATHS.items()], ignore_index=True)
        print(f"Saving token counts to {INTERMEDIATE_FILE_PATH}...")
        df.to_parquet(INTERMEDIATE_FILE_PATH)
    else:
        print(f"Loading token counts from {INTERMEDIATE_FILE_PATH}...")
        df = pd.read_parquet(INTERMEDIATE_FILE_PATH)
    
    print(df.head())
    print(df.columns)
    # return

    print("Truncation rates (share of finish_reason values per model):")
    print(pd.crosstab(df["model"], df["finish_reason"]))    
    print(pd.crosstab(df["model"], df["finish_reason"], normalize="index"))
    # return

    # All responses kept (incl. truncated). Test on per-question mean log10 length.
    df["log_val"] = np.log10(df["token_count"].clip(lower=1))
    df["perm_id"] = (df["dataset"].astype(str) + "__" + df["benchmark"].astype(str)
                     + "__" + df["ID"].astype(str))
    res = pairwise_mean_tests(df, id_col="perm_id", model_col="model", value_col="log_val",
                              model_order=MODEL_ORDER, n_perms=N_PERM, chunk=PERM_CHUNK_TOKENS)
    for _, r in res.iterrows():
        print(f"{r['model1']} vs {r['model2']}: {r['n_paired']} paired questions, "
              f"ratio = {10 ** r['observed_diff']:.2f}x, p_bh = {r['p_value_bh']:.4g}")
        
    obs = {(r.model1, r.model2): r.observed_diff for r in res.itertuples()}

    def ratio(a, b):
        """Geometric mean length of a / b."""
        return 10 ** obs[(a, b)] if (a, b) in obs else 10 ** (-obs[(b, a)])

    for row in ["Q3B", "Q7B", "LUNAR-SCe", "LUNAR-OS-SCe"]:
        print(f"{row:14s} LUNAR: {ratio('LUNAR', row):.2f}   LUNAR-OS: {ratio('LUNAR-OS', row):.2f}")

    letters = {m: chr(65 + i) for i, m in enumerate(MODEL_ORDER)}
    # per_q = df.groupby(["perm_id", "model"], observed=True)["log_val"].mean().reset_index()

    fig, _ = plot_half_violins(
        {m: df.loc[df["model"] == m, "log_val"].to_numpy() for m in MODEL_ORDER},
        MODEL_ORDER, letter_pmatrix(res, MODEL_ORDER, letters),
        ylabel="Output length (tokens)", ylim=YLIM, log10_ticks=True,
    )
    fig.savefig(OUT_PATH, dpi=200, format="pdf", bbox_inches="tight")
    print(f"Saved figure to {OUT_PATH}")


if __name__ == "__main__":
    main()
