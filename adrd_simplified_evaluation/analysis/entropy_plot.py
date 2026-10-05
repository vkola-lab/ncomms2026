"""Mean token entropy per model: half-violin + box, paired permutation tests."""

from common.config import N_PERM, setup_style
from common.loading import load_entropy_summaries
from common.permutation import pairwise_mean_tests
from common.plots import plot_half_violins
from common.reporting import letter_pmatrix

setup_style(hatch_linewidth=0.5)

# Configuration

DATA_DIR = "intermediate_files/entropies/summaries"  # JSON files written by entropy/entropy_generate.py
OUT_PATH = "../figures_main/fig2_entropy_distribution.pdf"

# Entropy file name -> display name
KEY_NAMES = {
    "q3b": "Q3B",
    "oversample_dedup": "LUNAR-OS-SCe",
    "oversample_dedup_sce_tanh": "LUNAR-OS",
    "oversample": "LUNAR-SCe",
    "oversample_sce_tanh": "LUNAR",
    "q7b": "Q7B",
}
KEY_ORDER = ["Q3B", "LUNAR-OS-SCe", "LUNAR-SCe", "LUNAR-OS", "LUNAR", "Q7B"]

YLIM = (0, 2.5)          # raise if the printed max mean entropy is higher
SHOW_LEGEND = False      # the figure sits next to the output-length figure (same letters)


def main():
    df = load_entropy_summaries(DATA_DIR, KEY_NAMES, KEY_ORDER)

    res = pairwise_mean_tests(df, id_col="q_idx", model_col="method", value_col="mean_entropy",
                              model_order=KEY_ORDER, n_perms=N_PERM)
    for _, r in res.iterrows():
        print(f"{r['model1']} vs {r['model2']}: {r['n_paired']} paired questions, "
              f"mean diff = {r['observed_diff']:+.3f} nats, p_bh = {r['p_value_bh']:.4g}")

    letters = {k: chr(65 + i) for i, k in enumerate(KEY_ORDER)}
    fig, _ = plot_half_violins(
        {k: df.loc[df["method"] == k, "mean_entropy"].to_numpy() for k in KEY_ORDER},
        KEY_ORDER, letter_pmatrix(res, KEY_ORDER, letters),
        ylabel="Mean token entropy (nats)", figsize=(2.3, 1.8), box_alpha=0.7,
        ylim=YLIM, show_legend=SHOW_LEGEND,
    )
    fig.savefig(OUT_PATH, dpi=200, format="pdf", bbox_inches="tight")
    print(f"Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
