"""Training dynamics from W&B logs: group reward, reward std, token entropy, response length."""

import warnings

from common.config import setup_style
from common.loading import ema_smooth_long, load_wandb_metric, wandb_to_long
from common.plots_training import make_training_dynamics_figure

warnings.filterwarnings("ignore")
setup_style(theme_context="paper")

# Configuration

WANDB_DIR = "intermediate_files/wandb"  # W&B CSV exports

# (csv file, W&B metric key, value name, y-axis label), in panel order (row by row)
METRICS = [
    ("wandb_export_reward.csv", "train/reward", "Group rewards", "Mean group\nrewards"),
    ("wandb_export_reward_std.csv", "train/reward_std", "Group rewards std", "Group rewards\nstd"),
    ("wandb_export_entropy.csv", "train/entropy", "Entropy", "Mean token\nentropy"),
    ("wandb_export_mean_length.csv", "mean_length", "Mean response length",
     "Mean response\nlength (tokens)"),
]

# W&B run name -> display name
RUN_NAMES = {
    "qwen2.5 3B nacc inc oversample dedup": "LUNAR-OS-SCe",
    "qwen2.5 3B nacc inc oversample dedup sce tanh": "LUNAR-OS",
    "qwen2.5 3B nacc inc oversample": "LUNAR-SCe",
    "qwen2.5 3B nacc inc oversample sce tanh cont": "LUNAR",
}
# The LUNAR run was resumed: fill its gaps from the original run, then drop the original
MERGE_RUNS = {"qwen2.5 3B nacc inc oversample sce tanh cont": "qwen2.5 3B nacc inc oversample sce tanh"}

ORDER = ["LUNAR-OS-SCe", "LUNAR-OS", "LUNAR-SCe", "LUNAR"]
DASHES = {"LUNAR": "", "LUNAR-SCe": (2, 2), "LUNAR-OS": (5, 2), "LUNAR-OS-SCe": (2, 1, 1, 1)}
MARKERS = {"LUNAR": "o", "LUNAR-SCe": "s", "LUNAR-OS": "^", "LUNAR-OS-SCe": "D"}

EMA_ALPHA = 0.01
FIGSIZE = (4.3, 2.7)
OUT_PATH = "../figures_main/fig2_train_dynamics.pdf"


def main() -> None:
    print("Loading data")
    panels = []
    for csv_name, key, value_name, ylabel in METRICS:
        df = load_wandb_metric(f"{WANDB_DIR}/{csv_name}", key, RUN_NAMES, MERGE_RUNS)
        long = ema_smooth_long(wandb_to_long(df, value_name, ORDER), value_name, alpha=EMA_ALPHA)
        panels.append((long, value_name, ylabel))

    print("Making figure")
    fig = make_training_dynamics_figure(panels, ORDER, DASHES, MARKERS, figsize=FIGSIZE)
    fig.savefig(OUT_PATH, dpi=200, format="pdf", bbox_inches="tight")
    print(f"Saved figure to {OUT_PATH}")


if __name__ == "__main__":
    main()
