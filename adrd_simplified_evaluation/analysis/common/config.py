"""Settings shared by all analysis scripts."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

BASE_RESULTS = Path("/projectnb/vkolagrp/projects/adrd_foundation_model/results")

# Result-directory model name -> display name (one spelling for every figure)
ALL_MODELS = {
    "Qwen2.5-3B-Instruct": "Q3B",
    "NACC-3B": "LUNAR-OS-SCe",
    "NACC-3B-SCE": "LUNAR-OS",
    "NACC-3B-OS": "LUNAR-SCe",
    "NACC-3B-OS-SCE": "LUNAR",
    "NACC-3B-OS-SFT": "SFT",
    "Qwen2.5-7B-Instruct": "Q7B",
}

N_BOOT = 1000
N_PERM = 10000
PERM_CHUNK = 1000  # permutations processed per chunk (memory control)
FONTSIZE = 7


def model_subset(*raw_names: str) -> dict[str, str]:
    """Ordered {raw name: display name} for the models a script uses."""
    return {name: ALL_MODELS[name] for name in raw_names}


def setup_style(hatch_linewidth: float | None = None, theme_context: str | None = None) -> None:
    """
    Arial + seaborn whitegrid. With `theme_context` (e.g. "paper"), seaborn's
    set_theme is applied as in the original training-dynamics script; note that
    set_theme resets rcParams, including the font family.
    """
    plt.rcParams["font.family"] = "Arial"
    if theme_context is not None:
        sns.set_theme(style="whitegrid", context=theme_context)
        return
    if hatch_linewidth is not None:
        mpl.rcParams["hatch.linewidth"] = hatch_linewidth
    sns.set_style("whitegrid")
