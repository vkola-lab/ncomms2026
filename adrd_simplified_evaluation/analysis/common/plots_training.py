"""Training-curve figures (main, ablation, per-task)."""

import math

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec
from matplotlib.legend_handler import HandlerTuple

from .bootstrap import EXTERNAL, INTERNAL

MARKERS = {EXTERNAL: "^", INTERNAL: "o"}
COLORS = {INTERNAL: "#ff7f00", EXTERNAL: "#377eb8"}
LINESTYLES = {"3B": "-", "7B": "--"}


# Helpers

def draw_baselines(ax, baselines: dict[str, pd.DataFrame], linewidth: float = 0.6,
                   band_alpha: float = 0.12) -> None:
    """One line + CI band per (model size, group). Color = group, line style = size."""
    for size, base_df in baselines.items():
        for _, r in base_df.iterrows():
            c = COLORS[r["in_distribution"]]
            ax.axhline(r["point"], color=c, linestyle=LINESTYLES[size], linewidth=linewidth, zorder=1)
            if band_alpha > 0:
                ax.axhspan(r["low"], r["high"], facecolor=c, alpha=band_alpha, linewidth=0, zorder=0)


def baseline_legend_handles(linewidth: float = 0.8) -> dict[str, mlines.Line2D]:
    return {size: mlines.Line2D([], [], color="black", linewidth=linewidth, linestyle=LINESTYLES[size])
            for size in ["3B", "7B"]}


def select_every_200_and_last(df: pd.DataFrame, step_col: str = "training_steps") -> pd.DataFrame:
    if df is None or df.empty:
        return df
    steps = np.sort(df[step_col].unique())
    keep = set(steps[steps % 200 == 0])
    keep.add(steps[-1])
    return df[df[step_col].isin(list(keep))].copy().sort_values(step_col)


def _plot_curves(ax, ci_df, linewidth, markersize, label_it=True, thin=False, keep_order=False):
    for label, group_df in ci_df.groupby("in_distribution", sort=not keep_order):
        if thin:
            group_df = select_every_200_and_last(group_df, step_col="training_steps")
        x = group_df["training_steps"].to_numpy()
        y = group_df["point"].to_numpy()
        yerr = np.vstack([y - group_df["low"].to_numpy(), group_df["high"].to_numpy() - y])
        ax.errorbar(x, y, yerr=yerr, linewidth=linewidth, marker=MARKERS.get(label, "x"),
                    markersize=markersize, alpha=0.9, color=COLORS.get(label),
                    label=label if label_it else None)


def _figure_legend(fig, axes, first_ax, fontsize, baseline_lw, anchor_y):
    baseline_handles = baseline_legend_handles(linewidth=baseline_lw)
    line_handles, line_labels = first_ax.get_legend_handles_labels()
    for ax in axes:
        if ax.get_legend():
            ax.get_legend().remove()
    fig.legend(line_handles + [baseline_handles["3B"], baseline_handles["7B"]],
               line_labels + ["Q3B", "Q7B"], loc="lower center", ncol=4, fontsize=fontsize,
               bbox_to_anchor=(0.5, anchor_y), frameon=False,
               handler_map={tuple: HandlerTuple(ndivide=None, pad=0.5)}, handlelength=3)


# Figures

def make_main_figure(ci_left, ci_right, baselines_left, baselines_right,
                     titles=("SFT", "LUNAR")) -> plt.Figure:
    """Two panels (e.g. SFT and LUNAR), each with internal/external curves and baselines."""
    fig = plt.figure(figsize=(4.6, 1.5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.3, 1.5])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharey=ax1)
    fontsize = 7

    panels = [(ax1, ci_left, baselines_left, "Macro-averaged\naccuracy", titles[0]),
              (ax2, ci_right, baselines_right, "", titles[1])]
    for ax, ci_df, baselines, ylabel, title in panels:
        _plot_curves(ax, ci_df.sort_values(["training_steps", "in_distribution"]),
                     linewidth=0.8, markersize=2, keep_order=True)
        draw_baselines(ax, baselines, linewidth=0.6)
        ax.set_title(title, fontsize=fontsize)
        ax.set_ylabel(ylabel, fontsize=fontsize)
        ax.set_xlabel("Training steps", fontsize=fontsize)
        ax.tick_params(axis="both", labelsize=fontsize)

    ax1.set_ylim(0.4, 0.75)
    _figure_legend(fig, [ax1, ax2], ax1, fontsize, baseline_lw=0.8, anchor_y=-0.1)
    plt.tight_layout()
    return fig


def make_ablation_figure(ci_list, baselines, panel_titles) -> plt.Figure:
    """2x2 panels, one per training variant, curves thinned to every 200 steps + last."""
    fontsize = 10
    fig, axes = plt.subplots(2, 2, figsize=(8, 5), sharey=True)
    axes_flat = axes.flatten()

    for i, ax in enumerate(axes_flat):
        ci_df = ci_list[i].copy()
        if "in_distribution" in ci_df.columns:
            _plot_curves(ax, ci_df, linewidth=1.5, markersize=4, label_it=(i == 0), thin=True)
        draw_baselines(ax, baselines, linewidth=0.8)
        ax.set_title(panel_titles[i], fontsize=fontsize + 1)
        ax.set_ylabel("Macro-averaged\naccuracy" if i % 2 == 0 else "", fontsize=fontsize)
        ax.set_xlabel("Training steps" if i in [2, 3] else "", fontsize=fontsize)
        ax.tick_params(axis="both", labelsize=fontsize)

    for ax in axes_flat:
        ax.set_ylim(0.4, 0.8)
    _figure_legend(fig, axes_flat, axes_flat[0], fontsize, baseline_lw=0.8, anchor_y=-0.02)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    return fig


def make_per_task_figure(ci_by_benchmark: dict[str, pd.DataFrame],
                         baselines_by_benchmark: dict[str, dict[str, pd.DataFrame]],
                         benchmark_labels: dict[str, str]) -> plt.Figure:
    """One panel per benchmark (in `benchmark_labels` order)."""
    benchmarks = list(benchmark_labels)
    ncols = 2
    nrows = math.ceil(len(benchmarks) / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6, 8), sharey=True)
    fontsize = 8

    for idx, benchmark in enumerate(benchmarks):
        col = idx % ncols
        ax = axes[idx // ncols, col] if nrows > 1 else axes[col]

        ci_bench = ci_by_benchmark[benchmark].sort_values(["training_steps", "in_distribution"])
        if "in_distribution" in ci_bench.columns:
            _plot_curves(ax, ci_bench, linewidth=1.5, markersize=4, label_it=(idx == 0))
        draw_baselines(ax, baselines_by_benchmark.get(benchmark, {}), linewidth=0.9)

        ax.set_title(benchmark_labels[benchmark], fontsize=fontsize + 1, fontweight="bold")
        ax.set_xlabel("Training steps", fontsize=fontsize)
        ax.set_ylabel("Macro-averaged\naccuracy" if col == 0 else "", fontsize=fontsize)
        ax.tick_params(axis="x", labelsize=fontsize - 1)
        ax.tick_params(axis="y", labelsize=fontsize - 1)

    for idx2 in range(len(benchmarks), nrows * ncols):
        (axes[idx2 // ncols, idx2 % ncols] if nrows > 1 else axes[idx2 % ncols]).axis("off")

    first_ax = axes[0, 0] if nrows > 1 else axes[0]
    all_axes = axes.flatten() if nrows > 1 else axes
    _figure_legend(fig, all_axes, first_ax, fontsize, baseline_lw=0.9, anchor_y=-0.03)
    plt.tight_layout(rect=[0, 0.07, 1, 1])
    return fig


# Training dynamics from W&B logs (reward, reward std, entropy, response length)

def make_training_dynamics_figure(panels, order, dashes, markers, figsize=(4.3, 2.3),
                                  fontsize=7, linewidth=1, markersize=4, markevery=100,
                                  step_col="train/global_step"):
    """
    2x2 grid of smoothed training curves, one line per variant.
    panels: list of four (long_df, value_col, ylabel), filled row by row.
    """
    import seaborn as sns

    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=True, constrained_layout=False)
    handles, labels = [], []

    for idx, (ax, (data, value_col, ylabel)) in enumerate(zip(axes.ravel(), panels)):
        sns.lineplot(data=data, x=step_col, y=value_col, hue="Variant", hue_order=order,
                     palette="colorblind", style="Variant", style_order=order, dashes=dashes,
                     markers=markers, markevery=markevery, markersize=markersize,
                     linewidth=linewidth, ax=ax, legend="auto" if idx == 0 else False)
        if idx == 0:
            handles, labels = ax.get_legend_handles_labels()
            if ax.legend_ is not None:
                ax.legend_.remove()
        ax.set_ylabel(ylabel, fontsize=fontsize)

    axes[1, 0].set_xlabel("Training steps", fontsize=fontsize)
    axes[1, 1].set_xlabel("Training steps", fontsize=fontsize)
    for ax in axes.ravel():
        ax.tick_params(axis="both", labelsize=fontsize)

    label_to_handle = dict(zip(labels, handles))
    legend_labels = [l for l in order if l in label_to_handle]
    fig.legend(handles=[label_to_handle[l] for l in legend_labels], labels=legend_labels,
               loc="lower center", ncol=len(legend_labels), frameon=False, fontsize=fontsize,
               bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    return fig
