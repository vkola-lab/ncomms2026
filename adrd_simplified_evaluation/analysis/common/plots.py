"""All figure code except the training curves (see plots_training.py)."""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib.patches import Patch, Rectangle

from .reporting import annotate_pmatrix, significance_marker

METRIC_NAMES = {
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
    "macro_precision": "Macro Precision",
    "macro_recall": "Macro Recall",
    "macro_f1": "Macro F1",
}


def save_figure(fig_or_plt, output_dir: str, figname: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{figname}.pdf")
    fig_or_plt.savefig(path, bbox_inches="tight", dpi=300)
    print(f"Saved {path}")


def prepare_plot_tables(all_metrics, pairwise_pvalues, model_map, class_map, metrics):
    """Add abbreviated model/class columns and filter to `metrics`."""
    df = all_metrics.copy()
    df["model_abbrev"] = df["model"].map(model_map)
    df["class_abbrev"] = df["class"].map(class_map).fillna(df["class"])

    pv = pairwise_pvalues.copy()
    pv["model1_abbrev"] = pv["model1"].apply(lambda x: model_map.get(x, x))
    pv["model2_abbrev"] = pv["model2"].apply(lambda x: model_map.get(x, x))
    pv["class_abbrev"] = pv["class"].map(class_map).fillna(pv["class"])

    return df[df["metric"].isin(metrics)].copy(), pv


def comparisons_for_class(pval_cls, models, show_all_comparisons):
    if show_all_comparisons:
        return pval_cls
    baseline = models[0]
    return pval_cls[(pval_cls["model1_abbrev"] == baseline) | (pval_cls["model2_abbrev"] == baseline)]


def class_values(df_model: pd.DataFrame, classes):
    """Point estimates and asymmetric error bars per class (0 if missing)."""
    values, err_low, err_high = [], [], []
    for cls in classes:
        cls_data = df_model[df_model["class_abbrev"] == cls]
        if len(cls_data) > 0:
            point, low, high = (cls_data[c].values[0] for c in ("point", "low", "high"))
            values.append(point)
            err_low.append(point - low)
            err_high.append(high - point)
        else:
            values.append(0.0)
            err_low.append(0.0)
            err_high.append(0.0)
    return values, err_low, err_high


def draw_brackets(ax, comparisons, pos_info, cls_idx, models, base_height,
                  fontsize, y_offset=0.15, y_step=0.12, fontsize_marker=None):
    """Draw significance brackets for one class, stacked upwards."""
    for comp_idx, (_, row) in enumerate(comparisons.sort_values("p_value_bh").iterrows()):
        m1, m2 = row["model1_abbrev"], row["model2_abbrev"]
        if m1 not in models or m2 not in models:
            continue
        x1 = pos_info[m1]["x"][cls_idx]
        x2 = pos_info[m2]["x"][cls_idx]
        y = base_height + y_offset + comp_idx * y_step
        ax.plot([x1, x1, x2, x2], [y - 0.01, y, y, y - 0.01], "k-", linewidth=0.6)
        ax.text((x1 + x2) / 2, y + 0.005, significance_marker(row["p_value_bh"]),
                ha="center", va="bottom",
                fontsize=fontsize if fontsize_marker is None else fontsize_marker,
                fontweight="bold")


def plot_classwise_points(
    all_metrics: pd.DataFrame,
    pairwise_pvalues: pd.DataFrame,
    model_map: dict[str, str],
    class_map: dict[str, str],
    class_order: list[str],
    dataset_order: list[str],
    metrics=("precision", "recall"),
    output_dir: str = ".",
    show_all_comparisons: bool = False,
    figname: str = "classwise_points",
    figsize=(4.0, 4.0),
    fontsize: int = 7,
    legend_ncol: int | None = None,
    dataset_titles: dict[str, str] | None = None,
) -> None:
    """
    Per-class point + CI plot with significance brackets (NP-one, biomarker).
    `dataset_titles` adds a title above each dataset's column (top row only).
    The x-range widens with the number of models so no point is clipped.
    """
    metrics = list(metrics)
    df, pvalues = prepare_plot_tables(all_metrics, pairwise_pvalues, model_map, class_map, metrics)

    models = list(model_map.values())
    palette = dict(zip(models, sns.color_palette("colorblind", n_colors=len(models))))
    markers = dict(zip(models, ["o", "s", "^", "D", "v", "P"]))

    datasets = [d for d in dataset_order if d in df["dataset"].unique()]
    n_models = len(models)
    group_gap, point_offset = 0.3, 0.1
    y_step = 0.12

    fig, axes = plt.subplots(len(metrics), len(datasets), figsize=figsize, squeeze=False)

    for row_idx, metric in enumerate(metrics):
        for col_idx, dataset in enumerate(datasets):
            ax = axes[row_idx, col_idx]

            present = df[df["dataset"] == dataset]["class_abbrev"].unique()
            classes = [c for c in class_order if c in present]
            n_classes = len(classes)
            centers = np.arange(n_classes) * group_gap
            df_subset = df[(df["dataset"] == dataset) & (df["metric"] == metric)]

            for cls_idx in range(n_classes - 1):
                mid = (centers[cls_idx] + centers[cls_idx + 1]) / 2
                ax.plot([mid, mid], [0, 1.0], color="#e0e0e0", linewidth=0.8, linestyle=":", zorder=0)

            pos_info = {}
            for i, model in enumerate(models):
                values, err_low, err_high = class_values(
                    df_subset[df_subset["model_abbrev"] == model], classes)
                x_pos = centers + (i - (n_models - 1) / 2) * point_offset
                pos_info[model] = {"x": x_pos, "height": np.array(values) + np.array(err_high)}

                ax.errorbar(x_pos, values, yerr=[err_low, err_high], fmt=markers[model],
                            color=palette[model], markersize=2, capsize=2, linewidth=0.8,
                            markeredgecolor=palette[model], markerfacecolor=palette[model],
                            label=model if (row_idx == 0 and col_idx == len(datasets) - 1) else "",
                            zorder=5)
                for x, val, eh in zip(x_pos, values, err_high):
                    if val > 0:
                        ax.text(x, val + eh + 0.02, f"{val:.2f}", ha="center", va="bottom",
                                fontsize=fontsize - 1, color="black")

            pval_subset = pvalues[(pvalues["dataset"] == dataset) & (pvalues["metric"] == metric)]
            max_height = max((np.max(pos_info[m]["height"]) for m in models
                              if len(pos_info[m]["height"]) > 0), default=0)

            max_brackets = 0
            for cls_idx, cls in enumerate(classes):
                pval_cls = pval_subset[pval_subset["class_abbrev"] == cls]
                if len(pval_cls) == 0:
                    continue
                comps = comparisons_for_class(pval_cls, models, show_all_comparisons)
                max_brackets = max(max_brackets, len(comps))
                draw_brackets(ax, comps, pos_info, cls_idx, models, max_height, fontsize,
                              y_step=y_step)

            ax.set_ylabel(METRIC_NAMES.get(metric, metric), fontsize=fontsize)
            ax.yaxis.set_label_coords(-0.1, 0.35)
            ax.set_xticks(centers)
            ax.set_xticklabels(classes, rotation=30, ha="center", fontsize=fontsize)
            if n_classes > 0:
                half = max(group_gap * 0.55, (n_models - 1) / 2 * point_offset + point_offset * 0.6)
                ax.set_xlim(centers[0] - half, centers[-1] + half)
            if dataset_titles and row_idx == 0:
                ax.set_title(dataset_titles.get(dataset, dataset), fontsize=fontsize)

            has_pvals = len(pval_subset[pval_subset["class_abbrev"].isin(classes)]) > 0
            ax.set_ylim(0, 0.9 + max_brackets * y_step if has_pvals else 1.2)
            yticks = ax.get_yticks()
            yticks = yticks[yticks <= 1.0]
            ax.set_yticks(yticks)
            ax.set_yticklabels([f"{y:.1f}" for y in yticks], fontsize=fontsize)

            ax.grid(True, color="#e0e0e0", linewidth=0.5, linestyle=":", alpha=0.3, axis="y")
            ax.grid(False, axis="x")
            ax.spines["left"].set_bounds(0, 1.0)
            sns.despine(ax=ax, right=True, top=True)

    handles = [plt.Line2D([0], [0], marker=markers[m], color="w", markerfacecolor=palette[m],
                          markeredgecolor=palette[m], markersize=5, label=m) for m in models]
    fig.legend(handles=handles, title="Model", loc="upper center",
               ncol=legend_ncol or len(models), bbox_to_anchor=(0.5, 0.05), frameon=True,
               fontsize=fontsize, title_fontsize=fontsize)

    plt.tight_layout(rect=[0, 0.02, 1, 1])
    save_figure(plt, output_dir, figname)


# Per-class bar plot (COG)

def plot_classwise_bars(
    all_metrics: pd.DataFrame,
    pairwise_pvalues: pd.DataFrame,
    model_map: dict[str, str],
    class_map: dict[str, str],
    class_order: list[str],
    dataset_order: list[str],
    metrics=("precision", "recall"),
    output_dir: str = ".",
    figname: str = "classwise_bars",
    figsize=(4.5, 4),
    fontsize: int = 7,
    show_all_comparisons: bool = True,
    dataset_titles: dict[str, str] | None = None,
    boxed_dataset: str | None = "NACC",
) -> None:
    """Per-class bars with CIs and significance brackets; one column per dataset."""
    metrics = list(metrics)
    df, pvalues = prepare_plot_tables(all_metrics, pairwise_pvalues, model_map, class_map, metrics)

    models = list(model_map.values())
    palette = dict(zip(models, sns.color_palette("colorblind", n_colors=len(models))))
    datasets = [d for d in dataset_order if d in df["dataset"].unique()]
    n_models = len(models)
    bar_width = 0.8 / n_models
    y_step = 0.12

    fig, axes = plt.subplots(len(metrics), len(datasets), figsize=figsize, squeeze=False)

    for row_idx, metric in enumerate(metrics):
        for col_idx, dataset in enumerate(datasets):
            ax = axes[row_idx, col_idx]

            present = df[df["dataset"] == dataset]["class_abbrev"].unique()
            classes = [c for c in class_order if c in present]
            df_subset = df[(df["dataset"] == dataset) & (df["metric"] == metric)]

            pos_info = {}
            for i, model in enumerate(models):
                values, err_low, err_high = class_values(
                    df_subset[df_subset["model_abbrev"] == model], classes)
                x_pos = np.arange(len(classes)) + i * bar_width - (n_models - 1) * bar_width / 2
                pos_info[model] = {"x": x_pos, "height": np.array(values) + np.array(err_high)}

                bars = ax.bar(
                    x_pos, values, bar_width,
                    label=model if (row_idx == 0 and col_idx == len(datasets) - 1) else "",
                    color=palette[model], alpha=0.8, yerr=[err_low, err_high], capsize=2,
                    error_kw={"linewidth": 0.6, "elinewidth": 0.6, "capthick": 0.6},
                )
                for bar, val, eh in zip(bars, values, err_high):
                    if val > 0:
                        ax.text(bar.get_x() + bar.get_width() / 2, val + eh + 0.02, f"{val:.2f}",
                                ha="center", va="bottom", fontsize=fontsize - 2.5, color="black")

            pval_subset = pvalues[(pvalues["dataset"] == dataset) & (pvalues["metric"] == metric)]
            max_height = max((np.max(pos_info[m]["height"]) for m in models
                              if len(pos_info[m]["height"]) > 0), default=0)

            max_brackets = 0
            for cls_idx, cls in enumerate(classes):
                pval_cls = pval_subset[pval_subset["class_abbrev"] == cls]
                if len(pval_cls) == 0:
                    continue
                comps = comparisons_for_class(pval_cls, models, show_all_comparisons)
                max_brackets = max(max_brackets, len(comps))
                draw_brackets(ax, comps, pos_info, cls_idx, models, max_height, fontsize,
                              y_step=y_step, fontsize_marker=fontsize - 2)

            ax.set_ylim(0, 1.2 + max_brackets * y_step)
            yticks = ax.get_yticks()
            yticks = yticks[yticks <= 1.0]
            ax.set_yticks(yticks)
            ax.set_yticklabels([f"{y:.1f}" for y in yticks], fontsize=fontsize)

            # if "f1" in metrics and col_idx != 1:
            #     ax.set_ylabel(METRIC_NAMES.get(metric, metric), fontsize=fontsize)
            
            if col_idx == 0:
                ax.set_ylabel(METRIC_NAMES.get(metric, metric), fontsize=fontsize)

            ax.set_xticks(np.arange(len(classes)))
            ax.set_xticklabels(classes, rotation=0, ha="right", fontsize=fontsize)

            if row_idx == 0:
                ax.set_title((dataset_titles or {}).get(dataset, dataset), fontsize=fontsize)

            ax.grid(True, alpha=0.3, axis="y")
            ax.grid(False, axis="x")
            sns.despine(ax=ax, left=True, bottom=True, right=True, top=True)
            if dataset == boxed_dataset:
                for spine in ["left", "bottom", "right", "top"]:
                    ax.spines[spine].set_visible(True)
                    ax.spines[spine].set_linewidth(1.0)

    # handles = [plt.Rectangle((0, 0), 1, 1, fc=palette[m], alpha=0.8, label=m) for m in models]
    # fig.legend(handles=handles, title="Model", loc="lower center", ncol=len(models),
    #            bbox_to_anchor=(0.5, -0.05), frameon=True, fontsize=fontsize,
    #            title_fontsize=fontsize)

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    save_figure(plt, output_dir, figname)


# Circular macro bar plot (ETPR)

def plot_macro_circular(
    all_metrics: pd.DataFrame,
    pairwise_pvalues: pd.DataFrame,
    model_map: dict[str, str],
    model_order,
    dataset_order,
    metrics=("macro_precision", "macro_recall"),
    output_dir: str = ".",
    filename: str = "macro_circular",
    figsize=(2.3, 2.3),
    fontsize: int = 7,
    table_positions: dict | None = None,
    value_label_offsets: dict | None = None,
    max_table_lines: int = 6,
) -> None:
    """Polar bars: one wedge group per dataset, one bar per model, significance box per dataset."""
    metrics = list(metrics)
    table_positions = table_positions or {}
    value_label_offsets = value_label_offsets or {}

    df = all_metrics.copy()
    df["model_abbrev"] = df["model"].map(model_map).fillna(df["model"])
    pv = pairwise_pvalues.copy()
    pv["model1_abbrev"] = pv["model1"].map(model_map).fillna(pv["model1"])
    pv["model2_abbrev"] = pv["model2"].map(model_map).fillna(pv["model2"])

    df = df[df["metric"].isin(metrics)].copy()
    pv = pv[pv["metric"].isin(metrics)].copy()

    datasets_present = sorted(df["dataset"].unique())
    datasets = [d for d in dataset_order if d in datasets_present] or datasets_present

    models = list(model_order)
    palette = dict(zip(models, sns.color_palette("colorblind", n_colors=len(models))))

    fig, axes = plt.subplots(1, len(metrics), figsize=figsize,
                             subplot_kw={"projection": "polar"}, squeeze=False)

    for metric_idx, metric in enumerate(metrics):
        ax = axes[0, metric_idx]
        group_width = 2 * np.pi / len(datasets)
        dataset_gap = group_width * 0.1
        bar_width = (group_width - dataset_gap) / len(models) * 0.95
        dataset_centers = {}

        for group_idx, dataset in enumerate(datasets):
            panel = df[(df["dataset"] == dataset) & (df["metric"] == metric)]
            base_angle = group_idx * group_width + dataset_gap / 2

            for model_idx, model in enumerate(models):
                mdata = panel[panel["model_abbrev"] == model]
                if len(mdata) == 0:
                    val = low = high = 0.0
                else:
                    val, low, high = (float(mdata[c].iloc[0]) for c in ("point", "low", "high"))
                err_low, err_high = val - low, high - val
                theta = base_angle + model_idx * bar_width + bar_width / 2

                ax.bar(theta, val, width=bar_width * 0.98, bottom=0.0, color=palette[model],
                       alpha=0.8, label=model if (metric_idx == 0 and group_idx == 0) else "",
                       edgecolor="white", linewidth=0.5)
                ax.errorbar(theta, val, yerr=[[err_low], [err_high]], fmt="none", ecolor="black",
                            capsize=3, capthick=0.6, elinewidth=0.6)

                if val > 0.1:
                    off = value_label_offsets.get(dataset, {"r_offset": 0.08, "theta_offset": 0.0})
                    ax.text(theta + off["theta_offset"], val + err_high + off["r_offset"],
                            f"{val:.2f}", ha="center", va="bottom", fontsize=fontsize,
                            rotation=0, fontweight="bold")

            dataset_centers[dataset] = base_angle + (len(models) * bar_width) / 2

        for dataset, theta in dataset_centers.items():
            ax.text(theta, 1.25, dataset, ha="center", va="center", fontsize=fontsize,
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.7))

            psubset = pv[(pv["dataset"] == dataset) & (pv["metric"] == metric)]
            table_lines = [
                f"{r['model1_abbrev']}-{r['model2_abbrev']}: {significance_marker(float(r['p_value_bh']))}"
                for _, r in psubset.iterrows()
                if r["model1_abbrev"] in models and r["model2_abbrev"] in models
            ]
            if len(table_lines) > max_table_lines:
                raise ValueError(f"{dataset}: {len(table_lines)} comparisons but only "
                                 f"{max_table_lines} fit in the annotation box")
            if table_lines:
                theta_offset, radius = table_positions.get(metric, {}).get(dataset, (0.15, 1.08))
                ax.text(theta + theta_offset, radius, "\n".join(table_lines), ha="center",
                        va="bottom", fontsize=fontsize - 2, fontfamily="monospace",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray",
                                  alpha=0.95, linewidth=0.8))

        ax.set_ylim(0, 1.28)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=fontsize - 2)
        ax.set_xticks([])
        ax.grid(True, alpha=0.35, axis="y")
        ax.grid(False, axis="x")

        for i in range(len(datasets)):
            ax.plot([i * group_width] * 2, [0, 1.28], "k--", alpha=0.3, linewidth=1.5)

        if "macro_f1" not in metrics:
            ax.set_title(METRIC_NAMES.get(metric, metric), fontsize=fontsize,
                         fontweight="bold", pad=20)

    handles = [plt.Rectangle((0, 0), 1, 1, fc=palette[m], alpha=0.8, label=m) for m in models]
    fig.legend(handles=handles, title="Model", title_fontsize=fontsize, loc="lower center",
               ncol=len(models), bbox_to_anchor=(0.5, -0.03), frameon=True, fontsize=fontsize)

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    save_figure(plt, output_dir, filename)


# Accuracy bars with p-value matrix (standard benchmarks)

def plot_accuracy_bars(results_df: pd.DataFrame, pmatrix: pd.DataFrame | None,
                       model_order, color_order, figsize=(2.3, 2), palette="colorblind",
                       bar_width=0.7, fontsize=7):
    """One bar per model (point + CI); letter p-value matrix; legend panel below."""
    df = results_df[(results_df["metric"] == "accuracy") & results_df["model"].isin(model_order)].copy()
    df["model"] = pd.Categorical(df["model"], categories=model_order, ordered=True)
    df = df.sort_values("model").set_index("model").reindex(model_order).reset_index()
    yerr = np.array([df["point"] - df["low"], df["high"] - df["point"]])

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 1, height_ratios=[20, 1], hspace=0.3)
    ax = fig.add_subplot(gs[0])
    ax_legend = fig.add_subplot(gs[1])

    color_map = dict(zip(color_order, sns.color_palette(palette, n_colors=len(color_order))))
    colors = [color_map[m] for m in model_order]
    hatch_patterns = ["///", "\\\\\\", "|||", "---", "+++", "xxx", "ooo"]
    hatches = [hatch_patterns[i % len(hatch_patterns)] for i in range(len(model_order))]
    x_pos = np.arange(len(model_order))

    bars = [ax.bar(x_pos[i], df["point"].iloc[i], width=bar_width, color=colors[i],
                   edgecolor="black", linewidth=0.5, alpha=0.85, hatch=hatches[i])[0]
            for i in range(len(model_order))]
    ax.errorbar(x_pos, df["point"], yerr=yerr, fmt="none", ecolor="black",
                capsize=5, capthick=1, linewidth=1)

    for bar, point_val, high_val in zip(bars, df["point"], df["high"]):
        if pd.notna(point_val) and pd.notna(high_val):
            ax.text(bar.get_x() + bar.get_width() / 2.0, high_val + 0.01, f"{point_val:.2f}",
                    ha="center", va="bottom", fontsize=fontsize, zorder=10)

    ax.set_ylabel("Macro-averaged accuracy", fontsize=fontsize)
    ax.tick_params(axis="y", labelsize=fontsize)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([])
    ax.grid(axis="y", alpha=0.5, linestyle="--")
    ax.set_axisbelow(True)
    ax.set_ylim(0, 1.4)
    ax.set_yticks(np.arange(0, 1.1, 0.2))

    if pmatrix is not None:
        annotate_pmatrix(ax, pmatrix, xy=(0.985, 0.98), title="p-values               ",
                         text_fontsize=4)
        labels = [f"{m} ({pmatrix.index[i]})" for i, m in enumerate(model_order)]
    else:
        labels = list(model_order)

    ax_legend.axis("off")
    handles = [Rectangle((0, 0), 1, 1, facecolor=colors[i], edgecolor="black", linewidth=0.5,
                         hatch=hatches[i], alpha=0.85) for i in range(len(model_order))]
    ax_legend.legend(handles, labels, loc="center", ncol=max(1, int(np.ceil(len(model_order) / 3))),
                     frameon=False, fontsize=fontsize - 2, handlelength=1.5, handleheight=1.5,
                     bbox_to_anchor=(0.5, 0))
    return fig, [ax, ax_legend]


# Half-violin + box distributions (output length, entropy)

def plot_half_violins(values_by_model: dict[str, np.ndarray], model_order, pmatrix: pd.DataFrame,
                      ylabel: str, figsize=(2.3, 2), fontsize=7, palette="colorblind",
                      linewidth=0.5, box_alpha=0.85, ylim=None, log10_ticks=False,
                      show_legend=True):
    """
    Right-half violin + box per model, letter p-value matrix, legend panel below.
    With `log10_ticks`, values are log10 data and `ylim` is given in original units;
    ticks are labelled 10, 100, 1,000, ...
    """
    color_map = dict(zip(model_order, sns.color_palette(palette, n_colors=len(model_order))))
    letters = {m: chr(65 + i) for i, m in enumerate(model_order)}
    hatch_patterns = ["///", "|||", "---", "+++", "xxx", "ooo"]
    hatch_map = {m: hatch_patterns[i % len(hatch_patterns)] for i, m in enumerate(model_order)}

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 1, height_ratios=[20, 1], hspace=0.3)
    ax = fig.add_subplot(gs[0])
    ax_legend = fig.add_subplot(gs[1])
    ax_legend.axis("off")

    violin_width, box_width = 0.6, 0.3
    legend_handles = []

    for i, model in enumerate(model_order):
        data = values_by_model[model]
        color, hatch = color_map[model], hatch_map[model]

        parts = ax.violinplot([data], positions=[i], widths=violin_width,
                              showmeans=False, showextrema=False, showmedians=False)
        for pc in parts["bodies"]:
            pc.set_facecolor(color)
            pc.set_alpha(0.85)
            pc.set_edgecolor("black")
            pc.set_linewidth(linewidth)
            pc.set_hatch(hatch)
            verts = pc.get_paths()[0].vertices  # keep right half only
            verts[:, 0] = np.clip(verts[:, 0], np.mean(verts[:, 0]), np.inf)

        ax.boxplot([data], positions=[i - violin_width / 2 - box_width / 2 + 0.175],
                   widths=box_width, patch_artist=True, showfliers=False,
                   medianprops=dict(color="black", linewidth=linewidth),
                   boxprops=dict(facecolor=color, edgecolor="black", alpha=box_alpha, linewidth=linewidth),
                   whiskerprops=dict(color="black", linewidth=linewidth),
                   capprops=dict(color="black", linewidth=linewidth))

        legend_handles.append(Patch(facecolor=color, edgecolor="black", alpha=0.85,
                                    linewidth=linewidth, hatch=hatch,
                                    label=f"{model} ({letters[model]})"))

    annotate_pmatrix(ax, pmatrix, xy=(0.995, 0.98), title="p-values               ",
                     text_fontsize=fontsize - 2)

    ax.set_xticks(np.arange(len(model_order)))
    ax.set_xticklabels([])  # legend carries labels
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.grid(axis="y", alpha=1, linestyle="-")
    ax.set_xlim(-0.5, len(model_order) - 0.65)
    ax.tick_params(axis="both", labelsize=fontsize)

    if ylim is not None:
        if log10_ticks:
            ax.set_ylim(np.log10(ylim[0]), np.log10(ylim[1]))
            ticks = np.arange(np.ceil(np.log10(ylim[0])), np.floor(np.log10(ylim[1])) + 1)
            ax.set_yticks(ticks)
            ax.set_yticklabels([rf"$\mathdefault{{10^{{{int(t)}}}}}$" for t in ticks])
        else:
            ax.set_ylim(*ylim)

    if show_legend:
        ax_legend.legend(handles=legend_handles, loc="center", ncol=int(np.ceil(len(model_order) / 2)),
                         frameon=False, fontsize=fontsize - 2, handlelength=1.5, handleheight=1.5,
                         bbox_to_anchor=(0.5, 0))

    plt.tight_layout()
    return fig, [ax, ax_legend]


# Biomarker forest plot (NACC on top, external cohort below)

FOREST_MODEL_COLORS = {"Q3B": "#0173B2", "LUNAR": "#DE8F05", "Q7B": "#029E73"}
FOREST_MODEL_MARKERS = {"Q3B": "o", "LUNAR": "s", "Q7B": "^"}


def _forest_panel(ax, boot_by_biomarker, perm_by_biomarker, metric, model_order, biomarkers,
                  biomarker_labels, model_colors, model_markers, filled, use_fdr, value_col):
    """Draw one panel (row) of the forest plot."""
    n_models = len(model_order)
    group_width = 0.6
    model_spacing = group_width / n_models
    group_gap = 1.0
    all_lo_values = []
    group_tops = {}

    marker_size = 2
    cap_width = 0.03
    ci_linewidth = 0.8
    cap_linewidth = 1
    marker_edgewidth = 1

    for bi, biomarker in enumerate(biomarkers):
        boot_df = boot_by_biomarker.get(biomarker)
        if boot_df is None:
            continue
        plot_data = boot_df[boot_df["metric"] == metric].copy()
        plot_data = plot_data.drop_duplicates(subset=["model", "metric"])

        group_center = bi * group_gap
        tops_for_group = []

        for mi, model in enumerate(model_order):
            data = plot_data[plot_data["model"] == model]
            if len(data) == 0:
                continue

            val = data[value_col].values[0]
            lo = data["low"].values[0]
            hi = data["high"].values[0]
            color = model_colors[model]
            marker = model_markers[model]

            x_pos = group_center + (mi - (n_models - 1) / 2) * model_spacing
            all_lo_values.append(lo)

            # CI line and caps
            ax.plot([x_pos, x_pos], [lo, hi], color=color, linewidth=ci_linewidth,
                    solid_capstyle="round", zorder=2)
            ax.plot([x_pos - cap_width, x_pos + cap_width], [lo, lo],
                    color=color, linewidth=cap_linewidth, zorder=2)
            ax.plot([x_pos - cap_width, x_pos + cap_width], [hi, hi],
                    color=color, linewidth=cap_linewidth, zorder=2)
            # Point estimate
            fc = color if filled else "white"
            ax.plot(x_pos, val, marker=marker, markersize=marker_size, color=fc,
                    markeredgecolor=color, markeredgewidth=marker_edgewidth, zorder=3)
            # Value label
            ax.text(x_pos, hi + 0.02, f"{val:.2f}", ha="center", va="bottom", fontsize=7,
                    color="#444444", zorder=4)

            tops_for_group.append(hi + 0.02 + 0.03)

        group_tops[biomarker] = max(tops_for_group) if tops_for_group else 0.8

        # Significance brackets
        perm_df = perm_by_biomarker.get(biomarker)
        if perm_df is not None and len(perm_df) > 0:
            p_col = "p_value_bh" if use_fdr else "p_value"
            metric_results = perm_df[perm_df["metric"] == metric].copy()

            sig_pairs = []
            seen_pairs = set()
            for _, row in metric_results.iterrows():
                m1, m2 = row["model1"], row["model2"]
                pair = tuple(sorted([m1, m2]))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                if m1 not in model_order or m2 not in model_order:
                    continue

                mi1, mi2 = model_order.index(m1), model_order.index(m2)
                x1 = group_center + (mi1 - (n_models - 1) / 2) * model_spacing
                x2 = group_center + (mi2 - (n_models - 1) / 2) * model_spacing
                if x1 > x2:
                    x1, x2 = x2, x1
                sig_pairs.append((x1, x2, row[p_col]))

            # Narrow brackets first; stack a bracket above any it overlaps
            sig_pairs.sort(key=lambda p: (p[1] - p[0], p[0]))
            pair_levels = []
            for x1, x2, p_val in sig_pairs:
                level = 0
                while any(plevel == level and not (x2 < px1 or x1 > px2)
                          for (px1, px2, _, plevel) in pair_levels):
                    level += 1
                pair_levels.append((x1, x2, p_val, level))

            base_offset = 0.05
            bar_increment = 0.08
            bar_tips = 0.014
            global_max = group_tops[biomarker]

            for x1, x2, p_val, level in pair_levels:
                y_bar = global_max + base_offset + level * bar_increment
                ax.plot([x1, x1, x2, x2], [y_bar - bar_tips, y_bar, y_bar, y_bar - bar_tips],
                        color="#333333", linewidth=0.8, zorder=5)
                ax.text((x1 + x2) / 2, y_bar, significance_marker(p_val), ha="center",
                        va="bottom", fontsize=5, fontweight="bold", color="#333333", zorder=5)

    # Axis formatting
    x_centers = [bi * group_gap for bi in range(len(biomarkers))]
    ax.set_xticks(x_centers)
    ax.set_xticklabels([biomarker_labels.get(b, b) for b in biomarkers], fontsize=7)
    ax.set_xlim(x_centers[0] - 0.5, x_centers[-1] + 0.5)

    y_lo = min(all_lo_values) - 0.05 if all_lo_values else 0.0
    ax.set_ylim(y_lo, None)

    ax.grid(True, axis="y", linestyle=(0, (2, 2)), linewidth=0.6, alpha=0.18, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1)
    ax.spines["bottom"].set_linewidth(1)
    ax.tick_params(axis="both", which="major", labelsize=7, width=1, length=4)


def forest_plot_biomarker(nacc_boot, nacc_perm, ext_boot, ext_perm, metric="macro_f1",
                          biomarker_order=("PET", "CSF", "DAT"),
                          biomarker_labels=None, model_colors=None, model_markers=None,
                          figsize=None, use_fdr=True, value_col="point"):
    """
    Two-panel (stacked) forest plot: NACC on top (filled markers), external cohort
    below (open markers, grey shade).

    nacc_boot, nacc_perm, ext_boot, ext_perm: {biomarker: df}. Bootstrap frames need
    model / metric / `value_col` / low / high; permutation frames need model1 /
    model2 / metric / p_value(_bh). Model names are display names; the models drawn
    and their order are the keys of `model_colors`.
    value_col: column drawn as the point estimate ("point" = statistic on the
    original data).
    """
    import matplotlib.lines as mlines

    biomarker_labels = biomarker_labels or {"PET": "Amyloid PET", "CSF": "Amyloid CSF", "DAT": "DAT"}
    model_colors = model_colors or FOREST_MODEL_COLORS
    model_markers = model_markers or FOREST_MODEL_MARKERS
    model_order = list(model_colors.keys())
    biomarkers = [b for b in biomarker_order if b in nacc_boot or b in ext_boot]

    if figsize is None:
        figsize = (4, 4)

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=figsize, dpi=300, sharex=True)
    metric_name = METRIC_NAMES.get(metric, metric)
    panel_args = dict(metric=metric, model_order=model_order, biomarkers=biomarkers,
                      biomarker_labels=biomarker_labels, model_colors=model_colors,
                      model_markers=model_markers, use_fdr=use_fdr, value_col=value_col)

    # Top panel: NACC (filled markers)
    _forest_panel(ax_top, nacc_boot, nacc_perm, filled=True, **panel_args)
    ax_top.set_ylabel(f"{metric_name} (NACC)", fontsize=7, fontweight="normal")

    # Bottom panel: external cohort (open markers, grey shade)
    _forest_panel(ax_bot, ext_boot, ext_perm, filled=False, **panel_args)
    ax_bot.set_ylabel(f"{metric_name} (External)", fontsize=7, fontweight="normal")
    ax_bot.axhspan(0, 1, transform=ax_bot.transAxes, color="black", alpha=0.035, zorder=-10)

    # Horizontal legend below the bottom panel
    handles = [
        mlines.Line2D([], [], color=model_colors[m], marker=model_markers[m], linestyle="None",
                      markersize=3, label=m, markeredgewidth=1)
        for m in model_order
    ]
    fig.legend(handles=handles, fontsize=7, loc="lower center", bbox_to_anchor=(0.5, -0.02),
               ncol=len(model_order), frameon=True, handletextpad=0.4, columnspacing=1.5)

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    return fig, (ax_top, ax_bot)