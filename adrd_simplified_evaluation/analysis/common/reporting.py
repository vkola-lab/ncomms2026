"""Significance markers, p-value matrix annotations and LaTeX tables."""

import numpy as np
import pandas as pd


def significance_marker(p_value) -> str:
    if isinstance(p_value, str):
        return p_value
    if pd.isna(p_value):
        return ""
    if p_value < 0.0001:
        return "****"
    elif p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    elif p_value <= 1.0:
        return "ns"
    return str(p_value)


# P-value matrix (standard benchmarks, output length, entropy)

def letter_pmatrix(res_df: pd.DataFrame, model_order, model_to_letter) -> pd.DataFrame:
    """Symmetric letter-labelled matrix of BH-adjusted p-values from a results table."""
    letters = [model_to_letter[m] for m in model_order]
    mat = pd.DataFrame(np.nan, index=letters, columns=letters)
    for _, row in res_df.iterrows():
        m1, m2 = row["model1"], row["model2"]
        if m1 in model_to_letter and m2 in model_to_letter:
            l1, l2 = model_to_letter[m1], model_to_letter[m2]
            mat.loc[l1, l2] = row["p_value_bh"]
            mat.loc[l2, l1] = row["p_value_bh"]
    return mat


def get_annotate_matrix(matrix_content: pd.DataFrame) -> list[str]:
    """Upper-triangular text rendering of a p-value matrix."""
    n = len(matrix_content)
    row_labels = matrix_content.index.tolist()
    col_labels = matrix_content.columns.tolist()[1:]

    lines = ["  ".join(["  "] + ["{: <5}".format(label) for label in col_labels])]
    for i in range(n - 1):
        row_data = ["{: <5}".format(significance_marker(matrix_content.iloc[i, j]))
                    for j in range(i + 1, n)]
        lines.append("  ".join(["     "] * i + row_data) + "  " + row_labels[i])
    return lines


def annotate_pmatrix(ax, matrix_content: pd.DataFrame, xy=(0.995, 0.98),
                     title: str = "p-values", text_fontsize: float = 4) -> None:
    lines = get_annotate_matrix(matrix_content)
    lines.insert(0, title)
    ax.annotate(
        "\n".join(lines),
        xy=xy,
        xycoords="axes fraction",
        ha="right",
        va="top",
        bbox=dict(boxstyle="round,pad=0.3", edgecolor="black",
                  facecolor=(1, 1, 1, 0.0), lw=0.5),
        fontsize=text_fontsize,
        family="monospace",
    )


# LaTeX tables

def _best_second(stats: pd.DataFrame, metrics, group_cols):
    best, second = set(), set()
    for m in metrics:
        col = f"{m}_point"
        ranks = stats.groupby(group_cols, observed=False)[col].rank(method="first", ascending=False)
        best |= {(i, m) for i in stats.index[(ranks == 1) & stats[col].notna()]}
        second |= {(i, m) for i in stats.index[(ranks == 2) & stats[col].notna()]}
    return best, second


def _fmt_cell(row, m, i, best, second) -> str:
    s = f"{row[f'{m}_point']:.3f} [{row[f'{m}_low']:.3f}, {row[f'{m}_high']:.3f}]"
    if (i, m) in best:
        s = f"\\textbf{{{s}}}"
    if (i, m) in second:
        s = f"\\underline{{{s}}}"
    return s


def _pivot(df: pd.DataFrame, index_cols) -> pd.DataFrame:
    stats = df.pivot_table(index=index_cols, columns="metric",
                           values=["point", "low", "high"]).reset_index()
    stats.columns = [f"{c[1]}_{c[0]}" if c[1] else c[0] for c in stats.columns]
    return stats


def latex_table_per_class(all_metric: pd.DataFrame, model_map, class_map, class_order,
                          by_dataset: bool = True) -> str:
    """Per-class precision / recall / F1 with CIs; best bold, second underlined per class."""
    df = all_metric.copy()
    df["model"] = df["model"].map(model_map).fillna(df["model"])
    df["class"] = df["class"].map(class_map).fillna(df["class"])
    
    class_order = [c for c in class_order if c in list(class_map.values())]

    index_cols = (["dataset"] if by_dataset else []) + ["class", "model"]
    stats = _pivot(df, index_cols)
    stats["class"] = pd.Categorical(stats["class"], categories=class_order, ordered=True)
    stats["model"] = pd.Categorical(stats["model"], categories=list(model_map.values()), ordered=True)
    stats = stats.sort_values(index_cols).reset_index(drop=True)

    metrics = ["precision", "recall", "f1"]
    rank_cols = (["dataset"] if by_dataset else []) + ["class"]
    best, second = _best_second(stats, metrics, rank_cols)

    headers = (["Dataset"] if by_dataset else []) + ["Class", "Model", "Precision", "Recall", "F1-score"]
    colspec = ("l" if by_dataset else "") + "llccc"
    lines = ["\\begin{table}[ht]", "\\centering", "\\small",
             f"\\begin{{tabular}}{{{colspec}}}", "\\hline",
             " & ".join(headers) + " \\\\", "\\hline"]

    prev_ds, prev_cl = None, None
    for i, row in stats.iterrows():
        if prev_cl is not None and row["class"] != prev_cl:
            lines.append("\\hline")
        parts = []
        if by_dataset:
            parts.append(row["dataset"] if row["dataset"] != prev_ds else "")
        parts += [row["class"] if row["class"] != prev_cl else "", str(row["model"])]
        cells = [_fmt_cell(row, m, i, best, second) for m in metrics]
        lines.append(" & ".join(str(p) for p in parts) + " & " + " & ".join(cells) + " \\\\")
        prev_ds = row["dataset"] if by_dataset else None
        prev_cl = row["class"]

    lines += ["\\hline", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def latex_table_macro(all_metric: pd.DataFrame, model_map, dataset_map=None) -> str:
    """Macro precision / recall / F1 with CIs per dataset."""
    df = all_metric.copy()
    df["model"] = df["model"].map(model_map).fillna(df["model"])
    if dataset_map is not None:
        df["dataset"] = df["dataset"].map(dataset_map).fillna(df["dataset"])

    stats = _pivot(df, ["dataset", "model"])
    stats["model"] = pd.Categorical(stats["model"], categories=list(model_map.values()), ordered=True)
    stats = stats.sort_values(["dataset", "model"]).reset_index(drop=True)

    metrics = ["macro_precision", "macro_recall", "macro_f1"]
    best, second = _best_second(stats, metrics, ["dataset"])

    headers = ["Dataset", "Model", "Macro Precision", "Macro Recall", "Macro F1"]
    lines = ["\\begin{table}[ht]", "\\centering", "\\small",
             "\\begin{tabular}{llccc}", "\\hline",
             " & ".join(headers) + " \\\\", "\\hline"]

    prev_ds = None
    for i, row in stats.iterrows():
        if prev_ds is not None and row["dataset"] != prev_ds:
            lines.append("\\hline")
        ds_disp = row["dataset"] if row["dataset"] != prev_ds else ""
        cells = [_fmt_cell(row, m, i, best, second) for m in metrics]
        lines.append(f"{ds_disp} & {row['model']} & " + " & ".join(cells) + " \\\\")
        prev_ds = row["dataset"]

    lines += ["\\hline", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def save_text(text: str, path: str) -> None:
    import os
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w") as f:
        f.write(text)
    print(f"Saved {path}")
