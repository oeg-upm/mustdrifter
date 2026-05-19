import logging

logger = logging.getLogger(__name__)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


import numpy as np
import matplotlib.cm as cm
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize


def _apply_period_labels(table, period_labels, logger=None):
    table = table.copy()

    if period_labels is None:
        return table

    if not period_labels:
        if logger:
            logger.warning("The provided period_labels dictionary is empty.")
        return None

    sample_index = next(iter(table.index), None)

    if sample_index is not None and isinstance(sample_index, str):
        normalized_period_labels = {str(k): v for k, v in period_labels.items()}
    else:
        normalized_period_labels = dict(period_labels)

    ordered_keys = [key for key in normalized_period_labels if key in table.index]

    table = table.loc[
        table.index.intersection(ordered_keys),
        table.columns.intersection(ordered_keys),
    ]

    table = table.loc[ordered_keys, ordered_keys]

    if table.empty:
        return None

    table = table.rename(
        index=normalized_period_labels,
        columns=normalized_period_labels,
    )

    return table

def _plot_heatmap(
    table,
    title,
    export=False,
    filename=None,
    figsize=(8, 6),
    cmap="RdYlGn_r",
    fmt=".3f",
    vmin= None,
    vmax= None,
):
    if table is None or table.empty:
        return None

    abs_table = table.abs()

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(title, fontsize=16, weight="bold")

    if vmin is None: vmin= table.min().min()
    if vmax is None: vmax= table.max().max()

    sns.heatmap(abs_table, 
                annot=True,
                cmap=cmap,
                fmt=fmt,
                ax=ax,
                vmin=vmin,
                vmax=vmax
                )

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    fig.tight_layout()

    if export and filename is not None:
        fig.savefig(
            filename,
            format="svg",
            bbox_inches="tight",
            pad_inches=0,
            transparent=True,
        )

    plt.close(fig)
    return fig

def generate_magnitude_heatmaps(
    meta_tables,
    export=True,
    period_labels=None,
    base_filename= "",
    figsize=(8, 6),
    cmap="RdYlGn_r",
    fmt=".3f",
    title="",
    **kwargs
):
    titles_dimension = {
        "semantic": "Semantic Drift",
        "syntactic_content": "Syntactic Content Drift",
        "syntactic_style": "Syntactic Style Drift",
        "lexical": "Lexical Drift",
        "thematic": "Thematic Drift",
    }

    tiles_metrics = {
        "mmd": "Maximum Mean Discrepancy",
        "cos": "Cosine Drift",
        "kl": "Kullback–Leibler Divergence",
        "js": "Jensen–Shannon Divergence",
        "log": "Log-Likelihood",
        "ks": "Kolmogorov–Smirnov Test",
    }

    generated_figures = {}

    # Caso 1: una sola tabla (aggregate_by="dimension")
    if isinstance(meta_tables, pd.DataFrame):
        table = _apply_period_labels(meta_tables, period_labels, logger)
        if title == "":
            title = "Meta-Magnitude"
        fig = _plot_heatmap(
            table=table,
            title=title,
            export=export,
            filename=f"{base_filename}.svg",
            figsize=figsize,
            cmap=cmap,
            fmt=fmt,
            vmin=0,
            vmax=1
        )

        return fig

    if not isinstance(meta_tables, dict) or not meta_tables:
        logger.warning("meta_tables is empty or has an invalid format.")
        return None

    first_value = next(iter(meta_tables.values()))

    # Caso 2: {dimension: table}  (aggregate_by="metric")
    if isinstance(first_value, pd.DataFrame):
        for dimension, table in meta_tables.items():
            labeled_table = _apply_period_labels(table, period_labels, logger)

            if labeled_table is None or labeled_table.empty:
                logger.warning(f"Empty table for dimension='{dimension}'.")
                continue
            
            if title == "":
                title = titles_dimension.get(dimension, str(dimension))

            fig = _plot_heatmap(
                table=labeled_table,
                title=title,
                export=export,
                filename=f"{base_filename}_{dimension}.svg",
                figsize=figsize,
                cmap=cmap,
                fmt=fmt,
                vmin=0,
                vmax=1
            )

            if fig is not None:
                generated_figures[dimension] = fig

        return generated_figures if generated_figures else None

    # Caso 3: {dimension: {metric: table}}  (aggregate_by=None)
    if isinstance(first_value, dict):
        for dimension, metric_tables in meta_tables.items():
            generated_figures[dimension] = {}

            for metric, table in metric_tables.items():
                labeled_table = _apply_period_labels(table, period_labels, logger)

                if labeled_table is None or labeled_table.empty:
                    logger.warning(
                        f"Empty table for dimension='{dimension}', metric='{metric}'."
                    )
                    continue
                if title == "":
                    title = tiles_metrics.get(metric, str(metric))
                
                fig = _plot_heatmap(
                    table=labeled_table,
                    title=tiles_metrics.get(metric, str(metric)),
                    export=export,
                    filename=f"{base_filename}_{dimension}_{metric}.svg",
                    figsize=figsize,
                    cmap=cmap,
                    fmt=fmt,
                )

                if fig is not None:
                    generated_figures[dimension][metric] = fig

            if not generated_figures[dimension]:
                del generated_figures[dimension]

        return generated_figures if generated_figures else None

    logger.warning("Unsupported meta_tables structure.")
    return None


def plot_aggregated_dimension_values_heatmap(
    tables,
    period_labels=None,
    dimensions_order=None,
    dimension_labels=None,
    dimension_short_names=None,
    title="Multidimensional discourse drift by dimension",
    cmap_name="RdYlGn_r",
    figsize=(26,8),
    filename=None,
    show_values=True,
    export= False,
    **kwargs
):
    if dimensions_order is None:
        dimensions_order = list(tables.keys())

    if dimension_labels is None:
        dimension_labels = {
            "semantic": "Semantic",
            "syntactic_content": "Syntactic Content",
            "syntactic_style": "Syntactic Style",
            "lexical": "Lexical",
            "thematic": "Thematic",
        }

    if dimension_short_names is None:
        dimension_short_names = {
            "semantic": "S",
            "syntactic_content": "SC",
            "syntactic_style": "SS",
            "lexical": "L",
            "thematic": "T",
        }

    first_table = tables[dimensions_order[0]]
    periods = list(first_table.index)
    n_periods = len(periods)

    labels = [
        period_labels.get(period, str(period)) if period_labels else str(period)
        for period in periods
    ]

    cmap = cm.get_cmap(cmap_name)
    norm = Normalize(vmin=0, vmax=1)

    fig = plt.figure(figsize=figsize)

    ax = fig.add_axes([0.04, 0.16, 0.80, 0.68])
    cax = fig.add_axes([0.86, 0.53, 0.015, 0.28])

    ax.set_xlim(0, n_periods)
    ax.set_ylim(0, n_periods)
    ax.invert_yaxis()
    ax.set_aspect("auto")

    inner_positions = {
        0: (0, 0),
        1: (0, 1),
        2: (0, 2),
        3: (0, 3),
        4: (0, 4),
    }

    inner_cols = 1
    inner_rows = len(dimensions_order)

    cell_padding_x = 0.06
    cell_padding_y = 0.06
    inner_gap_y = 0.002
    inner_gap_x = 0.006

    inner_width = 1 - 2 * cell_padding_x
    slot_height = (1 - 2 * cell_padding_y - (inner_rows - 1) * inner_gap_y) / inner_rows
    inner_height = slot_height
    
    def format_value(value):
        return f"{value:.2f}".replace("0.", ".")

    for row_idx, reference_period in enumerate(periods):
        for col_idx, test_period in enumerate(periods):
            ax.add_patch(
                Rectangle(
                    (col_idx, row_idx),
                    1,
                    1,
                    facecolor="white",
                    edgecolor="0.65",
                    linewidth=0.7,
                )
            )

            if reference_period == test_period:
                ax.text(
                    col_idx + 0.5,
                    row_idx + 0.5,
                    "—",
                    ha="center",
                    va="center",
                    fontsize=18,
                )
                continue

            for dim_idx, dimension in enumerate(dimensions_order):
                value = tables[dimension].loc[reference_period, test_period]

                x = col_idx + cell_padding_x

                slot_y = (
                    row_idx
                    + cell_padding_y
                    + dim_idx * (slot_height + inner_gap_y)
                )

                y = slot_y + (slot_height - inner_height) / 2

                color = "lightgray" if np.isnan(value) else cmap(norm(value))

                ax.add_patch(
                    Rectangle(
                        (x, y),
                        inner_width,
                        inner_height,
                        facecolor=color,
                        edgecolor="0.85",
                        linewidth=0.35,
                    )
                )

                short_name = dimension_short_names.get(dimension, dimension)

                if np.isnan(value):
                    combined_text = short_name
                    text_color = "black"
                else:
                    combined_text = (
                        f"{short_name}: {format_value(value)}"
                        if show_values
                        else short_name
                    )
                    text_color = "white" if (value >= 0.82 or value <= 0.12) else "black"

                ax.text(
                    x + inner_width / 2,
                    y + inner_height / 2,
                    combined_text,
                    ha="center",
                    va="center",
                    fontsize=18,
                    fontweight="bold",
                    color=text_color,
                )

    ax.set_xticks(np.arange(n_periods) + 0.5)
    ax.set_yticks(np.arange(n_periods) + 0.5)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(labels, fontsize=18)

    ax.tick_params(length=0)
    ax.set_title(title, fontsize=26, fontweight="bold", pad=18)

    colorbar = fig.colorbar(
        cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=cax,
    )
    colorbar.set_label("Normalized drift score", rotation=270, labelpad=18)

    legend_lines = [
        f"{dimension_short_names.get(dim, dim)}  {dimension_labels.get(dim, dim)}"
        for dim in dimensions_order
    ]

    ax.text(
        1.01,
        0.47,
        "Dimensions\n\n" + "\n".join(legend_lines),
        transform=ax.transAxes,
        va="top",
        fontsize=18,
        linespacing=1.5,
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="white",
            edgecolor="0.75",
        ),
    )

    if export and filename is not None:
        plt.savefig(
            filename,
            format="svg",
            bbox_inches="tight",
            pad_inches=0,
            transparent=True,
        )

    return fig, ax