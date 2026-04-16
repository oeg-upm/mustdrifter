import logging

logger = logging.getLogger(__name__)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

        fig = _plot_heatmap(
            table=table,
            title="Meta-Magnitude",
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

            fig = _plot_heatmap(
                table=labeled_table,
                title=titles_dimension.get(dimension, str(dimension)),
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