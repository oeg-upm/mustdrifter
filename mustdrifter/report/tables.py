from itertools import product
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def _new_table(periods):
    return pd.DataFrame(index=periods, columns=periods, dtype=float)


def _finalize_table(table, fill_diagonal):
    if fill_diagonal is not None:
        for period in table.index.intersection(table.columns):
            row = table.loc[period].drop(labels=period, errors="ignore")
            col = table[period].drop(labels=period, errors="ignore")

            if not (row.isna().all() and col.isna().all()):
                table.loc[period, period] = fill_diagonal

    table = table.dropna(axis=0, how="all").dropna(axis=1, how="all")
    return table


def _normalize_table(table, method=None):
    if method is None:
        return table.copy()

    values = table.to_numpy(dtype=float).copy()

    # ignorar diagonal y NaN al calcular la normalización
    if table.shape[0] == table.shape[1]:
        np.fill_diagonal(values, np.nan)

    valid_mask = ~np.isnan(values)
    valid_values = values[valid_mask]

    if valid_values.size == 0:
        return table.copy()

    normalized = values.copy()

    if method == "minmax":
        min_val = np.nanmin(valid_values)
        max_val = np.nanmax(valid_values)
        denom = max_val - min_val

        if denom == 0:
            normalized[valid_mask] = 0.0
        else:
            normalized[valid_mask] = (valid_values - min_val) / denom

    elif method == "zscore":
        mean_val = np.nanmean(valid_values)
        std_val = np.nanstd(valid_values)

        if std_val == 0:
            normalized[valid_mask] = 0.0
        else:
            normalized[valid_mask] = (valid_values - mean_val) / std_val

    else:
        raise ValueError("normalize must be one of: None, 'minmax', 'zscore'")

    return pd.DataFrame(normalized, index=table.index, columns=table.columns)


def _aggregate_tables(
    tables_to_aggregate,
    periods,
    aggregation_func,
    absolute_values,
    min_values,
    fill_diagonal,
    normalize=None,
):
    aggregated = _new_table(periods)

    prepared_tables = []

    for table in tables_to_aggregate:
        prepared_table = table.copy()

        if absolute_values:
            prepared_table = prepared_table.abs()

        prepared_table = _normalize_table(prepared_table, method=normalize)
        prepared_tables.append(prepared_table)

    for reference_period, test_period in product(periods, periods):
        if reference_period == test_period:
            continue

        values = []

        for table in prepared_tables:
            if reference_period not in table.index or test_period not in table.columns:
                continue

            value = table.loc[reference_period, test_period]

            if pd.isna(value):
                continue

            values.append(float(value))

        if len(values) >= min_values:
            aggregated.loc[reference_period, test_period] = aggregation_func(values)

    return _finalize_table(aggregated, fill_diagonal)


def get_drift_tables(
    dimension_drift_loaders,
    periods=None,
    dimensions=None,
    metrics=None,
    semantic_score_key="magnitude",
    syntactic_content_score_key="magnitude",
    syntactic_style_score_key="magnitude",
    lexical_score_key="magnitude",
    thematic_score_key="magnitude",
    fill_diagonal=np.nan,
    sort_periods=True,
    aggregate_by=None,
    aggregation_func=np.nanmean,
    absolute_values=True,
    min_values=1,
    normalize="minmax",
    **kwargs,
):
    score_keys_mapping = {
        "semantic": semantic_score_key,
        "syntactic_content": syntactic_content_score_key,
        "syntactic_style": syntactic_style_score_key,
        "lexical": lexical_score_key,
        "thematic": thematic_score_key,
    }

    if dimensions is None:
        dimensions = list(dimension_drift_loaders.keys())

    invalid_dimensions = [dim for dim in dimensions if dim not in dimension_drift_loaders]
    if invalid_dimensions:
        raise ValueError(
            f"Invalid dimensions: {invalid_dimensions}. "
            f"Allowed values are: {list(dimension_drift_loaders.keys())}"
        )

    if metrics is None:
        metrics = ["mmd", "cos", "ks", "kl", "js", "log"]

    if aggregate_by not in [None, "metric", "dimension"]:
        raise ValueError(
            "aggregate_by must be one of: None, 'metric', 'dimension'"
        )

    if normalize not in [None, "minmax", "zscore"]:
        raise ValueError(
            "normalize must be one of: None, 'minmax', 'zscore'"
        )

    if sort_periods:
        periods = sorted(periods, key=lambda x: int(x))

    base_tables = {
        dimension: {
            metric: _new_table(periods)
            for metric in metrics
        }
        for dimension in dimensions
    }

    for dimension in dimensions:
        load_method = dimension_drift_loaders[dimension]
        score_key = score_keys_mapping[dimension]

        for metric in metrics:
            logger.info(
                f"Building meta table for dimension '{dimension}' and metric '{metric}'"
            )

            table = base_tables[dimension][metric]

            for reference_period, test_period in product(periods, periods):
                if reference_period == test_period:
                    continue

                try:
                    drift_data = load_method(
                        reference_period=reference_period,
                        test_period=test_period,
                        metric=metric,
                    )

                    if drift_data is None:
                        continue

                    score = drift_data.get(score_key)
                    if score is None:
                        logger.debug(
                            f"Score key '{score_key}' not found for "
                            f"{dimension} drift ({reference_period} -> {test_period}, metric={metric})"
                        )
                        continue

                    score = float(score)

                    if np.isnan(score):
                        continue

                    if absolute_values:
                        score = abs(score)

                    table.loc[reference_period, test_period] = score

                except Exception as exc:
                    logger.debug(
                        f"Could not load {dimension} drift "
                        f"({reference_period} -> {test_period}, metric={metric}): {exc}"
                    )

            table = _finalize_table(table, fill_diagonal)

            if table.isna().all().all():
                logger.info(
                    f"Removing empty table for dimension '{dimension}' and metric '{metric}'"
                )
                continue

            base_tables[dimension][metric] = table

    cleaned_base_tables = {}

    for dimension, metric_tables in base_tables.items():
        cleaned_metric_tables = {}

        for metric, table in metric_tables.items():
            if table.isna().all().all():
                continue

            cleaned_metric_tables[metric] = table

        if cleaned_metric_tables:
            cleaned_base_tables[dimension] = cleaned_metric_tables

    if aggregate_by is None:
        return cleaned_base_tables

    if aggregate_by == "metric":
        aggregated_by_dimension = {}

        for dimension, metric_tables in cleaned_base_tables.items():
            tables_to_aggregate = list(metric_tables.values())

            if not tables_to_aggregate:
                continue

            aggregated_table = _aggregate_tables(
                tables_to_aggregate=tables_to_aggregate,
                periods=periods,
                aggregation_func=aggregation_func,
                absolute_values=absolute_values,
                min_values=min_values,
                fill_diagonal=fill_diagonal,
                normalize=normalize,
            )

            if aggregated_table.isna().all().all():
                continue

            aggregated_by_dimension[dimension] = aggregated_table

        return aggregated_by_dimension

    aggregated_all = _aggregate_tables(
        tables_to_aggregate=[
            table
            for metric_tables in cleaned_base_tables.values()
            for table in metric_tables.values()
        ],
        periods=periods,
        aggregation_func=aggregation_func,
        absolute_values=absolute_values,
        min_values=min_values,
        fill_diagonal=fill_diagonal,
        normalize=normalize,
    )

    return aggregated_all