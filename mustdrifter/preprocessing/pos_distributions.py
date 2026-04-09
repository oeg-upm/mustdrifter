import pandas as pd
import numpy as np
from itertools import permutations
import re
import logging
logger = logging.getLogger(__name__)


def _rule_exists_in_document(
    pos_sequence: list[str],
    rule_tags: tuple[str, ...]
) -> int:
    current_pos = 0

    for rule_tag in rule_tags:
        found = False

        while current_pos < len(pos_sequence):
            if pos_sequence[current_pos] == rule_tag:
                found = True
                current_pos += 1
                break
            current_pos += 1

        if not found:
            return 0

    return 1

def get_syntactic_content_distribution(
    pos_annotations,
    docs_df,
    allowed_upos=["NOUN", "PRON", "VERB", "ADV", "ADJ", "DET"],
    **kwargs
):
    """
    Compute the probability distribution of syntactic content rules by period.

    A rule is defined as any permutation of the UPOS tags in `allowed_upos`.
    For each document, a rule is considered present if it appears as an
    ordered non-contiguous subsequence in the document's UPOS sequence.

    Parameters
    ----------
    pos_annotations : pd.DataFrame
        DataFrame of Stanza POS annotations with at least:
        - doc_id
        - id
        - upos

    docs_df : pd.DataFrame
        DataFrame with at least:
        - doc_id
        - period_id

    allowed_upos : list[str] | None, optional
        UPOS tags used to build the rules. If None, defaults to:
        ["NOUN", "PRON", "VERB", "ADV", "ADJ", "DET"]

    Returns
    -------
    pd.DataFrame
        DataFrame with:
        - first column: period_id
        - one column per rule, sorted alphabetically
        - values: probability distribution over rules within each period
    """
    logger.debug("Calculating syntactic content dimension...")

    df = pos_annotations.merge(
        docs_df[["doc_id", "period_id"]],
        on="doc_id",
        how="left"
    )

    df = df[df["upos"].isin(allowed_upos)].copy()
    df = df.dropna(subset=["period_id"]).copy()
    df = df.sort_values(["doc_id", "id"])

    rules = sorted(["+".join(rule) for rule in permutations(allowed_upos)])
    rules_split = {rule: tuple(rule.split("+")) for rule in rules}

    if df.empty:
        return pd.DataFrame(columns=["period_id"] + rules)

    doc_period = (
        df[["doc_id", "period_id"]]
        .drop_duplicates(subset=["doc_id"])
        .set_index("doc_id")["period_id"]
    )

    doc_pos = df.groupby("doc_id")["upos"].apply(list)
    target_tag_set = set(allowed_upos)

    logger.debug("Built document-level UPOS sequences and period mapping.")
    
    rows = []

    for doc_id, pos_seq in doc_pos.items():
        row = {
            "doc_id": doc_id,
            "period_id": doc_period.loc[doc_id]
        }

        if not target_tag_set.issubset(set(pos_seq)):
            for rule in rules:
                row[rule] = 0
            rows.append(row)
            continue

        for rule, rule_tags in rules_split.items():
            row[rule] = _rule_exists_in_document(pos_seq, rule_tags)

        rows.append(row)

    df_doc_rules = pd.DataFrame(rows)

    df_period_counts = (
        df_doc_rules
        .drop(columns=["doc_id"])
        .groupby("period_id", as_index=False)
        .sum()
    )
    logger.debug("Calculated document-level rule presence and aggregated counts by period.")
    
    rule_totals = df_period_counts[rules].sum(axis=1)

    df_period_probs = df_period_counts.copy()
    df_period_probs[rules] = (
        df_period_probs[rules]
        .div(rule_totals.replace(0, pd.NA), axis=0)
        .fillna(0.0)
    )

    df_period_probs = df_period_probs[["period_id"] + rules]

    return df_period_probs

def get_syntactic_style_distribution(
    pos_annotations,
    docs_df,
    context_size=4,
    min_count=4,
    **kwargs
):
    """
    Compute conditional probabilities of UPOS tags:
        P(next_upos | previous `context_size` UPOS tags)

    Contexts with fewer than `min_count` global occurrences
    are removed.

    Returns
    -------
    pd.DataFrame
        Wide-format DataFrame with:
        - one row per period_id
        - one column per conditional probability P(next_upos|context)
    """
    df = pos_annotations.merge(
        docs_df[["doc_id", "period_id"]],
        on="doc_id",
        how="left",
    )

    df = df[df["upos"].notna()].copy()

    if df.empty:
        return pd.DataFrame(columns=["period_id"])

    df = df.sort_values(["doc_id", "id"])

    doc_period = (
        df[["doc_id", "period_id"]]
        .drop_duplicates(subset=["doc_id"])
        .set_index("doc_id")["period_id"]
    )

    doc_pos = df.groupby("doc_id", observed=True)["upos"].apply(list)

    rows = []

    for doc_id, pos_seq in doc_pos.items():
        if len(pos_seq) <= context_size:
            continue

        period_id = doc_period.loc[doc_id]

        for i in range(context_size, len(pos_seq)):
            context = "+".join(pos_seq[i - context_size:i])
            next_upos = pos_seq[i]

            rows.append(
                {
                    "period_id": period_id,
                    "context": context,
                    "next_upos": next_upos,
                }
            )

    if not rows:
        return pd.DataFrame(columns=["period_id"])

    df_transitions = pd.DataFrame(rows)

    global_context_counts = (
        df_transitions
        .groupby("context", observed=True)
        .size()
        .reset_index(name="global_context_count")
    )

    valid_contexts = global_context_counts.loc[
        global_context_counts["global_context_count"] >= min_count,
        "context",
    ]

    df_transitions = df_transitions[
        df_transitions["context"].isin(valid_contexts)
    ].copy()

    if df_transitions.empty:
        return pd.DataFrame(columns=["period_id"])

    counts = (
        df_transitions
        .groupby(["period_id", "context", "next_upos"], observed=True)
        .size()
        .reset_index(name="upos_count")
    )

    context_counts = (
        counts
        .groupby(["period_id", "context"], observed=True)["upos_count"]
        .sum()
        .reset_index(name="context_count")
    )

    result = counts.merge(
        context_counts,
        on=["period_id", "context"],
        how="left",
    )

    result["probability"] = result["upos_count"] / result["context_count"]
    result["dimension"] = "P(" + result["next_upos"] + "|" + result["context"] + ")"

    wide_df = (
        result.pivot_table(
            index="period_id",
            columns="dimension",
            values="probability",
            fill_value=0.0,
        )
        .reset_index()
    )

    wide_df.columns.name = None

    cols = ["period_id"] + sorted(c for c in wide_df.columns if c != "period_id")
    wide_df = wide_df[cols]

    return wide_df

def get_syntactic_style_sub_distributions(
    reference_sample,
    test_sample,
    dimensions,
    shared_contexts_only=False,
):
    pattern = re.compile(r"^P\((.+)\|(.+)\)$")

    reference_grouped = {}
    test_grouped = {}

    for idx, dimension_name in dimensions.items():
        match = pattern.match(dimension_name)
        if not match:
            continue

        next_upos = match.group(1)
        context = match.group(2)

        if context not in reference_grouped:
            reference_grouped[context] = {}
        if context not in test_grouped:
            test_grouped[context] = {}

        reference_grouped[context][next_upos] = float(reference_sample[int(idx)])
        test_grouped[context][next_upos] = float(test_sample[int(idx)])

    if shared_contexts_only:
        contexts = sorted(set(reference_grouped).intersection(set(test_grouped)))
    else:
        contexts = sorted(set(reference_grouped).union(set(test_grouped)))

    distributions = []

    for context in contexts:
        reference_context = reference_grouped.get(context, {})
        test_context = test_grouped.get(context, {})

        labels = sorted(set(reference_context).union(set(test_context)))

        reference_distribution = np.array(
            [reference_context.get(label, 0.0) for label in labels],
            dtype=np.float64,
        )
        test_distribution = np.array(
            [test_context.get(label, 0.0) for label in labels],
            dtype=np.float64,
        )

        reference_sum = reference_distribution.sum()
        test_sum = test_distribution.sum()

        if reference_sum > 0.0:
            reference_distribution = reference_distribution / reference_sum

        if test_sum > 0.0:
            test_distribution = test_distribution / test_sum

        distributions.append(
            {
                "context": context,
                "labels": labels,
                "reference_distribution": reference_distribution,
                "test_distribution": test_distribution,
            }
        )

    return distributions

def get_lexical_distribution(
    pos_annotations, 
    docs_df, 
    allowed_upos=["NOUN", "VERB", "ADV", "ADJ"],
    **kwargs
):
    lexical_column= "lemma" # can be changed to "text" if we want to use the original word forms instead of lemmas

    logger.debug("Calculating lexical distribution...")
    df = pos_annotations.merge(
        docs_df[["doc_id", "period_id"]],
        on="doc_id",
        how="left"
    )

    df = df[df["upos"].isin(allowed_upos)].copy()
    df[lexical_column] = df[lexical_column].astype(str)
    logger.debug(f"Filtered annotations to allowed UPOS tags: {allowed_upos}. Initial size: {len(pos_annotations)}. Remaining rows: {len(df)}.")

    vocabulary = np.sort(df[lexical_column].dropna().unique())

    counts = (
        df.groupby(["period_id", lexical_column], observed=True)
        .size()
        .reset_index(name="freq")
    )
    logger.debug("Counted occurrences of each word per period_id.")

    lexical_dist = (
        counts
        .pivot(index="period_id", columns=lexical_column, values="freq")
        .reindex(columns=vocabulary, fill_value=0)
        .fillna(0)
        .sort_index()
        .pipe(lambda x: x.div(x.sum(axis=1), axis=0))
        .reset_index()
    )
    logger.debug("Calculated lexical distribution and normalized by row sums.")

    lexical_dist.columns.name = None
    cols = ["period_id"] + sorted(c for c in lexical_dist.columns if c != "period_id")
    lexical_dist = lexical_dist[cols]
    logger.debug("Reordered columns in lexical distribution DataFrame.")

    return lexical_dist
