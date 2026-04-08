import pandas as pd
import numpy as np
from bertopic import BERTopic



from itertools import permutations
import logging
logger = logging.getLogger(__name__)


def get_pos_distribution(df_annotations):
    logger.debug("Calculating POS distribution...")
    pos_dist = (
        df_annotations.groupby(["doc_id", "upos"])
          .size()
          .unstack(fill_value=0)
          .pipe(lambda df: df.div(df.sum(axis=1), axis=0))
          .reset_index()
    )

    pos_dist.columns.name = None
    cols = ["doc_id"] + sorted(c for c in pos_dist.columns if c != "doc_id")
    pos_dist = pos_dist[cols]
    logger.debug("POS distribution calculated successfully.")

    return pos_dist

def get_pos_ngram_distribution(df, n_min=2, n_max=4):
    logger.debug(f"Calculating POS n-gram distribution for n={n_min} to n={n_max}...")
    rows = []

    for doc_id, g in df.groupby("doc_id", sort=False):
        g["id"] = pd.to_numeric(g["id"], errors="coerce")
        g= g.sort_values("id")
        tokens = g["upos"].dropna().astype(str).tolist()

        for n in range(n_min, n_max + 1):
            if len(tokens) < n:
                continue

            ngrams = zip(*[tokens[i:] for i in range(n)])

            for ng in ngrams:
                rows.append((doc_id, "+".join(ng)))

        logger.debug(f"Processed doc_id {doc_id} with {len(tokens)} tokens for POS n-grams.")

    df_ngrams = pd.DataFrame(rows, columns=["doc_id", "ngram"])

    matrix = (
        df_ngrams
        .value_counts(["doc_id", "ngram"])
        .unstack(fill_value=0)
        .sort_index()
    )

    matrix["doc_id"]= matrix.index
    matrix.index.name = None
    matrix.columns.name = None
    logger.info("POS n-gram distribution calculated successfully.")
    return matrix

def get_lexical_distribution(df_annotations, docs_df, allowed_upos=["NOUN", "VERB", "ADV", "ADJ"]):
    lexical_column= "lemma" # can be changed to "text" if we want to use the original word forms instead of lemmas

    logger.debug("Calculating lexical distribution...")
    df = df_annotations.merge(
        docs_df[["doc_id", "period_id"]],
        on="doc_id",
        how="left"
    )

    df = df[df["upos"].isin(allowed_upos)].copy()
    df[lexical_column] = df[lexical_column].astype(str)
    logger.debug(f"Filtered annotations to allowed UPOS tags: {allowed_upos}. Remaining rows: {len(df)}.")

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
    df_annotations,
    docs_df,
    allowed_upos=["NOUN", "PRON", "VERB", "ADV", "ADJ", "DET"]
):
    """
    Compute the probability distribution of syntactic content rules by period.

    A rule is defined as any permutation of the UPOS tags in `allowed_upos`.
    For each document, a rule is considered present if it appears as an
    ordered non-contiguous subsequence in the document's UPOS sequence.

    Parameters
    ----------
    df_annotations : pd.DataFrame
        DataFrame with at least:
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

    df = df_annotations.merge(
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

    rule_totals = df_period_counts[rules].sum(axis=1)

    df_period_probs = df_period_counts.copy()
    df_period_probs[rules] = (
        df_period_probs[rules]
        .div(rule_totals.replace(0, pd.NA), axis=0)
        .fillna(0.0)
    )

    df_period_probs = df_period_probs[["period_id"] + rules]

    return df_period_probs


def _rule_exists_contiguous(pos_sequence, rule_tags):
    """
    Check if rule_tags appears as a contiguous subsequence.
    """
    n = len(rule_tags)
    L = len(pos_sequence)

    for i in range(L - n + 1):
        if tuple(pos_sequence[i:i+n]) == rule_tags:
            return 1

    return 0


def get_syntactic_style_distribution(
    df_annotations,
    docs_df,
    allowed_upos=["NOUN", "PRON", "VERB", "ADV", "ADJ", "DET"]
):
    logger.debug("Calculating syntactic style dimension...")

    df = df_annotations.merge(
        docs_df[["doc_id", "period_id"]],
        on="doc_id",
        how="left"
    )

    df = df[df["upos"].isin(allowed_upos)].copy()
    logger.debug(
        f"Filtered annotations to allowed UPOS tags: {allowed_upos}. Remaining rows: {len(df)}."
    )

    if df.empty:
        logger.debug("No rows remaining after filtering. Returning empty DataFrame.")
        return pd.DataFrame(columns=["period_id"])

    df = df.sort_values(["doc_id", "id"])
    logger.debug("Sorted annotations by doc_id and id.")

    rules = np.sort(["+".join(rule) for rule in permutations(allowed_upos)])
    logger.debug(f"Generated {len(rules)} syntactic style rules.")

    doc_period = (
        df[["doc_id", "period_id"]]
        .drop_duplicates(subset=["doc_id"])
        .set_index("doc_id")["period_id"]
    )

    doc_pos = df.groupby("doc_id", observed=True)["upos"].apply(list)
    logger.debug("Built UPOS sequences for each doc_id.")

    rules_split = {rule: tuple(rule.split("+")) for rule in rules}


    rows = []

    for doc_id, pos_seq in doc_pos.items():
        row = {
            "doc_id": doc_id,
            "period_id": doc_period.loc[doc_id]
        }

        if len(pos_seq) < len(allowed_upos):
            for rule in rules:
                row[rule] = 0
            rows.append(row)
            continue

        for rule, rule_tags in rules_split.items():
            row[rule] = _rule_exists_contiguous(pos_seq, rule_tags)

        rows.append(row)

    logger.debug("Built document-level contiguous rule matrix.")

    doc_rule_df = pd.DataFrame(rows)

    syntactic_style_dist = (
        doc_rule_df
        .drop(columns=["doc_id"])
        .groupby("period_id", observed=True)
        .sum()
        .reindex(columns=rules, fill_value=0)
        .fillna(0)
        .sort_index()
        .pipe(lambda x: x.div(x.sum(axis=1), axis=0))
        .fillna(0.0)
        .reset_index()
    )

    logger.debug("Calculated syntactic style distribution and normalized by row sums.")

    syntactic_style_dist.columns.name = None
    cols = ["period_id"] + sorted(c for c in syntactic_style_dist.columns if c != "period_id")
    syntactic_style_dist = syntactic_style_dist[cols]

    logger.debug("Reordered columns in syntactic style DataFrame.")

    return syntactic_style_dist


def get_thematic_dimension(
    docs_df,
    nr_topics=30,
    embedding_model="intfloat/multilingual-e5-large"
    ):

    df = docs_df.copy()
    df["content"] = df["content"].fillna("").astype(str).str.strip()

    docs = df["content"].tolist()

    topic_model = BERTopic(
        embedding_model=embedding_model,
        nr_topics=nr_topics,
        verbose=True
    )

    topics, _ = topic_model.fit_transform(docs)
    df["topic"] = topics

    topic_counts = (
        df.groupby(["period_id", "topic"], observed=True)
        .size()
        .reset_index(name="freq")
    )

    topic_vocabulary = sorted(topic_counts["topic"].unique())

    thematic_dist = (
        topic_counts
        .pivot(index="period_id", columns="topic", values="freq")
        .reindex(columns=topic_vocabulary, fill_value=0)
        .fillna(0)
        .sort_index()
        .pipe(lambda x: x.div(x.sum(axis=1), axis=0))
        .reset_index()
    )

    thematic_dist.columns.name = None
    cols = ["period_id"] + sorted(c for c in thematic_dist.columns if c != "period_id")
    thematic_dist = thematic_dist[cols]


    return thematic_dist