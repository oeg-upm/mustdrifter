import pandas as pd
import numpy as np


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
    logger.debug("Calculating lexical distribution...")
    df = df_annotations.merge(
        docs_df[["doc_id", "period_id"]],
        on="doc_id",
        how="left"
    )

    df = df[df["upos"].isin(allowed_upos)].copy()
    df["text"] = df["text"].astype(str)
    logger.debug(f"Filtered annotations to allowed UPOS tags: {allowed_upos}. Remaining rows: {len(df)}.")
    
    vocabulary = np.sort(df["text"].dropna().unique())

    counts = (
        df.groupby(["period_id", "text"], observed=True)
        .size()
        .reset_index(name="freq")
    )
    logger.debug("Counted occurrences of each word per period_id.")

    lexical_dist = (
        counts
        .pivot(index="period_id", columns="text", values="freq")
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
