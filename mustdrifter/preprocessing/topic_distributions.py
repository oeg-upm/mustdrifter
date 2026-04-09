
from bertopic import BERTopic

import logging
logger = logging.getLogger(__name__)


def get_thematic_dimension(
    docs_df,
    nr_topics=30,
    embedding_model="intfloat/multilingual-e5-large",
    **kwargs
    ):
    logger.debug("Computing thematic dimension with BERTopic")
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
    logger.debug("BERTopic fitting complete")
    
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