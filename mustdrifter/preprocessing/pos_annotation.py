import stanza
import pandas as pd

from .lang_detection import detect_lang


import logging
logger = logging.getLogger(__name__)

# ---------- POS dynamic pipelines ----------

def pos_tags_to_df(pos_tags, doc_id):
    logger.debug("Converting POS tags to DataFrame...")
    rows = []
    for sentence_id, sentence in enumerate(pos_tags):
        for token in sentence:
            rows.append({
                "doc_id": doc_id[sentence_id],
                "id": token.get("id"),
                "text": token.get("text"),
                "upos": token.get("upos"),
                "xpos": token.get("xpos"),
                "feats": token.get("feats"),
                "start_char": token.get("start_char"),
                "end_char": token.get("end_char"),
                "misc": token.get("misc"),
            })
    logger.debug("POS tags converted to DataFrame successfully.")
    return pd.DataFrame(rows)


PIPELINES = {}

def get_pipeline(lang):
    if lang == "und":
        return None

    logger.debug(f"Getting POS pipeline for language: {lang}")
    if lang not in PIPELINES:
        try:
            logger.debug(f"Downloading and initializing Stanza pipeline for language: {lang}")
            stanza.download(lang, processors="tokenize,pos", verbose=False)
            PIPELINES[lang] = stanza.Pipeline(
                lang=lang,
                processors="tokenize,pos",
                tokenize_no_ssplit=True,
                use_gpu=True,
                verbose=False
            )
        except:
            PIPELINES[lang] = None
        logger.debug(f"Stanza pipeline for language {lang} initialized: {PIPELINES[lang] is not None}")
    return PIPELINES[lang]

def annotate_pos(dataset, dataset_name):
    logger.info("Starting POS annotation...")
    
    dataset["doc_id"]= dataset.index
    dataset["lang"]= dataset["content"].apply(detect_lang)
    
    annotations= []
    logger.debug("Annotating POS tags for each language group...")
    for lang, group_lang in dataset.groupby("lang"):
        pos_tagger= get_pipeline(lang)
        if pos_tagger is None: continue
        tags= pos_tagger(group_lang["content"].tolist())
        tags_df= pos_tags_to_df(tags.to_dict(), group_lang["doc_id"].tolist() )
        annotations.append(tags_df)
        logger.debug(f"POS annotation completed for language: {lang}")
    
    annotations= pd.concat(annotations)
    
    if dataset_name is not None:
        dataset.to_csv(f"{dataset_name}.csv", index=True)
        annotations.to_csv(f"{dataset_name}_pos.csv", index=False)
        logger.info(f"Dataset and POS annotations saved to {dataset_name}.csv and {dataset_name}_pos.csv respectively.")

    return dataset, annotations
