import stanza
import pandas as pd

from .lang_detection import detect_lang


import logging
logger = logging.getLogger(__name__)
import os
import emoji
import re
from pathlib import Path

BASE_DIR= Path(__file__).resolve().parent
STANZA_DIR= BASE_DIR / "stanza_models"
os.makedirs(STANZA_DIR, exist_ok=True)

def remove_emojis(text):
    if pd.isna(text):
        return ""
    text = str(text)
    text = emoji.replace_emoji(text, replace="")
    text = re.sub(r"\s+", " ", text).strip()
    return text

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
            model_path = STANZA_DIR / lang

            if not os.path.exists(model_path):
                logger.debug(f"Downloading and initializing Stanza pipeline for language: {lang}")
                stanza.download(lang, processors="tokenize,pos", verbose=False, model_dir=STANZA_DIR)
                
            PIPELINES[lang] = stanza.Pipeline(
                lang=lang,
                processors="tokenize,pos",
                tokenize_no_ssplit=True,
                use_gpu=True,
                verbose=False,
                model_dir=STANZA_DIR
            )
        except:
            PIPELINES[lang] = None
        logger.debug(f"Stanza pipeline for language {lang} initialized: {PIPELINES[lang] is not None}")
    return PIPELINES[lang]

def annotate_pos(dataset, dataset_name):   
    dataset["doc_id"]= dataset.index
    
    dataset["content"] = dataset["content"].astype(str).apply(remove_emojis)

    dataset["lang"]= dataset["content"].apply(detect_lang)
    logger.debug("Language detection completed for all documents.")
    
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
        logger.debug(f"Dataset and POS annotations saved to {dataset_name}.csv and {dataset_name}_pos.csv respectively.")

    return dataset, annotations
