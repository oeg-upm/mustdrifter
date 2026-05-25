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
                "doc_id":     doc_id[sentence_id],
                "id":         token.get("id"),
                "text":       token.get("text"),
                "upos":       token.get("upos"),
                "lemma":      token.get("lemma"),
                "xpos":       token.get("xpos"),
                "feats":      token.get("feats"),
                "start_char": token.get("start_char"),
                "end_char":   token.get("end_char"),
                "misc":       token.get("misc"),
            })

    logger.debug("POS tags converted to DataFrame successfully.")
    return pd.DataFrame(rows)


PIPELINES = {}

def get_pipeline(lang, device="cuda"):
    if lang == "und":
        return None

    logger.debug(f"Getting POS pipeline for language: {lang}")
    if lang not in PIPELINES:
        try:
            model_path = STANZA_DIR / lang

            if not os.path.exists(model_path):
                logger.debug(f"Downloading and initializing Stanza pipeline for language: {lang}")
                stanza.download(lang, processors="tokenize,pos,lemma", verbose=False, model_dir=str(STANZA_DIR))

            PIPELINES[lang] = stanza.Pipeline(
                lang=lang,
                processors="tokenize,pos,lemma",
                tokenize_no_ssplit=True,
                device=device,
                verbose=False,
                model_dir=str(STANZA_DIR)
            )

        except Exception as e:
            logger.error(f"Error initializing Stanza pipeline for language {lang}: {e}")
            PIPELINES[lang] = None

        logger.debug(f"Stanza pipeline for language {lang} initialized: {PIPELINES[lang] is not None}")
    return PIPELINES[lang]

def annotate_pos(dataset, dataset_name, device="cuda"):
    """
    Annotates parts of speech (POS) for a given dataset.

    This function processes a dataset by detecting the language of each document, 
    and then annotates parts of speech (POS) tags for each language group using 
    a language-specific POS tagger. The annotated dataset and POS tags are saved 
    to CSV files if a dataset name is provided.
    
    Parameters
    ----------
        dataset (pd.DataFrame): The input dataset containing at least a "content" column 
            with text data to be processed.
        dataset_name (str): The name of the dataset, used for saving the output files. 
            If None, the results are not saved to files.
        device (str, optional): The device to use for processing (e.g., "cuda" or "cpu"). 
            Defaults to "cuda".
    
    Returns
    -------
    tuple
        A tuple containing:
            - pd.DataFrame: The input dataset with additional columns for document ID, 
              content (processed), and detected language.
            - pd.DataFrame: A DataFrame containing the POS annotations for the dataset.
    """
    dataset["doc_id"]=   dataset.index
    dataset["content"]=  dataset["content"].astype(str).apply(remove_emojis)
    dataset["lang"]=     dataset["content"].apply(detect_lang)

    logger.debug("Language detection completed for all documents.")

    annotations= []
    logger.debug("Annotating POS tags for each language group.")

    for lang, group_lang in dataset.groupby("lang"):
        pos_tagger= get_pipeline(lang, device=device)

        if pos_tagger is None: continue

        tags=    pos_tagger(group_lang["content"].tolist())
        tags_df= pos_tags_to_df(tags.to_dict(), group_lang["doc_id"].tolist())
        annotations.append(tags_df)

        logger.debug(f"POS annotation completed for language: {lang}")

    annotations= pd.concat(annotations)

    if dataset_name is not None:
        dataset.to_csv(f"{dataset_name}.csv", index=True)
        annotations.to_csv(f"{dataset_name}_pos.csv", index=False)
        logger.debug(f"Dataset and POS annotations saved to {dataset_name}.csv and {dataset_name}_pos.csv respectively.")

    return dataset, annotations
