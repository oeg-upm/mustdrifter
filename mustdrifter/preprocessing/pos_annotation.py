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
            logger.debug(f"Looking for the {lang} model at {str(model_path)}")
            if not os.path.exists(model_path):
                logger.debug(f"Downloading and initializing Stanza pipeline for language: {lang}")
                stanza.download(lang, processors="tokenize,pos,lemma", verbose=False, model_dir=str(STANZA_DIR))
            else:
                logger.debug(f"Stanza model for language {lang} already exists")

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

def annotate_pos(dataset, dataset_name, device="cuda", annotate_pos_batch_size=128, MAX_TEXT_LENGTH = 10000):
    """
    Annotate part-of-speech (POS) information for a dataset.

    This function detects the language of each document and applies
    language-specific POS tagging pipelines to generate token-level POS
    annotations. Annotated datasets can optionally be exported to CSV files.

    Parameters
    ----------
    dataset : pd.DataFrame
        Input dataset containing at least a `"content"` column with the
        documents to process.
    dataset_name : str
        Dataset name used to export the generated files. If None, results
        are not exported.
    device : str, optional
        Computation device used during POS annotation, such as `"cpu"` or
        `"cuda"`. Defaults to `"cuda"`.

    Returns
    -------
    tuple
        Tuple containing:

        - `pd.DataFrame`: input dataset enriched with document identifiers,
        processed content, and detected language labels.
        - `pd.DataFrame`: token-level POS annotations.

    Notes
    -----
    Language detection is performed before POS annotation in order to apply
    language-specific tagging pipelines.

    POS annotations are generated independently for each detected language
    group.
    """
    dataset["doc_id"]=   dataset.index
    dataset["content"]=  dataset["content"].astype(str).apply(remove_emojis)
    #dataset["lang"]=     dataset["content"].apply(detect_lang)
    dataset["lang"] = dataset["content"].str[:MAX_TEXT_LENGTH].apply(detect_lang)
    
    logger.debug("Language detection completed for all documents.")

    annotations= []
    logger.debug("Annotating POS tags for each language group.")

    for lang, group_lang in dataset.groupby("lang"):
        """
        pos_tagger= get_pipeline(lang, device=device)

        if pos_tagger is None: continue
        
        
        tags=    pos_tagger(group_lang["content"].tolist())
        tags_df= pos_tags_to_df(tags.to_dict(), group_lang["doc_id"].tolist())
        annotations.append(tags_df)

        logger.debug(f"POS annotation completed for language: {lang}")
        """
        logger.info(
            f"Annotating {len(group_lang)} documents for language: {lang}"
        )
    
        pos_tagger = get_pipeline(lang, device=device)
        
        if pos_tagger is None:
            logger.warning(f"Skipping unsupported language: {lang}")
            continue
    
        texts = group_lang["content"].str[:MAX_TEXT_LENGTH].tolist() #group_lang["content"].tolist()
        doc_ids = group_lang["doc_id"].tolist()
    
        lang_annotations = []
        
        for i in range(0, len(texts), annotate_pos_batch_size):
            batch_texts = texts[i:i + annotate_pos_batch_size]
            batch_ids = doc_ids[i:i + annotate_pos_batch_size]
    
            try:
                logger.debug(f"Tagging the following texts: \n{batch_texts}")
                tags = pos_tagger(batch_texts)
    
                tags_df = pos_tags_to_df(
                    tags.to_dict(),
                    batch_ids
                )
    
                lang_annotations.append(tags_df)
    
            except Exception:
                logger.exception(
                    f"POS annotation failed for language {lang}, "
                    f"batch {i // annotate_pos_batch_size}, "
                    f"documents {i}:{i + len(batch_texts)}"
                )
                raise
    
        if lang_annotations:
            annotations.extend(lang_annotations)
    
        logger.info(f"POS annotation completed for language: {lang}")
    
    annotations = pd.concat(annotations, ignore_index=True)
    #annotations= pd.concat(annotations)

    if dataset_name is not None:
        dataset.to_csv(f"{dataset_name}.csv", index=True)
        annotations.to_csv(f"{dataset_name}_pos.csv", index=False)
        logger.debug(f"Dataset and POS annotations saved to {dataset_name}.csv and {dataset_name}_pos.csv respectively.")

    return dataset, annotations
