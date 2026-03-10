from pathlib import Path
import fasttext
import urllib.request

import logging
logger = logging.getLogger(__name__)

MODEL_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"
MODEL_PATH = Path(__file__).resolve().parent / "lid.176.ftz"

_lid = None


def _get_lid():
    global _lid

    if not MODEL_PATH.exists():
        urllib.request.urlretrieve(MODEL_URL, str(MODEL_PATH))
    _lid = fasttext.load_model(str(MODEL_PATH))

    return _lid


def detect_lang(text):
    if not isinstance(text, str) or len(text.strip()) < 5:
        return "und"
    if _lid is None: lid = _get_lid()
    label, prob = lid.predict(text.replace("\n", " "), k=1)
    return label[0].replace("__label__", "")