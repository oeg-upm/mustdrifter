import fasttext
import urllib

# ---------- language detection ----------
url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"
urllib.request.urlretrieve(url, "lid.176.ftz")
lid = fasttext.load_model("lid.176.ftz")

def detect_lang(text):
    if not isinstance(text, str) or len(text.strip()) < 20:
        return "und"
    label, prob = lid.predict(text.replace("\n", " "), k=1)
    return label[0].replace("__label__", "")
