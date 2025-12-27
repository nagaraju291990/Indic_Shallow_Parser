from indicnlp.tokenize import indic_tokenize, sentence_tokenize
from .config import LANG_MAP
from .resources import R_LIST, FILTER_POINTS

ZERO_WIDTH = ["\u200d", "\u200b", "\u200e"]

def normalize_text(text: str) -> str:
    replacements = {
        "व ें": "वें",
        "स े": "से ",
        "म ें": "में",
        "16 ़": "16",
    }

    for k, v in replacements.items():
        text = text.replace(k, v)

    for z in ZERO_WIDTH:
        text = text.replace(z, "")

    text = " ".join(text.split())
    text = " ".join(w for w in text.split() if w not in R_LIST)
    text = " ".join(indic_tokenize.trivial_tokenize(text))
    text = " ".join(w for w in text.split() if w not in FILTER_POINTS)

    return text.strip()

def split_sentences(text, language):
    return sentence_tokenize.sentence_split(
        text, lang=LANG_MAP[language]
    )

