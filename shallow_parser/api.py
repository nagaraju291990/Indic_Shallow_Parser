import json
from .config import MODELS_PATH, HISTORY_FILE
from .normalization import normalize_text, split_sentences
from .roots import get_morph_roots
from .model import load_model, predict
from .cache import has, get, set
from .ssf import format_as_ssf
from .resources import init_indic_nlp
from .conll import format_as_conll

init_indic_nlp()

def shallow_parse(text, language="hin", mode="ssf"):
    if isinstance(text, str):
        text = normalize_text(text)
        sentences = split_sentences(text, language)
    else:
        sentences = [normalize_text(t) for t in text]

    new_sents = [s for s in sentences if not has(s, language)]

    if new_sents:
        model = load_model(language, MODELS_PATH[language]["all"])
        preds = predict(new_sents, model, language)
        roots = get_morph_roots(new_sents, language)

        with open(HISTORY_FILE, "a") as hf:
            for i, s in enumerate(new_sents):
                preds[s]["root"] = roots[i]
                set(s, language, preds[s])
                hf.write(json.dumps({s: preds[s]}) + "\n")

    results = [get(s, language) for s in sentences]

    if mode == "ssf":
        return format_as_ssf(results, language)

    if mode == "conll":
        return format_as_conll(results, language)

    return results

