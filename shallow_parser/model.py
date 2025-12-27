import tensorflow as tf
from .config import TASKS
from .resources import LABELS
from .tokenizer import encode_sentence

_MODELS = {}

def load_model(language, model_path):
    if language not in _MODELS:
        _MODELS[language] = tf.saved_model.load(model_path)
    return _MODELS[language]

def predict(sentences, model, language):
    encoded = [encode_sentence(s, language) for s in sentences]

    masks = tf.constant([e[1] for e in encoded])
    segs  = tf.constant([e[2] for e in encoded])
    ids   = tf.constant([e[0] for e in encoded])
    tokens = [e[3] for e in encoded]

    preds = model([masks, segs, ids])

    results = {}
    for i, sent in enumerate(sentences):
        out = {}
        for t_i, task in enumerate(TASKS[language]):
            label_map = LABELS[language][task]
            decoded = [
                label_map[x] if label_map[x] != "unk" else ""
                for x in preds[t_i][0][i].numpy()
            ]
            words = sent.split()
            out[task] = [
                f"{words[j]}$%:%${decoded[j]}"
                if j < len(decoded) else f"{words[j]}$%:%$"
                for j in range(len(words))
            ]
        results[sent] = out

    return results

