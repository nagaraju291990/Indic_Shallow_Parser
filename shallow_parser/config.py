VERSION = "5.0"
MAX_SEQ_LEN = 400

INDIC_NLP_RESOURCES = "indic_nlp_resources/"
HISTORY_FILE = "nov_12_2022_history_1.json"

LANG_MAP = {"hin": "hi"}

TASKS = {
    "hin": [
        "pos",
        "hi_morph_pos",
        "hi_morph_gender",
        "hi_morph_number",
        "hi_morph_person",
        "hi_morph_case",
        "hi_morph_vib",
        "chunk",
    ]
}

MODELS_PATH = {
    "hin": {"all": "models/hi_all"}
}

VOCAB_PATH = {
    "hin": "models/hi_all/assets/vocab.txt"
}

