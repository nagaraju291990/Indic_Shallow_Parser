import codecs
from indicnlp import common, loader
from .config import INDIC_NLP_RESOURCES

def read_labels(path):
    return [line.strip() for line in codecs.open(path)]

def init_indic_nlp():
    common.set_resources_path(INDIC_NLP_RESOURCES)
    loader.load()

LABELS = {
    "hin": {
        "pos": read_labels("labels/hin/hi_pos_v2.txt"),
        "chunk": read_labels("labels/hin/hi_chunk_v2.txt"),
        "hi_morph_pos": read_labels("labels/hin/hi_morph_pos_v2.txt"),
        "hi_morph_gender": read_labels("labels/hin/hi_morph_gender_v2.txt"),
        "hi_morph_number": read_labels("labels/hin/hi_morph_number_v2.txt"),
        "hi_morph_person": read_labels("labels/hin/hi_morph_person_v2.txt"),
        "hi_morph_case": read_labels("labels/hin/hi_morph_case_v2.txt"),
        "hi_morph_vib": read_labels("labels/hin/hi_morph_vib_v2.txt"),
    }
}

R_LIST = set(read_labels("labels/hin/r_list.txt"))
FILTER_POINTS = set(
    c.strip()
    for c in codecs.open("labels/hin/out2.chars.txt.final")
    if c.strip() != "f"
)

