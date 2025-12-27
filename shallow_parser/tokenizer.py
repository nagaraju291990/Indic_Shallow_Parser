import tokenization
from .config import VOCAB_PATH, MAX_SEQ_LEN
from .utils import convert_single_example


TOKENIZER = {
    "hin": tokenization.FullTokenizer(
        vocab_file=VOCAB_PATH["hin"],
        do_lower_case=True,
        split_on_punc=False,
    )
}

class InputExample:
    def __init__(self, guid, text):
        self.guid = guid
        self.text_a = text
        self.text_b = None

def encode_sentence(text, language):
    example = InputExample("unused", text)
    return convert_single_example(
        0, example, MAX_SEQ_LEN, TOKENIZER[language]
    )

