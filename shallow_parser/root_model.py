"""
Word → Morphological Root extraction
SentencePiece + TensorFlow SavedModel
"""

import tensorflow as tf
import tensorflow_text as tftxt
import numpy as np
from typing import List, Dict

# ======================
# Paths (relative-safe)
# ======================

VOCAB_FILE = "models/spm.model"
MODEL_PATH = "models/hi_morph_word2root/"

# ======================
# Tokenizer & Model
# ======================

_tokenizer = tftxt.SentencepieceTokenizer(
    model=tf.io.gfile.GFile(VOCAB_FILE, "rb").read(),
    add_eos=True
)

_model = tf.saved_model.load(MODEL_PATH)
_serving_fn = _model.signatures["serving_default"]

# ======================
# Cache
# ======================

_CACHE: Dict[str, str] = {}

# ======================
# Helpers
# ======================

def _decode(ids: np.ndarray) -> str:
    return _tokenizer.detokenize(ids).numpy().decode("utf-8").strip()

def _safe_root(word: str, root: str) -> str:
    if not root or " " in root:
        return word
    return root

# ======================
# Public APIs
# ======================

def get_root(sentence: str) -> List[str]:
    """
    Word-by-word root extraction.
    Returns: [word$%:%$root, ...]
    """
    output = []

    for word in sentence.split():
        if word in _CACHE:
            output.append(_CACHE[word])
            continue

        ids = _tokenizer.tokenize(word)
        ids = tf.expand_dims(tf.cast(ids, tf.int64), axis=0)

        out = _serving_fn(ids)["outputs"].numpy()[0]
        root = _safe_root(word, _decode(out))

        final = f"{word}$%:%${root}"
        _CACHE[word] = final
        output.append(final)

    return output


def get_root_batch(sentence: str) -> List[str]:
    """
    ✅ Correct batch implementation.
    No shape mismatch. No Pack error.
    """
    words = sentence.split()
    missing = [w for w in words if w not in _CACHE]

    if missing:
        # 1️⃣ Tokenize → RaggedTensor
        ragged = _tokenizer.tokenize(missing)

        # 2️⃣ Convert to dense padded tensor (B, T)
        padded = ragged.to_tensor(default_value=0)

        # 3️⃣ Run model
        outputs = _serving_fn(tf.cast(padded, tf.int64))["outputs"].numpy()

        # 4️⃣ Decode & cache
        for word, out_ids in zip(missing, outputs):
            root = _safe_root(word, _decode(out_ids))
            _CACHE[word] = f"{word}$%:%${root}"

    return [_CACHE[w] for w in words]

