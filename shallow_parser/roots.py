# shallow_parser/roots.py
from typing import List
from .root_model import get_root_batch

def get_morph_roots(sentences: List[str], language: str):
    if language not in ("hin", "hi"):
        return [[f"{w}$%:%${w}" for w in s.split()] for s in sentences]

    return [get_root_batch(s) for s in sentences]

