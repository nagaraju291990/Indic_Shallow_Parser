_CACHE = {}

def has(sentence, language):
    return f"{sentence}$###${language}" in _CACHE

def get(sentence, language):
    return _CACHE[f"{sentence}$###${language}"]

def set(sentence, language, value):
    _CACHE[f"{sentence}$###${language}"] = value

