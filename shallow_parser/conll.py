# shallow_parser/conll.py

from .config import LANG_MAP
from wxconv import WXC
from re import search

def format_as_conll(parsed_sentences, language):
    """
    Convert parsed output to CoNLL format.
    One blank line between sentences.
    """
    conv = WXC(order='utf2wx', lang=language)
    inlang = LANG_MAP[language]
    lines = []

    for sent in parsed_sentences:
        pos_ = sent["pos"]
        words = [w.split("$%:%$")[0] for w in pos_]

        roots = sent["root"]
        xpos = pos_
        upos = sent.get(inlang + "_morph_pos", pos_)

        gender = sent.get(inlang + "_morph_gender", [])
        number = sent.get(inlang + "_morph_number", [])
        person = sent.get(inlang + "_morph_person", [])
        case = sent.get(inlang + "_morph_case", [])
        vib = sent.get(inlang + "_morph_vib", [])
        suff = []
        for index, w_v in enumerate(vib):
          w, i_v = w_v.split('$%:%$')
          assert w == words[index]
          if i_v in ['', '0']:
            suff.append(w + '$%:%$' + i_v)
          else:
            i_suff = conv.convert(i_v)
            suff.append(w + '$%:%$' + i_suff)
        chunk = sent.get("chunk", [])

        for i, word in enumerate(words):
            feats = []

            def val(arr):
                return arr[i].split("$%:%$")[1] if i < len(arr) else "_"

            feats.append(f"Gender={val(gender)}")
            feats.append(f"Number={val(number)}")
            feats.append(f"Person={val(person)}")
            feats.append(f"Case={val(case)}")
            feats.append(f"Vib={val(vib)}")
            feats.append(f"Suff={val(suff)}")
            feat_str = "|".join(
                f for f in feats if not f.endswith("=_")
            ) or "_"

            line = "\t".join([
                str(i + 1),                   # ID
                word,                         # FORM
                roots[i].split("$%:%$")[1],   # LEMMA
                val(upos),                    # UPOS
                val(xpos),                    # XPOS
                feat_str,                     # FEATS
                "_",                           # HEAD
                val(chunk),                   # DEPREL (chunk)
                "_"                            # MISC
            ])

            lines.append(line)

        lines.append("")  # blank line between sentences

    return lines
