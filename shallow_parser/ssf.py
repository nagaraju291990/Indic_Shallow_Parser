from .config import LANG_MAP
from wxconv import WXC


def ssf_sentence_start(i):
    return f'<Sentence id="{i}">'

def ssf_sentence_end():
    return "</Sentence>"

def format_as_ssf(parsed_sentences, language):
    conv = WXC(order='utf2wx', lang=language)
    inlang = LANG_MAP[language]
    output = []
    sent_id = 1

    for sent in parsed_sentences:
        output.append(ssf_sentence_start(sent_id))

        pos_ = sent["pos"]
        words = [w.split("$%:%$")[0] for w in pos_]
        chunk_ = sent.get("chunk", [w + "$%:%$ " for w in words])
        root_ = sent["root"]

        mp = sent.get(inlang + "_morph_pos", [w + "$%:%$ " for w in words])
        mg = sent.get(inlang + "_morph_gender", [w + "$%:%$ " for w in words])
        mn = sent.get(inlang + "_morph_number", [w + "$%:%$ " for w in words])
        mper = sent.get(inlang + "_morph_person", [w + "$%:%$ " for w in words])
        mc = sent.get(inlang + "_morph_case", [w + "$%:%$ " for w in words])
        mv = sent.get(inlang + "_morph_vib", [w + "$%:%$ " for w in words])
        ms = [conv.convert(mv_i.split("$%:%$")[1]) for mv_i in mv]
        chunk_id = 0
        token_id = 0
        open_chunk = False

        for i, word in enumerate(words):
            chunk_tag = chunk_[i].split("$%:%$")[1]
            pos_tag = pos_[i].split("$%:%$")[1]

            if "B-" in chunk_tag or i == 0:
                if open_chunk:
                    output.append("\t))")
                chunk_id += 1
                token_id = 1
                open_chunk = True
                output.append(
                    f"{chunk_id}\t((\t{chunk_tag.replace('B-', '').replace('I-', '')}"
                )
            else:
                token_id += 1

            fs = (
                f"<fs af='{root_[i].split('$%:%$')[1]},"
                f"{mp[i].split('$%:%$')[1]},"
                f"{mg[i].split('$%:%$')[1]},"
                f"{mn[i].split('$%:%$')[1]},"
                f"{mper[i].split('$%:%$')[1]},"
                f"{mc[i].split('$%:%$')[1]},"
                f"{mv[i].split('$%:%$')[1]},"
                f"{ms[i]}'>"
            )

            output.append(
                f"{chunk_id}.{token_id}\t{word}\t{pos_tag}\t{fs}"
            )

        if open_chunk:
            output.append("\t))")

        output.append(ssf_sentence_end())
        sent_id += 1

    return output
