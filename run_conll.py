from shallow_parser import shallow_parse
import sys

INPUT_FILE = sys.argv[1]
OUTPUT_FILE = sys.argv[2]
LANGUAGE = "hin"

with open(INPUT_FILE, "r", encoding="utf-8") as fin, \
     open(OUTPUT_FILE, "w", encoding="utf-8") as fout:

    for line in fin:
        line = line.strip()
        if not line:
            continue

        conll_lines = shallow_parse(line, LANGUAGE, mode="conll")
        fout.write("\n".join(conll_lines))
        fout.write("\n")   # ensure final newline

