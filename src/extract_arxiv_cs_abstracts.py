"""
This file takes the raw arXiv json file and outputs a resulting .txt file filled with abstract texts for use later.

--input_jsonl: the raw arxiv json file containing the abstracts, such as the one linked in README (REQUIRED)

--output_txt: directory for a text file containing the output abstracts (REQUIRED)

--max_papers: a number to define how many papers to be used
"""

import argparse
import json
import re
from pathlib import Path

def stream_arxiv_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)

def is_cs_paper(rec):
    cats = rec.get("categories", "")
    return any(c.startswith("cs.") for c in cats.split())

def clean_text(t):
    return re.sub(r"\s+", " ", t.replace("\n", " ")).strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_jsonl", type=str, required=True)
    ap.add_argument("--output_txt", type=str, required=True)
    ap.add_argument("--max_papers", type=int, default=None)
    args = ap.parse_args()

    in_path = Path(args.input_jsonl)
    out_path = Path(args.output_txt)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    scanned = cs_matched = written = 0

    with out_path.open("w", encoding="utf-8") as out:
        for rec in stream_arxiv_jsonl(in_path):
            scanned += 1
            if not is_cs_paper(rec):
                continue
            cs_matched += 1
            abs_text = clean_text(rec.get("abstract", ""))
            if abs_text:
                out.write(abs_text + "\n")
                written += 1
            if args.max_papers is not None and written >= args.max_papers:
                break

    print(f"Total records scanned: {scanned}")
    print(f"CS papers matched: {cs_matched}")
    print(f"Abstracts written: {written}")
    print(f"Wrote: {out_path}")

if __name__ == "__main__":
    main()
