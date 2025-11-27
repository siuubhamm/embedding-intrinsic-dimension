import argparse
from pathlib import Path
import pandas as pd
from utils.lid_utils import tokenize


def compute_stats(sample_path: Path):
    abstracts = [ln.strip() for ln in sample_path.open("r", encoding="utf-8") if ln.strip()]

    token_counts = []
    all_tokens = []
    for abs_text in abstracts:
        toks = tokenize(abs_text)
        token_counts.append(len(toks))
        all_tokens.extend(toks)

    total_tokens = len(all_tokens)
    vocab_size = len(set(all_tokens))

    return {
        "dataset": "arxiv_cs",
        "sample_file": sample_path.name,
        "num_abstracts": len(abstracts),
        "total_tokens": total_tokens,
        "vocab_size": vocab_size,
        "lexical_diversity": (vocab_size / total_tokens) if total_tokens > 0 else 0.0,
        "avg_tokens_per_abstract": (sum(token_counts) / len(token_counts)) if token_counts else 0.0,
        "median_tokens_per_abstract": sorted(token_counts)[len(token_counts)//2] if token_counts else 0,
        "min_tokens_per_abstract": min(token_counts) if token_counts else 0,
        "max_tokens_per_abstract": max(token_counts) if token_counts else 0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=str, nargs="+", required=True)
    ap.add_argument("--out_csv", type=str, required=True)
    args = ap.parse_args()

    rows = [compute_stats(Path(p)) for p in args.samples]
    df = pd.DataFrame(rows)
    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print("Wrote:", out)
    print(df)


if __name__ == "__main__":
    main()
