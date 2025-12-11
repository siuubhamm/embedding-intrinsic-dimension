"""
This file is for creating plots of the redundancy ratio versus the total ED

--summary_csv: csv file generated from summarize_results.py (REQUIRED)

--model: specify which model to use, defaults to gloVe

--level: either token-level for whole text level

--sample_file: input .txt for number of dimensions

--out_png: directory for saving output graph file
"""

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary_csv", type=str, required=True)
    ap.add_argument("--model", type=str, default="glove")
    ap.add_argument("--level", type=str, default="token", choices=["token","abstract"])
    ap.add_argument("--sample_file", type=str, default="sample_50000.txt")
    ap.add_argument("--out_png", type=str, default=None)
    args = ap.parse_args()

    m = pd.read_csv(args.summary_csv)
    sub = m[(m.model == args.model) & (m.level == args.level) & (m.sample_file == args.sample_file)].copy()
    if sub.empty:
        raise SystemExit("No rows matched filters. Check --model/--level/--sample_file.")

    out_png = args.out_png or f"runs/ratio_vs_ed_{args.level}_{args.model}_{Path(args.sample_file).stem}.png"

    plt.figure(figsize=(7,5))
    for k in sorted(sub.k.unique()):
        s2 = sub[sub.k == k].sort_values("ED")
        plt.errorbar(s2.ED, s2.ratio_mean, yerr=s2.ratio_std, fmt="o-", label=f"k={k}")

    plt.title(f"Redundancy ratio vs ED ({args.level}); {args.sample_file}")
    plt.xlabel("ED (Extrinsic Dimension)")
    plt.ylabel("ID / ED")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200)
    print("Saved:", out_png)


if __name__ == "__main__":
    main()
