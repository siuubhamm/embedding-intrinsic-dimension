"""
This file is for creating sample tokens of specified dimensions.

--input_txt: a .txt file with text to create tokens for, for example extract_arxiv_cs_abstracts.py output (REQUIRED)

--out_dir: specified directory to place output .json and .txt files

--seed: set for different randomizations, defaults to 2025.

--sizes: specified set of dimensions to create. defaults to 5000, 20000, and 50000
"""

import argparse
import json
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_txt", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--sizes", type=int, nargs="+", default=[5000, 20000, 50000])
    args = ap.parse_args()

    in_path = Path(args.input_txt)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    abstracts = [ln.strip() for ln in in_path.open("r", encoding="utf-8") if ln.strip()]
    n_total = len(abstracts)

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n_total)

    saved = {"seed": args.seed, "n_total": n_total, "sizes": {}}

    for N in args.sizes:
        idx = perm[:N].tolist()
        sample = [abstracts[i] for i in idx]

        txt_path = out_dir / f"sample_{N}.txt"
        idx_path = out_dir / f"sample_{N}_idx.json"
        txt_path.write_text("\n".join(sample) + "\n", encoding="utf-8")
        idx_path.write_text(json.dumps(idx), encoding="utf-8")

        saved["sizes"][str(N)] = {"txt_path": str(txt_path), "idx_path": str(idx_path)}

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(saved, indent=2) + "\n", encoding="utf-8")

    print("Saved samples to:", out_dir)
    print("Manifest:", manifest_path)


if __name__ == "__main__":
    main()
