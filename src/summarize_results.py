import argparse
from pathlib import Path
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", type=str, default="runs")
    ap.add_argument("--stats_csv", type=str, default=None)
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    df = pd.read_csv(runs_dir / "id_results.csv")

    summary = (df
        .groupby(["dataset","sample_file","model","ED","level","k"])
        .agg(global_id_mean=("global_id","mean"),
             global_id_std=("global_id","std"),
             ratio_mean=("ratio","mean"),
             ratio_std=("ratio","std"),
             n_points=("n_points","first"))
        .reset_index()
    )

    out_summary = runs_dir / "id_summary.csv"
    summary.to_csv(out_summary, index=False)
    print("Wrote:", out_summary)

    if args.stats_csv is not None:
        stats = pd.read_csv(args.stats_csv)
        merged = summary.merge(stats, on=["sample_file"], how="left")
        out_merged = runs_dir / "id_summary_with_stats.csv"
        merged.to_csv(out_merged, index=False)
        print("Wrote:", out_merged)


if __name__ == "__main__":
    main()
