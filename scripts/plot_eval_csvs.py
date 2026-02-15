import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser(description="Plot evaluation CSVs produced by test_adapters.py --results_csv.")
    p.add_argument("--csvs", nargs="+", required=True, help="List of CSV files to aggregate/plot.")
    p.add_argument("--metric", default=None, help="Metric column to plot (e.g., acc_test_base). If omitted, picks first acc* column.")
    p.add_argument("--group_by", nargs="+", default=["dataset"], help="Grouping keys for bar plot.")
    p.add_argument("--output", default="plot.png", help="Output image file.")
    return p.parse_args()


def main():
    args = parse_args()
    frames = [pd.read_csv(p) for p in args.csvs]
    df = pd.concat(frames, ignore_index=True)

    # pick metric
    metric = args.metric
    if metric is None:
        acc_cols = [c for c in df.columns if c.startswith("acc")]
        if not acc_cols:
            raise ValueError("No accuracy columns found. Specify --metric explicitly.")
        metric = acc_cols[0]

    # aggregate mean over other columns not grouped
    group_keys = args.group_by
    agg = df.groupby(group_keys)[metric].mean().reset_index()

    # plot
    fig, ax = plt.subplots(figsize=(8, 4))
    labels = [" | ".join(str(agg[k].iloc[i]) for k in group_keys) for i in range(len(agg))]
    ax.bar(labels, agg[metric])
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} by {'/'.join(group_keys)}")
    ax.set_ylim(0, 100)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    plt.savefig(args.output, dpi=150)
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
