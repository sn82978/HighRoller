"""Histograms of the per-run metrics a Q-learning sweep writes.

One figure per metric per split, showing the spread across the sweep's seeds.
This is what turns "the agent loses $6.75" into "the agent loses $6.75 +/- x
across N runs", which is the only honest way to report a single run's number.

    python QLearning/analysis.py                     # every family in metrics/
    python QLearning/analysis.py --family qlearning

Reads QLearning/metrics/<family>_<split>.csv, writes
QLearning/metrics/figs/<family>/<Split>_<metric>.png.

Previously this hardcoded `if '0.8T_0.2E' not in metrics_csv: continue`, so it
silently produced nothing for any other model name, and it mapped the split
labels 'eval'/'training' -- which the writer never emitted -- onto figure names,
leaving the actual figures named after whatever the filename happened to end
with. The progress report cites `Figs/7T2E/Evaluation_*.png` paths that do not
exist in the repo under any name.
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METRICS_DIR = os.path.join(ROOT, "QLearning/metrics")
FIGS_DIR = os.path.join(METRICS_DIR, "figs")

#: Identifiers and constants, not outcomes -- a histogram of these says nothing.
EXCLUDE_COLS = {"iteration", "seed", "n_markets", "markets"}

#: Split key -> the word that goes in the figure title and filename.
SPLIT_LABELS = {"train": "Training", "val": "Validation", "test": "Test"}


def make_histograms(df, split, out_dir, family):
    label = SPLIT_LABELS.get(split, split.title())
    numeric = [
        c for c in df.select_dtypes(include=["number"]).columns
        if c not in EXCLUDE_COLS and df[c].notna().any()
    ]
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for col in numeric:
        values = df[col].dropna()
        plt.figure(figsize=(8, 5))
        plt.hist(values, bins=min(15, max(3, len(values) // 2)),
                 color="skyblue", edgecolor="black", alpha=0.7)
        # The mean is the number that ends up quoted in the writeup; show where
        # it sits in the spread rather than letting it stand on its own.
        plt.axvline(values.mean(), color="crimson", linestyle="--", linewidth=1.5,
                    label=f"mean {values.mean():,.4g}")
        title = col.replace("_", " ").title()
        plt.title(f"{title} across {len(values)} runs ({label})",
                  fontsize=13, fontweight="bold", pad=12)
        plt.xlabel(title, fontsize=11)
        plt.ylabel("Runs", fontsize=11)
        plt.legend(fontsize=9)
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        path = os.path.join(out_dir, f"{label}_{col}.png")
        plt.savefig(path, dpi=200)
        plt.close()
        written.append(path)
    return written


def summarise(df, family, split):
    """The mean +/- sd table the writeup should quote instead of one run."""
    numeric = [c for c in df.select_dtypes(include=["number"]).columns if c not in EXCLUDE_COLS]
    out = pd.DataFrame({
        "mean": df[numeric].mean(),
        "sd": df[numeric].std(ddof=1),
        "min": df[numeric].min(),
        "max": df[numeric].max(),
    })
    out.insert(0, "runs", len(df))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default=None, help="model family; default every one found")
    ap.add_argument("--metrics-dir", default=METRICS_DIR)
    ap.add_argument("--figs-dir", default=FIGS_DIR)
    args = ap.parse_args()

    if not os.path.isdir(args.metrics_dir):
        raise SystemExit(f"no metrics directory at {args.metrics_dir} -- run training.py first")

    found = 0
    for name in sorted(os.listdir(args.metrics_dir)):
        if not name.endswith(".csv"):
            continue
        stem = name[:-4]
        family, _, split = stem.rpartition("_")
        if not family or split not in SPLIT_LABELS:
            print(f"  [skip] {name}: not a <family>_<split>.csv")
            continue
        if args.family and family != args.family:
            continue

        df = pd.read_csv(os.path.join(args.metrics_dir, name))
        if df.empty:
            print(f"  [skip] {name}: no rows")
            continue
        found += 1
        out_dir = os.path.join(args.figs_dir, family)
        written = make_histograms(df, split, out_dir, family)
        print(f"  [ok]   {name}: {len(df)} runs -> {len(written)} figures in {out_dir}")

        table = summarise(df, family, split)
        table_path = os.path.join(out_dir, f"{SPLIT_LABELS[split]}_summary.csv")
        table.to_csv(table_path)
        print(f"         summary -> {table_path}")

    if not found:
        raise SystemExit(
            f"no <family>_<split>.csv files matched in {args.metrics_dir}"
            + (f" for family {args.family!r}" if args.family else "")
        )


if __name__ == "__main__":
    main()
