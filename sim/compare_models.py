"""
Side-by-side comparison across every model in the repo.

Reads each model's markets.csv, restricts everyone to the same split's markets,
scores with sim.evaluation.score(), writes comparison.csv.

    python strategies/generate_trades.py --split val
    python BaselineModels/xgb_baseline.py --split val
    python QLearning/training.py            # writes QLearning/output/markets.csv too
    python sim/compare_models.py --split val

Models that haven't been run for --split yet just get skipped (with a printed note).
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(ROOT)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pandas as pd

from sim.evaluation import market_slugs, score

#: Every model's markets.csv, keyed by where it lives.
SOURCES = {
    "strategies (momentum_flip, buy_and_hold)": os.path.join(ROOT, "strategies/output/markets.csv"),
    "BaselineModels (xgb_baseline)": os.path.join(ROOT, "BaselineModels/output/markets.csv"),
    "QLearning": os.path.join(ROOT, "QLearning/output/markets.csv"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--out", default=os.path.join(ROOT, "comparison.csv"))
    args = ap.parse_args()

    if args.split == "test" and not args.allow_test:
        raise SystemExit("refusing --split test without --allow-test")

    expected = market_slugs(args.split, allow_test=args.allow_test)
    print(f"canonical {args.split!r} split: {len(expected):,} markets\n")

    frames = []
    for label, path in SOURCES.items():
        if not os.path.exists(path):
            print(f"  [skip] {label}: no markets.csv at {path}")
            continue
        df = pd.read_csv(path)
        df = df[df.split == args.split]
        if df.empty:
            print(f"  [skip] {label}: markets.csv has no rows for split={args.split!r}")
            continue
        for strat in df.strategy.unique():
            sub = df[df.strategy == strat]
            covered = set(sub.event_slug) & expected
            off_split = set(sub.event_slug) - expected
            if off_split:
                print(
                    f"  [warn] {strat}: {len(off_split)} rows reference markets outside "
                    f"the canonical {args.split!r} split -- dropping them"
                )
            sub = sub[sub.event_slug.isin(expected)]
            missing = len(expected) - len(covered)
            note = f" ({missing} of {len(expected)} canonical markets not evaluated)" if missing else ""
            print(f"  [ok]   {strat}: {len(sub):,} markets scored{note}")
            frames.append(sub)

    if not frames:
        raise SystemExit("nothing to compare -- run at least one model's script first")

    all_mk = pd.concat(frames, ignore_index=True)
    rows = [score(all_mk[all_mk.strategy == s]) for s in all_mk.strategy.unique()]
    summary = pd.DataFrame(rows).set_index("strategy")

    print(f"\n{'=' * 100}\n  CROSS-MODEL COMPARISON  (split: {args.split})\n{'=' * 100}")
    cols = [
        "markets",
        "markets_traded",
        "total_pnl",
        "roi_on_stake_%",
        "avg_return_%",
        "return_ci95_low_%",
        "return_ci95_high_%",
        "win_rate_%",
        "profit_factor",
        "sharpe_annualized",
        "t_stat",
        "max_drawdown_$",
    ]
    print(summary[cols].to_string(float_format=lambda x: f"{x:,.3f}"))

    summary.to_csv(args.out)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
