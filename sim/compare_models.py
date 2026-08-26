"""
Side-by-side comparison across every model in the repo.

Reads each model's markets.csv, restricts everyone to the same split's markets,
scores with sim.evaluation.score(), writes comparison.csv and (unless
--no-plots) a set of PNGs under comparison_figs/.

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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sim.evaluation import market_slugs, score

FIGS_DIR = os.path.join(ROOT, "comparison_figs")

#: Every model's markets.csv, keyed by where it lives.
SOURCES = {
    "strategies (momentum_flip, buy_and_hold)": os.path.join(ROOT, "strategies/output/markets.csv"),
    "BaselineModels (xgb_baseline)": os.path.join(ROOT, "BaselineModels/output/markets.csv"),
    "QLearning": os.path.join(ROOT, "QLearning/output/markets.csv"),
}


def align_to_common_markets(all_mk: pd.DataFrame):
    """Restrict every strategy to the markets they all cover.

    The tracks apply different sample cuts: the baselines drop markets whose
    tape is too short for the 16-candle feature warmup, the rule strategies drop
    any market without a complete 60-candle live window. On val that is 1 market
    against 12, so their totals sit on different denominators -- which is
    precisely the apples-to-oranges the proposal's "identical markets under
    identical fees" exists to rule out.

    Returns (aligned frame, common slugs, {strategy: n_dropped}).
    """
    per_strategy = all_mk.groupby("strategy").event_slug.apply(set)
    if not len(per_strategy):
        raise SystemExit("nothing to align")
    common = set.intersection(*per_strategy)
    if not common:
        raise SystemExit("the models share no markets -- cannot align")
    dropped = {s: len(v - common) for s, v in per_strategy.items() if v - common}
    return all_mk[all_mk.event_slug.isin(common)], common, dropped


def plot_headline_bars(summary: pd.DataFrame, split: str, figs_dir: str) -> str:
    """One PNG, one bar-chart panel per headline metric, all strategies side by side."""
    metrics = [
        ("total_pnl", "Total P&L ($)"),
        ("avg_return", "Avg return / market"),
        ("win_rate", "Win rate (traded markets)"),
        ("profit_factor", "Profit factor"),
        ("sharpe", "Sharpe (annualized)"),
        ("max_drawdown", "Max drawdown ($)"),
    ]
    strategies = list(summary.index)
    colors = plt.cm.tab10(np.linspace(0, 1, len(strategies)))

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (col, title) in zip(axes.flat, metrics):
        values = summary[col].to_numpy(dtype=float)
        bars = ax.bar(strategies, values, color=colors)
        if col == "avg_return":
            lo = (summary["avg_return"] - summary["return_ci95_lo"]).to_numpy(dtype=float)
            hi = (summary["return_ci95_hi"] - summary["avg_return"]).to_numpy(dtype=float)
            ax.errorbar(
                strategies, values, yerr=[np.abs(lo), np.abs(hi)],
                fmt="none", ecolor="black", capsize=4,
            )
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(title, fontsize=11)
        ax.tick_params(axis="x", rotation=30, labelsize=8)
        for bar, v in zip(bars, values):
            ax.annotate(
                f"{v:,.1f}", (bar.get_x() + bar.get_width() / 2, v),
                textcoords="offset points", xytext=(0, 3 if v >= 0 else -12),
                ha="center", fontsize=7,
            )

    fig.suptitle(f"Cross-model comparison — split: {split}", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(figs_dir, exist_ok=True)
    path = os.path.join(figs_dir, f"headline_bars_{split}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_equity_curves(all_mk: pd.DataFrame, split: str, figs_dir: str) -> str:
    """Cumulative P&L over time, one line per strategy, all on the same $100 stake."""
    fig, ax = plt.subplots(figsize=(11, 6))
    for strat in all_mk.strategy.unique():
        sub = all_mk[all_mk.strategy == strat].sort_values("start_ts")
        cum_pnl = np.cumsum(sub.pnl.to_numpy())
        ax.plot(range(len(cum_pnl)), cum_pnl, label=strat, linewidth=1.5)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Markets, in chronological order")
    ax.set_ylabel("Cumulative P&L ($)")
    ax.set_title(f"Cumulative P&L by market order — split: {split}", fontsize=12)
    ax.legend()
    fig.tight_layout()
    os.makedirs(figs_dir, exist_ok=True)
    path = os.path.join(figs_dir, f"equity_curves_{split}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--out", default=os.path.join(ROOT, "comparison.csv"))
    ap.add_argument("--no-plots", action="store_true", help="skip writing comparison_figs/")
    ap.add_argument(
        "--no-align", action="store_true",
        help="score each model on its own market set instead of the shared "
        "intersection. Totals are then not comparable across models",
    )
    ap.add_argument("--figs-dir", default=FIGS_DIR)
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

    if not args.no_align:
        all_mk, common, dropped = align_to_common_markets(all_mk)
        if dropped:
            print(f"\naligning to the {len(common):,} markets every model covers:")
            for s, n in sorted(dropped.items()):
                print(f"  {s}: dropping {n} market(s) the others did not evaluate")

    rows = [score(all_mk[all_mk.strategy == s]) for s in all_mk.strategy.unique()]
    summary = pd.DataFrame(rows).set_index("strategy")

    print(f"\n{'=' * 100}\n  CROSS-MODEL COMPARISON  (split: {args.split})\n{'=' * 100}")
    cols = [
        "n_markets",
        "n_traded",
        "total_pnl",
        "pnl_per_1k_deployed",
        "gross_pnl_per_1k_deployed",
        "fee_per_1k_deployed",
        "avg_return",
        "return_ci95_lo",
        "return_ci95_hi",
        "win_rate",
        "profit_factor",
        "sharpe",
        "t_stat",
        "max_drawdown",
        "turnover",
        "avg_holding_candles",
    ]
    print(summary[cols].to_string(float_format=lambda x: f"{x:,.3f}"))

    summary.to_csv(args.out)
    print(f"\nwrote {args.out}")

    if not args.no_plots:
        bars_path = plot_headline_bars(summary, args.split, args.figs_dir)
        curves_path = plot_equity_curves(all_mk, args.split, args.figs_dir)
        print(f"wrote {bars_path}")
        print(f"wrote {curves_path}")


if __name__ == "__main__":
    main()
