"""
Side-by-side comparison across every model in the repo.

Reads each model's markets.csv, restricts everyone to the same split's markets,
scores with sim.evaluation.score(), writes comparison.csv and (unless
--no-plots) a set of PNGs under comparison_figs/.

    python strategies/generate_trades.py --split val
    python BaselineModels/run_baselines.py --split val
    python QLearning/training.py            # writes QLearning/output/markets.csv too
    python sim/compare_models.py --split val

For the held-out split, swap the RL step for a replay of the trained tables --
retraining would produce different agents, and the val/test pair would stop
being a generalisation gap:

    python QLearning/evaluate_split.py --split test --allow-test

Every track must have run at the same --slippage; each market row records the
cost model it was simulated under and this script refuses to table rows that
disagree.

Models that haven't been run for --split yet just get skipped (with a printed note).
"""

import argparse
import os
import re
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
from sim.metrics import COST_MODEL_FIELD

FIGS_DIR = os.path.join(ROOT, "comparison_figs")

#: Every model's markets.csv, keyed by where it lives.
SOURCES = {
    "strategies (momentum_flip, buy_and_hold)": os.path.join(ROOT, "strategies/output/markets.csv"),
    "BaselineModels (xgb_baseline)": os.path.join(ROOT, "BaselineModels/output/markets.csv"),
    "QLearning": os.path.join(ROOT, "QLearning/output/markets.csv"),
}


SEED_SUFFIX = re.compile(r"^(?P<family>.+)_seed(?P<seed>\d+)$")


def seed_family(strategy: str):
    """'qlearning_seed07' -> 'qlearning'; None for a single-run strategy."""
    m = SEED_SUFFIX.match(str(strategy))
    return m.group("family") if m else None


#: Metrics that are ratios of sums. A mean of these across seeds is a mean of
#: ratios, which is not a ratio, and is unbounded when a seed happens to have
#: almost no losing markets: on test, three of the thirty seeds scored
#: profit_factor 207, 54 and 13, pulling the mean to 9.85 while the median was
#: 0.47. Reported with a median so the mean cannot be quoted on its own.
RATIO_METRICS = ("profit_factor", "fee_fraction_gross_pnl")


def score_seed_family(mk: pd.DataFrame, family: str) -> tuple[dict, dict]:
    """Mean and sd of each metric ACROSS seeds -- not the metrics of a mean run.

    The distinction matters. Averaging the 30 agents' per-market PnL first and
    scoring that once would report the Sharpe of an ensemble, not of the policy:
    averaging 30 independent runs cancels most of their variance, so the
    denominator collapses and Sharpe inflates by roughly sqrt(30) for a policy
    nobody could actually run. Scoring each seed and averaging the results
    answers the question the report asks -- what does one run of this agent do,
    and how much does that vary.
    """
    per_seed = [
        score(mk[mk.strategy == s]) for s in sorted(mk.strategy.unique())
    ]
    keys = [k for k in per_seed[0] if isinstance(per_seed[0][k], (int, float))]

    def across(fn):
        # A metric can be NaN for every seed -- fee_fraction_gross_pnl is
        # undefined wherever gross PnL is <= 0, and on a bad sweep that is
        # all 30 runs. numpy's nan-aware reducers answer NaN but emit a
        # RuntimeWarning doing it, which lands in the middle of the run's
        # output looking like a failure. The answer is still NaN; say so
        # without the noise.
        def one(k):
            vals = np.asarray([r[k] for r in per_seed], dtype=float)
            return float(fn(vals)) if np.isfinite(vals).any() else float("nan")

        return {k: one(k) for k in keys}

    mean = across(np.nanmean)
    stats = {
        "mean": mean,
        # ddof=1 needs two finite values; one seed has no spread, not a
        # spread of zero, and numpy warns rather than saying so.
        "sd": across(lambda v: np.nanstd(v, ddof=1) if np.isfinite(v).sum() > 1
                     else float("nan")),
        "median": across(np.nanmedian),
        "min": across(np.nanmin),
        "max": across(np.nanmax),
    }
    mean["strategy"] = f"{family} (mean of {len(per_seed)})"
    mean["split"] = per_seed[0]["split"]
    mean["n_seeds"] = len(per_seed)
    return mean, stats


def check_one_cost_model(all_mk: pd.DataFrame) -> float | None:
    """Refuse to table strategies that were simulated under different costs.

    "Identical markets under identical fees" is the comparison's whole claim,
    and until each row carried the cost model it was simulated under, nothing
    checked it -- the tracks simply had to be launched with matching flags.
    They were not: generate_trades.py defaulted --slippage 0.0 while every
    other track defaulted 0.25, so running each track's documented command
    priced the rule strategies' fills differently from the models they were
    tabled against. It moved momentum_flip from -77 to -273 per $1k and flipped
    the sign of its gross edge, and the table gave no sign of it.

    Returns the shared slippage_frac, or None when the column is absent.
    """
    if COST_MODEL_FIELD not in all_mk.columns:
        print(
            f"\n  [warn] no {COST_MODEL_FIELD!r} column -- these markets.csv files "
            "predate cost-model stamping, so nothing here can verify the tracks "
            "ran under the same costs. Regenerate them."
        )
        return None

    by_strategy = all_mk.groupby("strategy")[COST_MODEL_FIELD].agg(["min", "max"])
    if by_strategy.isna().any().any():
        missing = by_strategy[by_strategy.isna().any(axis=1)].index.tolist()
        raise SystemExit(
            f"no {COST_MODEL_FIELD} recorded for: {missing}. Regenerate those "
            "tracks -- an unlabelled cost model cannot be compared."
        )
    if not np.isclose(by_strategy["min"], by_strategy["max"]).all():
        raise SystemExit(
            f"a strategy has rows at more than one {COST_MODEL_FIELD}:\n"
            f"{by_strategy.to_string()}\nRegenerate that track."
        )

    values = sorted(by_strategy["min"].round(10).unique())
    if len(values) > 1:
        raise SystemExit(
            f"the tracks were simulated under different cost models "
            f"({COST_MODEL_FIELD} = {values}):\n{by_strategy['min'].to_string()}\n"
            "Re-run them all at the same --slippage before comparing. This table "
            "claims identical fees; at different slippage it would not have them."
        )
    print(f"  cost model: {COST_MODEL_FIELD}={values[0]} across every strategy")
    return float(values[0])


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


def _bar_label(v: float) -> str:
    """Enough decimals to distinguish the bar from zero.

    A fixed '%.1f' printed '-0.0' for four of the six avg_return bars, which
    reads as "no effect" for numbers spanning a factor of 40.
    """
    if not np.isfinite(v):
        return "n/a"
    if v == 0:
        return "0"
    return f"{v:,.1f}" if abs(v) >= 1 else f"{v:,.4f}".rstrip("0")


def plot_headline_bars(
    summary: pd.DataFrame, split: str, figs_dir: str, spreads: dict | None = None
) -> str:
    """One PNG, one bar-chart panel per headline metric, all strategies side by side.

    A collapsed sweep row gets an error bar of +/- one sd across its seeds. It
    is a mean of 30 runs whose sd exceeds the mean on most of these metrics, and
    drawn as a bare bar it claims a precision the sweep does not have.
    """
    spreads = spreads or {}
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

        # Bootstrap CI on the single-run rows; across-seed sd on the sweep rows.
        # NaN where neither applies -- matplotlib skips those points, whereas a
        # zero draws a flat cap across the bar top that reads as a tiny whisker.
        lo = np.full(len(strategies), np.nan)
        hi = np.full(len(strategies), np.nan)
        for i, s in enumerate(strategies):
            sd = spreads.get(s, {}).get("sd", {}).get(col, np.nan)
            if np.isfinite(sd):
                lo[i] = hi[i] = sd
            elif col == "avg_return":
                lo[i] = abs(summary.at[s, "avg_return"] - summary.at[s, "return_ci95_lo"])
                hi[i] = abs(summary.at[s, "return_ci95_hi"] - summary.at[s, "avg_return"])
        if np.isfinite(lo).any() or np.isfinite(hi).any():
            ax.errorbar(
                strategies, values, yerr=[lo, hi],
                fmt="none", ecolor="black", capsize=4,
            )

        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(title, fontsize=11)
        ax.tick_params(axis="x", rotation=30, labelsize=8)
        for bar, v in zip(bars, values):
            ax.annotate(
                _bar_label(v), (bar.get_x() + bar.get_width() / 2, v if np.isfinite(v) else 0),
                textcoords="offset points", xytext=(0, 3 if not (v < 0) else -12),
                ha="center", fontsize=7,
            )

    fig.suptitle(
        f"Cross-model comparison — split: {split}\n"
        "whiskers: bootstrap 95% CI on avg return; ± 1 sd across seeds on a collapsed sweep",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(figs_dir, exist_ok=True)
    path = os.path.join(figs_dir, f"headline_bars_{split}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_equity_curves(
    all_mk: pd.DataFrame, split: str, figs_dir: str, per_seed: bool = False
) -> str:
    """Cumulative P&L over time, one line per strategy, all on the same $100 stake.

    A seeded sweep is drawn as a band -- every seed faint, the mean bold -- not
    as one labelled line each. With 30 seeds the plain version put 35 entries in
    the legend, which ran off the bottom of the canvas and recycled the colour
    cycle three times, so the seeds were indistinguishable from each other *and*
    from the five policies the figure exists to compare.
    """
    fig, ax = plt.subplots(figsize=(11, 6))

    singles, families = [], {}
    for s in all_mk.strategy.unique():
        fam = seed_family(s) if not per_seed else None
        (families.setdefault(fam, []).append(s) if fam else singles.append(s))

    def curve(strategy: str) -> np.ndarray:
        sub = all_mk[all_mk.strategy == strategy].sort_values("start_ts")
        return np.cumsum(sub.pnl.to_numpy(dtype=float))

    for strat in singles:
        cum_pnl = curve(strat)
        ax.plot(range(len(cum_pnl)), cum_pnl, label=strat, linewidth=1.6, zorder=2)

    for fam, members in families.items():
        curves = [curve(s) for s in sorted(members)]
        n = min(len(c) for c in curves)
        stack = np.array([c[:n] for c in curves])
        for one in stack:
            ax.plot(range(n), one, color="0.55", linewidth=0.5, alpha=0.35, zorder=1)
        ax.plot(
            range(n), stack.mean(axis=0), color="black", linewidth=2.0, zorder=3,
            label=f"{fam} — mean of {len(members)} seeds (each in grey)",
        )

    ax.axhline(0, color="black", linewidth=0.8, zorder=0)
    ax.set_xlabel("Markets, in chronological order")
    ax.set_ylabel("Cumulative P&L ($)")
    ax.set_title(f"Cumulative P&L by market order — split: {split}", fontsize=12)
    ax.legend(loc="lower left", fontsize=9)
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
    ap.add_argument(
        "--per-seed", action="store_true",
        help="score every seed of a sweep as its own strategy instead of "
        "collapsing them into one mean row",
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
        # A market must appear once per strategy. Concurrent or repeated writers
        # append rather than replace, and a duplicated market is counted twice
        # by every total in the table -- silently, since nothing else notices.
        dupes = df.duplicated(subset=["strategy", "event_slug"]).sum()
        if dupes:
            raise SystemExit(
                f"{path} has {dupes:,} duplicate (strategy, market) rows for "
                f"split={args.split!r}. That file was written by more than one "
                "run. Regenerate it rather than scoring it -- every total here "
                "would double-count."
            )
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

    check_one_cost_model(all_mk)

    if not args.no_align:
        all_mk, common, dropped = align_to_common_markets(all_mk)
        if dropped:
            print(f"\naligning to the {len(common):,} markets every model covers:")
            for s, n in sorted(dropped.items()):
                print(f"  {s}: dropping {n} market(s) the others did not evaluate")

    # A seeded sweep is one policy measured 30 times, not 30 policies.
    families, singles = {}, []
    for s in all_mk.strategy.unique():
        fam = seed_family(s) if not args.per_seed else None
        (families.setdefault(fam, []).append(s) if fam else singles.append(s))

    rows, spreads = [], {}
    for s in singles:
        rows.append(score(all_mk[all_mk.strategy == s]))
    for fam, members in families.items():
        mean, stats = score_seed_family(all_mk[all_mk.strategy.isin(members)], fam)
        rows.append(mean)
        spreads[mean["strategy"]] = stats
        print(f"\ncollapsed {len(members)} seeds of {fam!r} into one row "
              f"(mean of per-seed scores; +/- is sd across seeds)")
        for metric in RATIO_METRICS:
            mu, med = stats["mean"].get(metric), stats["median"].get(metric)
            if mu is None or not np.isfinite(mu) or not np.isfinite(med):
                continue
            if abs(mu - med) > max(1.0, 2 * abs(med)):
                print(
                    f"  [warn] {fam}.{metric}: mean {mu:,.2f} but median "
                    f"{med:,.2f} (range {stats['min'][metric]:,.2f} to "
                    f"{stats['max'][metric]:,.2f}). This is a ratio of sums, so "
                    f"the mean across seeds is dominated by a few runs with "
                    f"almost no losing markets. Quote the median, or the range."
                )

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

    # A collapsed sweep row is a mean with no spread attached, and a mean of
    # 30 runs whose n_traded ranges over 0..598 is close to meaningless on its
    # own. Worse for the ratio metrics: on test the mean profit_factor is 9.85
    # because three seeds scored 207, 54 and 13, while the median is 0.47. So
    # the whole across-seed distribution is written out, not just the sd.
    if spreads:
        spread_path = re.sub(r"\.csv$", "", args.out) + "_spread.csv"
        tidy = pd.concat(
            {strat: pd.DataFrame(stats).T for strat, stats in spreads.items()},
            names=["strategy", "statistic"],
        )
        tidy.to_csv(spread_path)
        print(f"wrote {spread_path}  (mean/sd/median/min/max across seeds)")

    if not args.no_plots:
        bars_path = plot_headline_bars(summary, args.split, args.figs_dir, spreads)
        curves_path = plot_equity_curves(all_mk, args.split, args.figs_dir, args.per_seed)
        print(f"wrote {bars_path}")
        print(f"wrote {curves_path}")


if __name__ == "__main__":
    main()
