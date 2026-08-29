"""
Puts every model in the repo side by side.

Reads each model's markets.csv, cuts everyone down to the same set of markets,
scores them all with sim.evaluation.score(), then writes comparison.csv and
(unless you pass --no-plots) some PNGs into comparison_figs/.

    python strategies/generate_trades.py --split val
    python BaselineModels/run_baselines.py --split val
    python QLearning/training.py            # writes QLearning/output/markets.csv too
    python sim/compare_models.py --split val

For the test split, use the replay script instead of training again. If you
retrain you get 30 different agents, and then val vs test isn't a
generalisation gap anymore, it's just two unrelated runs:

    python QLearning/evaluate_split.py --split test --allow-test

Every track has to have run at the same --slippage. Each market row saves which
cost model it used, and this script refuses to build a table if they don't match.

Models you haven't run for --split yet just get skipped, with a note printed.
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
    """'qlearning_seed07' -> 'qlearning'. Returns None for a one-off strategy."""
    m = SEED_SUFFIX.match(str(strategy))
    return m.group("family") if m else None


#: These metrics are ratios of sums. Averaging them across seeds gives you a
#: mean of ratios, which isn't a ratio, and it blows up when a seed happens to
#: have almost no losing markets. On test, 3 of the 30 seeds had profit_factor
#: 207, 54 and 13, which dragged the mean to 9.85 -- the median was 0.47. We
#: report a median next to the mean so nobody quotes the mean by itself.
RATIO_METRICS = ("profit_factor", "fee_fraction_gross_pnl")


def score_seed_family(mk: pd.DataFrame, family: str) -> tuple[dict, dict]:
    """Mean and sd of each metric ACROSS seeds. Not the metrics of an averaged run.

    This distinction is easy to get wrong. If you averaged the 30 agents'
    per-market PnL first and then scored that once, you'd be reporting the
    Sharpe of an ensemble, not of the policy. Averaging 30 independent runs
    cancels out most of the variance, so the denominator shrinks and Sharpe
    jumps by about sqrt(30) -- for a policy nobody could actually run, since you
    only get one agent. Scoring each seed separately and then averaging answers
    the question we actually care about: what does one run of this thing do, and
    how much does it vary?
    """
    per_seed = [
        score(mk[mk.strategy == s]) for s in sorted(mk.strategy.unique())
    ]
    keys = [k for k in per_seed[0] if isinstance(per_seed[0][k], (int, float))]

    def across(fn):
        # A metric can be NaN for all 30 seeds -- fee_fraction_gross_pnl is
        # undefined whenever gross PnL <= 0, and on a bad sweep that's every
        # run. numpy's nan functions do return NaN here, but they print a
        # RuntimeWarning while doing it, which shows up in the middle of the
        # output and looks like something crashed. Same answer, less noise.
        def one(k):
            vals = np.asarray([r[k] for r in per_seed], dtype=float)
            return float(fn(vals)) if np.isfinite(vals).any() else float("nan")

        return {k: one(k) for k in keys}

    mean = across(np.nanmean)
    stats = {
        "mean": mean,
        # ddof=1 needs at least two finite values. One seed has no spread at
        # all, which isn't the same as a spread of zero, and numpy just warns
        # instead of telling you that.
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
    """Refuse to build a table out of strategies that used different costs.

    The whole claim of this comparison is "same markets, same fees". Nothing
    checked that until we started recording the cost model on each row -- you
    just had to remember to launch every track with matching flags. We didn't:
    generate_trades.py defaulted to --slippage 0.0 while everything else
    defaulted to 0.25, so running each track's own documented command priced the
    rule strategies differently from the models next to them in the table. That
    moved momentum_flip from -77 to -273 per $1k and flipped the sign of its
    gross edge, and nothing in the output hinted at it.

    Returns the shared slippage_frac, or None if the column isn't there.
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
    """Cut every strategy down to the markets all of them cover.

    The tracks throw away different markets: the baselines drop anything whose
    tape is too short for the 16-candle feature warmup, and the rule strategies
    drop anything without a full 60-candle live window. On val that's 1 market
    vs 12, so their totals end up sitting on different denominators. That's
    exactly the apples-to-oranges the "same markets, same fees" requirement is
    supposed to prevent.

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
    """Use enough decimals that you can tell the bar apart from zero.

    A hardcoded '%.1f' printed '-0.0' on four of the six avg_return bars, which
    looks like "no effect" for numbers that actually span a factor of 40.
    """
    if not np.isfinite(v):
        return "n/a"
    if v == 0:
        return "0"
    return f"{v:,.1f}" if abs(v) >= 1 else f"{v:,.4f}".rstrip("0")


def plot_headline_bars(
    summary: pd.DataFrame, split: str, figs_dir: str, spreads: dict | None = None
) -> str:
    """One PNG with a bar-chart panel per headline metric, all strategies together.

    The collapsed sweep row gets an error bar of +/- one sd across its seeds.
    It's a mean of 30 runs whose sd is bigger than the mean on most of these
    metrics, so drawing it as a plain bar would claim way more precision than
    we actually have.
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

        # Single-run rows get the bootstrap CI, sweep rows get the across-seed
        # sd. NaN when neither applies, because matplotlib skips NaN points --
        # a 0 would draw a flat cap on top of the bar that looks like a tiny
        # error bar.
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
    """Cumulative P&L over time, one line per strategy, same $100 stake for all.

    A seeded sweep gets drawn as a band -- each seed faint, the mean bold --
    instead of 30 separate labelled lines. The naive version put 35 entries in
    the legend, which ran off the bottom of the image and went through the
    colour cycle three times, so you couldn't tell the seeds apart from each
    other OR from the five policies the plot is actually supposed to compare.
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
        # Each market should show up once per strategy. Writers that run at the
        # same time (or twice in a row) append instead of replacing, and a
        # duplicated market gets counted twice in every total here. Nothing else
        # would catch it.
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

    # A seeded sweep is one policy measured 30 times, not 30 different policies.
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

    # The collapsed sweep row is just a mean with no spread attached, and a
    # mean over 30 runs whose n_traded goes from 0 to 598 doesn't tell you much
    # by itself. It's worse for the ratio metrics: on test the mean
    # profit_factor is 9.85 because three seeds hit 207, 54 and 13, but the
    # median is 0.47. So we write out the whole distribution, not just the sd.
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
