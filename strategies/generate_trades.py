"""
Two rule-based baselines on the 15s candle tape, run market by market.

    momentum_flip  wait for either side to trade at >= 0.55, buy it; whenever the
                   OTHER side crosses 0.55, sell the position and roll the whole
                   proceeds into that side. Whatever is held at the bell goes to
                   resolution.

    buy_and_hold   buy one side (default Down, i.e. "No" on Up) at the open and
                   hold to resolution, no matter what.

Candle tape is quoted Up-equivalent (Up: close, Down: 1 - close, see data/README.md).
Signals read off a candle's close, fills happen at the NEXT candle's open -- no lookahead.

Fills/fees/slippage go through sim.execution via sim.evaluation.simulate_market, same as
the other models (this used to charge slippage but not Polymarket's taker fee -- fixed now,
which is why the numbers are comparable across models). Split comes from
BaselineModels.data_loader via sim.evaluation.load_universe_candles.

'dev' (train+val) is the universe to iterate on. 'test' and 'all' read the held-out
block and refuse to run without --allow-test: this script used to default to 'all'
and build that universe with allow_test hardcoded True, so every number it has
published was computed over the test split.

Writes strategies/output/markets.csv.

Usage:
    python strategies/generate_trades.py --split val --slippage 0.25
    python strategies/generate_trades.py --split dev --slippage 0.25
    python strategies/generate_trades.py --split test --allow-test   # once, at the end
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pandas as pd

from sim.evaluation import (
    LAST_INDEX,
    load_universe_candles,
    results_to_frame,
    simulate_market,
)
from sim.execution import BUY_DOWN, BUY_UP, ExecutionConfig, HOLD, Side
from sim.metrics import write_markets

OUT_DIR = os.path.join(ROOT, "strategies/output")


def side_price(up_equiv_price, side):
    """Price of one leg given the Up-equivalent quote."""
    return up_equiv_price if side == "Up" else 1.0 - up_equiv_price


def make_momentum_flip(threshold):
    # buy whichever side first trades at >= threshold, flip when the other side does
    # (Portfolio.apply closes the held leg first so BUY_UP/BUY_DOWN handles the flip in one call)
    def decide(row, portfolio, i):
        up = row.close
        if portfolio.side is Side.FLAT:
            if up >= threshold:
                return BUY_UP
            if (1.0 - up) >= threshold:
                return BUY_DOWN
            return HOLD
        if portfolio.side is Side.UP:
            return BUY_DOWN if (1.0 - up) >= threshold else HOLD
        return BUY_UP if up >= threshold else HOLD

    return decide


def make_buy_and_hold(side):
    """Buy ``side`` at candle 0's open and never trade again."""

    def decide(row, portfolio, i):
        if i == 0:
            return BUY_UP if side == "Up" else BUY_DOWN
        return HOLD

    return decide


def load_candles(split, days=None, *, allow_test=False):
    """Candles for one split or universe. 'dev' is train+val; 'all' adds test.

    This used to build the "all" universe itself with allow_test hardcoded to
    True, and "all" was the default -- so the ordinary invocation read the
    held-out block. The universe loader now owns that decision and refuses test
    unless the caller's own --allow-test says otherwise.
    """
    df = load_universe_candles(split, allow_test=allow_test)
    if days:
        keep_slugs = set(
            df.drop_duplicates("event_slug").sort_values("start_ts").tail(
                days * 96
            ).event_slug
        )
        df = df[df.event_slug.isin(keep_slugs)]
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--split",
        default="val",
        choices=["train", "val", "test", "dev", "all"],
        help="market universe: a canonical split, 'dev' for train+val, or 'all' "
        "for the whole tape. 'test' and 'all' need --allow-test",
    )
    ap.add_argument(
        "--allow-test",
        action="store_true",
        help="permit reading the held-out block. The paper budgets exactly one "
        "such run, at the very end",
    )
    ap.add_argument("--days", type=int, help="use only the last N days within the split")
    ap.add_argument("--threshold", type=float, default=0.55)
    ap.add_argument("--stake", type=float, default=100.0, help="bankroll per market")
    ap.add_argument(
        "--slippage",
        type=float,
        default=0.25,
        # Was 0.0, while this very help string said the project default is 0.25
        # and every other track defaulted to 0.25. Running the documented
        # command therefore priced this track's fills differently from the
        # models it is tabled against -- and not by a rounding margin:
        # momentum_flip reads -77/1k at 0.0 against -273/1k at 0.25, and its
        # gross edge flips sign, +48/1k to -156/1k. The comparison's whole
        # claim is "identical markets under identical costs".
        help="adverse fill as a fraction of the candle's high-low range "
        "(sim.execution.ExecutionConfig.slippage_frac; project default is 0.25)",
    )
    ap.add_argument("--hold-side", default="Down", choices=["Up", "Down"])
    ap.add_argument("--out-dir", default=OUT_DIR)
    args = ap.parse_args()

    if args.split in ("test", "all") and args.allow_test:
        print(
            f"NOTE: --split {args.split} reads the HELD-OUT test block. The paper "
            "budgets one such run, at the end. Use --split val or dev to iterate."
        )

    try:
        df = load_candles(args.split, args.days, allow_test=args.allow_test)
    except PermissionError as exc:
        raise SystemExit(f"refusing to run: {exc}") from None
    config = ExecutionConfig(slippage_frac=args.slippage, stake_dollars=args.stake)

    strategies = {
        "momentum_flip": make_momentum_flip(args.threshold),
        f"buy_and_hold_{args.hold_side.lower()}": make_buy_and_hold(args.hold_side),
    }

    results, fills, skipped = [], [], 0
    for slug, ep in df.groupby("event_slug", sort=False):
        if len(ep) != LAST_INDEX + 1 or ep.close.isna().any() or pd.isna(ep.winner.iloc[0]):
            skipped += 1
            continue
        for name, decide in strategies.items():
            r = simulate_market(ep, decide, strategy=name, split=args.split, config=config)
            results.append(r)
            for t in r.portfolio.trades:
                fills.append(
                    dict(
                        strategy=name,
                        event_slug=slug,
                        candle_index=t.candle_index,
                        action=t.action,
                        side=t.side.name,
                        shares=t.shares,
                        price=t.price,
                        fee=t.fee,
                        cash_delta=t.cash_delta,
                    )
                )

    mk = results_to_frame(results)
    os.makedirs(args.out_dir, exist_ok=True)
    mk_path = os.path.join(args.out_dir, "markets.csv")
    kept = write_markets(mk_path, mk, args.split, slippage_frac=config.slippage_frac)
    fills_path = os.path.join(args.out_dir, f"fills_{args.split}.csv")
    pd.DataFrame(fills).to_csv(fills_path, index=False)
    print(f"{len(mk):>7,} rows -> {mk_path}"
          + (f"  (kept {kept:,} rows from other splits)" if kept else ""))
    print(f"{len(fills):>7,} rows -> {fills_path}")

    n_markets = df.event_slug.nunique() - skipped
    print(
        f"\n{n_markets:,} markets ({skipped:,} skipped) · split {args.split} · "
        f"threshold {args.threshold} · stake ${args.stake:g} · "
        f"slippage_frac {args.slippage}"
    )


if __name__ == "__main__":
    main()
