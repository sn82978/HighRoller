"""
Re-runs both strategies across a range of adverse-fill assumptions.

Zero-slippage alone is misleading -- these books quote wide (median candle high-low
is 0.04) so the real question is how much execution cost the edge survives, not
whether there's an edge at mid. slippage_frac is a fraction of the candle's
high-low range, same knob the other models use.

    python strategies/sweep_slippage.py
    python strategies/sweep_slippage.py --slippages 0 0.25 0.5 --split test
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd

from sim.evaluation import LAST_INDEX, results_to_frame, simulate_market
from sim.execution import ExecutionConfig
from generate_trades import OUT_DIR, load_candles, make_buy_and_hold, make_momentum_flip


def run(df, slippage_frac, threshold, stake, hold_side, split):
    config = ExecutionConfig(slippage_frac=slippage_frac, stake_dollars=stake)
    strategies = {
        "momentum_flip": make_momentum_flip(threshold),
        f"buy_and_hold_{hold_side.lower()}": make_buy_and_hold(hold_side),
    }
    results = []
    for slug, ep in df.groupby("event_slug", sort=False):
        if len(ep) != LAST_INDEX + 1 or ep.close.isna().any() or pd.isna(ep.winner.iloc[0]):
            continue
        for name, decide in strategies.items():
            results.append(simulate_market(ep, decide, strategy=name, split=split, config=config))
    return results_to_frame(results)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="all", choices=["train", "val", "test", "all"])
    ap.add_argument("--days", type=int)
    ap.add_argument("--threshold", type=float, default=0.55)
    ap.add_argument("--stake", type=float, default=100.0)
    ap.add_argument("--hold-side", default="Down", choices=["Up", "Down"])
    ap.add_argument(
        "--slippages",
        type=float,
        nargs="+",
        default=[0.0, 0.1, 0.25, 0.5, 0.75, 1.0],
        help="slippage_frac values to sweep (fraction of candle high-low range)",
    )
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "slippage_sweep.csv"))
    args = ap.parse_args()

    df = load_candles(args.split, args.days)
    out = []
    for s in args.slippages:
        mk = run(df, s, args.threshold, args.stake, args.hold_side, args.split)
        for name, g in mk.groupby("strategy"):
            deployed = float(g.stake_deployed.sum())
            traded = g[g.n_trades > 0]
            out.append(
                dict(
                    slippage_frac=s,
                    strategy=name,
                    total_pnl=g.pnl.sum(),
                    avg_return_pct=float(np.mean(
                        np.where(g.stake_deployed > 0,
                                 g.pnl / g.stake_deployed.where(g.stake_deployed > 0, 1.0),
                                 0.0))) * 100,
                    win_rate_pct=(traded.pnl > 0).mean() * 100 if len(traded) else float("nan"),
                    pnl_per_1k_deployed=g.pnl.sum() / deployed * 1000 if deployed else 0.0,
                )
            )
        print(f"  slippage_frac {s:<6} done")

    res = pd.DataFrame(out)
    print("\nTOTAL P&L")
    print(
        res.pivot(index="slippage_frac", columns="strategy", values="total_pnl").to_string(
            float_format=lambda x: f"{x:,.0f}"
        )
    )
    print("\nAVG RETURN PER MARKET (%)")
    print(
        res.pivot(index="slippage_frac", columns="strategy", values="avg_return_pct").to_string(
            float_format=lambda x: f"{x:,.2f}"
        )
    )
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    res.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
