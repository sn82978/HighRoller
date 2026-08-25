"""
Scores the markets.csv written by generate_trades.py and prints the metrics.

    python strategies/analyze_trades.py
    python strategies/analyze_trades.py --in-dir strategies/output --top 15

Headline numbers (ROI, Sharpe, t-stat, drawdown, CI) come from sim.evaluation.score(),
same as the other models. On top of that this adds a compounded-equity view and a
best/worst market listing.

compounded = a fixed fraction of bankroll (10% default) per market. Rolling the FULL
bankroll each time is guaranteed ruin (one loss = -100%), so treat the growth number
as a shape, not a real forecast.

Writes summary.csv and equity_curve.csv.
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd

from sim.evaluation import score

DEFAULT_IN = os.path.join(ROOT, "strategies/output")


def max_drawdown_pct(equity):
    """Peak-to-trough as a fraction of the running peak (compounded view only)."""
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    i = int(np.argmin(dd))
    return float(dd[i]), i


def sparkline(values, width=64):
    blocks = "▁▂▃▄▅▆▇█"
    v = np.asarray(values, dtype=float)
    if len(v) > width:
        v = v[np.linspace(0, len(v) - 1, width).astype(int)]
    lo, hi = v.min(), v.max()
    if hi - lo < 1e-12:
        return blocks[0] * len(v)
    idx = ((v - lo) / (hi - lo) * (len(blocks) - 1)).round().astype(int)
    return "".join(blocks[i] for i in idx)


def compounded_view(mk, fraction):
    mk = mk.sort_values("start_ts").reset_index(drop=True)
    ret = mk.return_pct.to_numpy() / 100.0
    stake = float(mk.stake.iloc[0])
    equity = stake * np.cumprod(1.0 + fraction * ret)
    dd, dd_i = max_drawdown_pct(equity)
    return equity, dd, dd_i


def fmt(v, spec=",.2f"):
    import math

    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return "—" if math.isnan(v) else "∞"
    return format(v, spec) if isinstance(v, float) else f"{v:,}"


def report(s, mk, fraction):
    """Human-readable block for one strategy."""
    p = lambda label, val: print(f"  {label:<32}{val:>18}")
    print(f"\n{'=' * 72}\n  {s['strategy'].upper()}  (split: {s['split']})\n{'=' * 72}")

    print("\n  ACTIVITY")
    p("markets in sample", fmt(s["markets"]))
    p("markets traded", f"{fmt(s['markets_traded'])}  ({s['participation_%']:.1f}%)")

    print("\n  P&L  (flat $%.0f per market)" % mk.stake.iloc[0])
    p("total staked", "$" + fmt(s["total_staked"]))
    p("total P&L", "$" + fmt(s["total_pnl"]))
    p("ROI on total staked", f"{s['roi_on_stake_%']:.3f}%")
    p("avg P&L per market", "$" + fmt(s["avg_pnl_per_market"]))
    p("median P&L per market", "$" + fmt(s["median_pnl"]))
    p("avg return per market", f"{s['avg_return_%']:.3f}%")
    p(
        "  95% CI (bootstrap)",
        f"[{s['return_ci95_low_%']:.3f}%, {s['return_ci95_high_%']:.3f}%]",
    )

    print("\n  QUALITY")
    p("win rate (market P&L > 0)", f"{s['win_rate_%']:.2f}%")
    p("profit factor", fmt(s["profit_factor"], ".3f"))
    p("Sharpe per market", fmt(s["sharpe_per_market"], ".4f"))
    p("Sharpe annualized", fmt(s["sharpe_annualized"], ".2f"))
    p("t-stat vs zero edge", fmt(s["t_stat"], ".2f"))

    print("\n  EQUITY")
    p("cumulative P&L", "$" + fmt(s["total_pnl"]))
    p("max drawdown", "$" + fmt(s["max_drawdown_$"]))
    p("  in units of one stake", f"{abs(s['max_drawdown_stakes']):.1f} stakes")

    equity, comp_dd, _ = compounded_view(mk, fraction)
    p(f"compounded @ {fraction:.0%} of bankroll", "$" + fmt(equity[-1]))
    p("  growth", f"{equity[-1] / mk.stake.iloc[0]:.4g}x")
    p("  max drawdown", f"{comp_dd * 100:.2f}%")

    cum_pnl = np.cumsum(mk.sort_values('start_ts').pnl.to_numpy())
    print(f"\n  cumulative P&L   {sparkline(cum_pnl)}")
    return equity, cum_pnl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default=DEFAULT_IN)
    ap.add_argument("--top", type=int, default=10, help="best/worst markets to list")
    ap.add_argument(
        "--fraction",
        type=float,
        default=0.10,
        help="bankroll fraction for the compounded view (1.0 is ruin — see docstring)",
    )
    args = ap.parse_args()

    mk_path = os.path.join(args.in_dir, "markets.csv")
    if not os.path.exists(mk_path):
        raise SystemExit(f"missing {mk_path} — run generate_trades.py first")
    markets = pd.read_csv(mk_path)

    print(
        f"\n{markets.event_slug.nunique():,} markets · "
        f"split: {markets.split.iloc[0]!r} · "
        f"{len(markets.strategy.unique())} strategies"
    )

    rows, curves = [], {}
    for name in markets.strategy.unique():
        mk = markets[markets.strategy == name]
        s = score(mk)
        equity, cum_pnl = report(s, mk, args.fraction)
        curves[name] = equity
        curves[f"{name}_cum_pnl"] = cum_pnl
        rows.append(s)

    summary = pd.DataFrame(rows).set_index("strategy").T
    print(f"\n{'=' * 72}\n  SIDE BY SIDE\n{'=' * 72}")
    headline = [
        "markets_traded",
        "total_pnl",
        "roi_on_stake_%",
        "avg_return_%",
        "win_rate_%",
        "profit_factor",
        "sharpe_per_market",
        "sharpe_annualized",
        "t_stat",
        "max_drawdown_$",
    ]
    print(summary.loc[headline].to_string(float_format=lambda x: f"{x:,.4f}"))

    for name in markets.strategy.unique():
        mk = markets[markets.strategy == name].sort_values("pnl")
        cols = ["event_slug", "winner", "pnl"]
        print(f"\n  {name} — worst {args.top}")
        print(mk.head(args.top)[cols].to_string(index=False))
        print(f"\n  {name} — best {args.top}")
        print(mk.tail(args.top)[cols].to_string(index=False))

    summary.to_csv(os.path.join(args.in_dir, "summary.csv"))
    eq = pd.DataFrame(curves)
    eq.insert(
        0,
        "start_ts",
        markets[markets.strategy == markets.strategy.iloc[0]]
        .sort_values("start_ts")
        .start_ts.to_numpy(),
    )
    eq.to_csv(os.path.join(args.in_dir, "equity_curve.csv"), index=False)
    print(f"\nwrote {args.in_dir}/summary.csv and {args.in_dir}/equity_curve.csv\n")


if __name__ == "__main__":
    main()
