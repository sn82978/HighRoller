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
    # Windows consoles default to cp1252, which cannot encode the block glyphs
    # and raised UnicodeEncodeError mid-report. Fall back to ASCII rather than
    # crash after printing half the numbers.
    blocks = "▁▂▃▄▅▆▇█"
    try:
        "".join(blocks).encode(sys.stdout.encoding or "utf-8")
    except (UnicodeEncodeError, LookupError):
        blocks = ".:-=+*#@"
    v = np.asarray(values, dtype=float)
    if len(v) > width:
        v = v[np.linspace(0, len(v) - 1, width).astype(int)]
    lo, hi = v.min(), v.max()
    if hi - lo < 1e-12:
        return blocks[0] * len(v)
    idx = ((v - lo) / (hi - lo) * (len(blocks) - 1)).round().astype(int)
    return "".join(blocks[i] for i in idx)


def per_market_return(mk):
    """PnL over the capital that market risked; 0.0 where none was.

    Denominator is the per-market allotment, not the sum of entry notionals --
    see sim.metrics._capital_basis for why that distinction flips the sign of
    this number on a policy that re-enters.
    """
    basis = capital_basis(mk)
    pnl = mk.pnl.to_numpy(dtype=float)
    return np.where(basis > 0, pnl / np.where(basis > 0, basis, 1.0), 0.0)


def capital_basis(mk):
    deployed = mk.stake_deployed.to_numpy(dtype=float)
    if "stake" not in mk.columns:
        return deployed
    stake = pd.to_numeric(mk.stake, errors="coerce").to_numpy(dtype=float)
    return np.where(np.isfinite(stake) & (stake > 0), stake, deployed)


def typical_stake(mk):
    """The per-market bankroll, read off the markets that actually traded."""
    basis = capital_basis(mk)
    basis = basis[basis > 0]
    return float(np.median(basis)) if basis.size else 0.0


def compounded_view(mk, fraction):
    mk = mk.sort_values("start_ts").reset_index(drop=True)
    ret = per_market_return(mk)
    stake = typical_stake(mk) or 100.0
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

    stake = typical_stake(mk) or 100.0

    print("\n  ACTIVITY")
    p("markets in sample", fmt(s["n_markets"]))
    p("markets traded", f"{fmt(s['n_traded'])}  ({s['trade_rate'] * 100:.1f}%)")
    p("turnover", fmt(s["turnover"], ".3f"))
    p("avg holding (candles)", fmt(s["avg_holding_candles"], ".1f"))

    print("\n  P&L  (flat $%.0f per entry)" % stake)
    p("capital deployed", "$" + fmt(s["capital_deployed"]))
    p("total P&L", "$" + fmt(s["total_pnl"]))
    p("P&L per $1k deployed", "$" + fmt(s["pnl_per_1k_deployed"]))
    p("  gross, before fees", "$" + fmt(s["gross_pnl_per_1k_deployed"]))
    p("  fees", "$" + fmt(s["fee_per_1k_deployed"]))
    p("avg P&L per market", "$" + fmt(s["avg_pnl_per_market"]))
    p("median P&L per market", "$" + fmt(s["median_pnl"]))
    p("avg return per market", f"{s['avg_return'] * 100:.3f}%")
    p(
        "  95% CI (bootstrap)",
        f"[{s['return_ci95_lo'] * 100:.3f}%, {s['return_ci95_hi'] * 100:.3f}%]",
    )

    print("\n  QUALITY")
    p("win rate (traded markets)", f"{s['win_rate'] * 100:.2f}%")
    p("profit factor", fmt(s["profit_factor"], ".3f"))
    p("fees / gross P&L", fmt(s["fee_fraction_gross_pnl"], ".3f"))
    p("Sharpe per market", fmt(s["sharpe_per_market"], ".4f"))
    p("Sharpe annualized", fmt(s["sharpe"], ".2f"))
    p("t-stat vs zero edge", fmt(s["t_stat"], ".2f"))

    print("\n  EQUITY")
    p("cumulative P&L", "$" + fmt(s["total_pnl"]))
    p("max drawdown", "$" + fmt(s["max_drawdown"]))
    p("  in units of one stake", f"{s['max_drawdown'] / stake:.1f} stakes")

    equity, comp_dd, _ = compounded_view(mk, fraction)
    p(f"compounded @ {fraction:.0%} of bankroll", "$" + fmt(equity[-1]))
    p("  growth", f"{equity[-1] / stake:.4g}x")
    p("  max drawdown", f"{comp_dd * 100:.2f}%")

    cum_pnl = np.cumsum(mk.sort_values('start_ts').pnl.to_numpy())
    print(f"\n  cumulative P&L   {sparkline(cum_pnl)}")
    return equity, cum_pnl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default=DEFAULT_IN)
    ap.add_argument(
        "--split", default=None,
        help="which split to score. Optional while markets.csv holds only one; "
        "required once it holds more, since scoring them together would average "
        "the held-out split into the tuning one",
    )
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

    # markets.csv now holds every split we've scored, so it isn't single-split
    # anymore like it was back when the header below just printed split.iloc[0].
    present = sorted(markets.split.unique())
    if args.split is not None:
        if args.split not in present:
            raise SystemExit(f"no rows for split {args.split!r}; file has {present}")
        markets = markets[markets.split == args.split]
    elif len(present) > 1:
        raise SystemExit(
            f"{mk_path} holds several splits ({present}). Pass --split to pick "
            "one -- scoring them together would mix the held-out split into the "
            "tuning one."
        )

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
        "n_traded",
        "total_pnl",
        "pnl_per_1k_deployed",
        "gross_pnl_per_1k_deployed",
        "fee_per_1k_deployed",
        "avg_return",
        "win_rate",
        "profit_factor",
        "sharpe_per_market",
        "sharpe",
        "t_stat",
        "max_drawdown",
        "turnover",
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
