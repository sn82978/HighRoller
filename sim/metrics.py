"""All the trading metrics live here, so every model is scored the same way.

Both scoring paths call into this: BaselineModels.metrics.trading_metrics
(starts from MarketResult objects) and sim.evaluation.score (starts from a
markets.csv). They used to be two separate implementations and they didn't
agree -- max_drawdown was positive in one and negative in the other, Sharpe was
taken over different markets, and fee drag/turnover only existed in one of them.
You can't put numbers in the same table if they're computed differently, which
is the whole point of the four-way comparison.

Some of these definitions are judgement calls, so here's what we picked and why:

Per-market return (and so Sharpe) covers EVERY market in the split. A market
the policy sat out counts as a return of 0.0, not as missing. This one matters
a lot. If you only score a selective policy on the markets it picked, it looks
way better than it should next to something like buy-and-hold: an agent that
trades 40 of 1,343 markets and gets lucky looks amazing, and the 1,303 markets
where its money sat idle just vanish. Sitting out is a real outcome with a real
return of zero. Both baselines trade 100% of markets so this doesn't change any
of their numbers -- it's here for the Q-learning agent, which is very selective.

Win rate only covers markets we actually traded. A market you sat out isn't a
win or a loss, and counting it as a loss would give the no-trade baseline 0%
instead of undefined. So win rate answers "when we bet, how often were we
right" and Sharpe answers "what did running this do to the account". Two
different questions, both standard, only confusing if nobody says which is
which.

Max drawdown is a positive dollar number measured from zero. The running peak
starts at 0 instead of at the first market's PnL, so a strategy that loses money
starting from market #1 actually gets that counted. Always a magnitude (a
drawdown OF $4,423), never a percent.

Fee drag divides by net gross PnL -- realised PnL with the fees added back,
where losing markets keep their negative PnL. It's NaN and not 0.0 when that's
<= 0. The other option is to only sum the profitable markets, but that makes the
number look best exactly when the strategy is doing worst: on identical
buy-and-hold trades the two differ by 21.5x (6.9% positive-only vs 147.3% net
gross). Also, a hard 0.0 is what a broken fee counter looks like, so NaN is
safer.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

#: 15-minute markets in a year: 4/hour * 24 * 365. Named instead of inlined
#: because a Sharpe number is meaningless unless you say what you scaled it from.
MARKETS_PER_YEAR = 35_040

#: The shared schema. Every track writes these columns to its markets.csv and
#: score_records() reads exactly these, so models are comparable automatically
#: instead of because we remembered to keep them in sync.
#:
#: The old schema was missing everything from `fees` through `early_exit`, which
#: is why sim.evaluation.score just couldn't compute fee drag, turnover or
#: holding period. The arithmetic wasn't wrong, the inputs were never saved.
MARKET_RECORD_FIELDS: tuple[str, ...] = (
    "strategy",
    "event_slug",
    "start_ts",
    "split",
    "stake",
    "pnl",
    "fees",
    "stake_deployed",
    "notional_traded",
    "n_trades",
    "n_fills",
    "entry_candle",
    "exit_candle",
    "early_exit",
    "winner",
)


#: Records which cost model a row was simulated under. It's not in
#: MARKET_RECORD_FIELDS because it describes the run, not the market, but we
#: write it anyway so "everything used the same fees" is something you can
#: actually check in the file instead of just hoping.
COST_MODEL_FIELD = "slippage_frac"


def write_markets(
    path: str, frame: pd.DataFrame, split: str, *, slippage_frac: float | None = None
) -> int:
    """Write one split into markets.csv without wiping the other splits.

    Everything used to just overwrite this file. You never notice while you're
    only running val, but the second you score test it deletes all the val rows,
    and then compare_models.py --split val finds nothing for that track. It
    doesn't even error -- a track with no rows just prints [skip].

    So we only replace the rows for this split. That also means re-running the
    same split replaces instead of duplicating. Returns how many rows we kept
    from the old file.
    """
    if slippage_frac is not None:
        frame = frame.copy()
        frame[COST_MODEL_FIELD] = float(slippage_frac)

    kept = 0
    if os.path.exists(path):
        old = pd.read_csv(path)
        if "split" in old.columns:
            old = old[old.split != split]
            kept = len(old)
            if kept:
                frame = pd.concat([old, frame], ignore_index=True)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    frame.to_csv(path, index=False)
    return kept


def _capital_basis(mk: pd.DataFrame) -> np.ndarray:
    """How much money was actually at risk per market. This is what we divide by.

    Use `stake` (the $100 we allot to each market), NOT `stake_deployed`, which
    adds up the notional of every single entry. They're different as soon as a
    strategy re-enters -- momentum_flip can flip 12 times in one market, so
    stake_deployed says $1,200 when only $100 was ever on the line.

    This really matters. Dividing by the summed version shrinks exactly the
    markets that flipped the most, and those are also the biggest losers. We got
    a mean per-market return of +8.3% and a Sharpe of +38.8 on a val run that
    lost $36,296 in actual dollars. It made a terrible strategy look great.

    Falls back to stake_deployed if `stake` is missing or zero -- that's the
    single-entry case where they're the same number anyway.
    """
    deployed = mk.stake_deployed.to_numpy(dtype=float)
    if "stake" not in mk.columns:
        return deployed
    stake = pd.to_numeric(mk.stake, errors="coerce").to_numpy(dtype=float)
    return np.where(np.isfinite(stake) & (stake > 0), stake, deployed)


def _max_drawdown_dollars(pnl: np.ndarray) -> float:
    """Worst peak-to-trough drop in cumulative PnL, returned as a positive number.

    We start the running peak at 0.0 so that a strategy losing money from market
    #1 onward actually gets that counted, instead of measuring from whatever
    hole it had already dug itself into.
    """
    if len(pnl) == 0:
        return 0.0
    cum = np.cumsum(pnl)
    peak = np.maximum.accumulate(np.concatenate([[0.0], cum]))[1:]
    return float(np.max(peak - cum))


def _bootstrap_ci(x: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    """Percentile interval on the mean. Seeded, so a rerun reproduces it."""
    if len(x) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def score_records(
    mk: pd.DataFrame,
    markets_per_year: int = MARKETS_PER_YEAR,
    *,
    order_by: str = "start_ts",
) -> dict[str, float]:
    """Score one policy on one split. `mk` has the MARKET_RECORD_FIELDS columns.

    One row per market, and that includes markets the policy chose not to trade.
    Keeping those rows is what lets us compare a picky policy against one that
    trades everything.
    """
    missing = {"pnl", "fees", "stake_deployed", "notional_traded", "n_trades"} - set(mk.columns)
    if missing:
        raise ValueError(f"market records are missing columns: {sorted(missing)}")
    n = len(mk)
    if n == 0:
        raise ValueError("no markets to score")

    # Drawdown depends on the order markets resolved in, so sort here instead of
    # assuming whoever called us already did.
    if order_by in mk.columns:
        mk = mk.sort_values(order_by, kind="stable")

    pnl = mk.pnl.to_numpy(dtype=float)
    fees = float(mk.fees.sum())
    notional = float(mk.notional_traded.sum())
    gross = float(pnl.sum()) + fees

    traded_mask = mk.n_trades.to_numpy(dtype=float) > 0
    n_traded = int(traded_mask.sum())

    # Money at risk per market, plus the total we actually committed. A market we
    # sat out deployed nothing, so it adds 0 to the per-$1k denominator but still
    # adds a 0.0 return to the series.
    basis = _capital_basis(mk)
    deployed = float(basis[traded_mask].sum())

    # Return for EVERY market, with untraded ones at 0.0. See the docstring at
    # the top -- this is the thing that keeps a picky policy honest next to one
    # that always trades.
    with np.errstate(divide="ignore", invalid="ignore"):
        rets = np.where(basis > 0, pnl / np.where(basis > 0, basis, 1.0), 0.0)

    sd = float(rets.std(ddof=1)) if n > 1 else 0.0
    if sd > 0:
        sharpe_per_market = float(rets.mean() / sd)
        sharpe = sharpe_per_market * float(np.sqrt(markets_per_year))
        t_stat = sharpe_per_market * float(np.sqrt(n))
    else:
        sharpe_per_market = sharpe = t_stat = float("nan")

    ci_lo, ci_hi = _bootstrap_ci(rets)

    traded_pnl = pnl[traded_mask]
    wins = traded_pnl[traded_pnl > 0]
    losses = traded_pnl[traded_pnl < 0]

    holding = np.full(n, np.nan)
    if {"entry_candle", "exit_candle"} <= set(mk.columns):
        entry = pd.to_numeric(mk.entry_candle, errors="coerce").to_numpy(dtype=float)
        exit_ = pd.to_numeric(mk.exit_candle, errors="coerce").to_numpy(dtype=float)
        holding = exit_ - entry

    early = mk.early_exit.to_numpy(dtype=bool) if "early_exit" in mk.columns else np.zeros(n, bool)

    return {
        "n_markets": n,
        "n_traded": n_traded,
        "trade_rate": n_traded / n,
        # Entries plus early closes; settlement is a redemption, not a fill.
        "total_fills": int(mk.n_fills.sum()) if "n_fills" in mk.columns else int(mk.n_trades.sum()),
        "total_pnl": float(pnl.sum()),
        "gross_pnl": gross,
        "total_fees": fees,
        "capital_deployed": deployed,
        "avg_pnl_per_market": float(pnl.mean()),
        "median_pnl": float(np.median(pnl)),
        "pnl_per_1k_deployed": float(pnl.sum() / deployed * 1000.0) if deployed > 0 else 0.0,
        "gross_pnl_per_1k_deployed": float(gross / deployed * 1000.0) if deployed > 0 else 0.0,
        "fee_per_1k_deployed": float(fees / deployed * 1000.0) if deployed > 0 else 0.0,
        # NaN, never 0.0, when there are no profits for fees to be a share of.
        "fee_fraction_gross_pnl": float(fees / gross) if gross > 0 else float("nan"),
        "avg_return": float(rets.mean()),
        "return_ci95_lo": ci_lo,
        "return_ci95_hi": ci_hi,
        # Over traded markets only; NaN for a policy that never took a position.
        "win_rate": float((traded_pnl > 0).mean()) if n_traded else float("nan"),
        "profit_factor": (
            float(wins.sum() / abs(losses.sum())) if losses.size and losses.sum() < 0
            else float("nan")
        ),
        "sharpe": sharpe,
        "sharpe_per_market": sharpe_per_market,
        "t_stat": t_stat,
        "max_drawdown": _max_drawdown_dollars(pnl),
        "turnover": float(notional / deployed) if deployed > 0 else 0.0,
        "avg_holding_candles": (
            float(np.nanmean(holding[traded_mask])) if n_traded and not np.all(np.isnan(holding[traded_mask]))
            else float("nan")
        ),
        "early_exit_rate": float(early[traded_mask].mean()) if n_traded else 0.0,
    }


def comparison_table(named: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Stack several policies' scores into one table for the paper."""
    return pd.DataFrame(named).T.rename_axis("policy").reset_index()
