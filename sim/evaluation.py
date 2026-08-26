"""
Shared evaluation harness so strategies/, QLearning/ and BaselineModels/ are actually
comparable: same train/val/test split (market_slugs), same fill engine (simulate_market,
which drives sim.execution.Portfolio), same metrics (score() on MARKET_RECORD_FIELDS).
Point every model at this instead of reimplementing any of the three per model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import pandas as pd

from BaselineModels.data_loader import (
    SPLIT_NAMES,
    compute_bounds,
    load_split,
    market_universe,
)
from sim.execution import BUY_DOWN, BUY_UP, ExecutionConfig, Portfolio, Side
from sim.metrics import MARKET_RECORD_FIELDS, MARKETS_PER_YEAR, score_records

# last live candle index (0..59 once pre-open rows are dropped)
LAST_INDEX = 59
# where a held position redeems; one past the last live candle, matching
# BaselineModels/backtest.py so holding periods mean the same thing in both.
SETTLEMENT_CANDLE = LAST_INDEX + 1

__all__ = [
    "LAST_INDEX",
    "SETTLEMENT_CANDLE",
    "MARKET_RECORD_FIELDS",
    "MARKETS_PER_YEAR",
    "UNIVERSES",
    "market_slugs",
    "load_split_candles",
    "load_universe_candles",
    "MarketResult",
    "simulate_market",
    "results_to_frame",
    "score",
]


def market_slugs(split: str, dataset: str = "candles_15s", *, allow_test: bool = False) -> set[str]:
    """event_slugs belonging to `split`, per BaselineModels.data_loader's split."""
    universe = market_universe(dataset)
    bounds = compute_bounds(universe, dataset)
    lo, hi = bounds.range_for(split)
    keep = universe[(universe.start_ts >= lo) & (universe.start_ts < hi)]
    keep = keep[keep.winner.isin(("Up", "Down"))]
    return set(keep.event_slug)


def _add_next_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """Live-window rows, sorted, carrying the next candle's OHLC for fills."""
    df = df[df.candle_index >= 0].sort_values(["event_slug", "candle_index"], ignore_index=True)
    g = df.groupby("event_slug", sort=False)
    for col in ("open", "high", "low"):
        df[f"next_{col}"] = g[col].shift(-1)
    return df


def load_split_candles(split: str, *, allow_test: bool = False) -> pd.DataFrame:
    """live-window candles for a split, with next_open/high/low added for fills."""
    return _add_next_ohlc(load_split(split, dataset="candles_15s", allow_test=allow_test))


#: Multi-split universes. "dev" is everything you may iterate on freely; "all"
#: additionally contains the held-out block and is gated behind allow_test.
UNIVERSES: dict[str, tuple[str, ...]] = {
    "dev": ("train", "val"),
    "all": ("train", "val", "test"),
}


def load_universe_candles(name: str, *, allow_test: bool = False) -> pd.DataFrame:
    """Candles for one split or one named multi-split universe.

    The single place a multi-split load is allowed to happen, because the last
    one was not: ``strategies/generate_trades.py`` built its own "all" universe
    by calling ``load_split("test", allow_test=True)`` with the flag hardcoded,
    and ``--split all`` was its default. Every headline number that track has
    published was therefore computed over the held-out block, while the progress
    report states in two places that the test split has never been read.

    ``data_loader.load_split`` refuses the test split without ``allow_test``,
    but that guard only works if no caller hardcodes the flag. Routing every
    multi-split load through here means the flag has to come from the caller's
    own ``--allow-test``, so reading the held-out block stays a deliberate act
    that shows up in a shell history and a diff.
    """
    if name in SPLIT_NAMES:
        return load_split_candles(name, allow_test=allow_test)
    try:
        parts = UNIVERSES[name]
    except KeyError:
        raise ValueError(
            f"unknown universe {name!r}; expected one of "
            f"{sorted(SPLIT_NAMES) + sorted(UNIVERSES)}"
        ) from None
    if "test" in parts and not allow_test:
        raise PermissionError(
            f"universe {name!r} contains the held-out test split. Use 'dev' "
            "(train+val) while iterating, or pass allow_test=True for the single "
            "final evaluation reported in the paper."
        )
    frames = [load_split(s, dataset="candles_15s", allow_test=allow_test) for s in parts]
    return _add_next_ohlc(pd.concat(frames, ignore_index=True))


DecideFn = Callable[[pd.Series, Portfolio, int], int]


@dataclass
class MarketResult:
    """One policy's play of one market, in the interchange schema.

    Field-for-field the same quantities ``BaselineModels.metrics.MarketResult``
    carries, so both feed :func:`sim.metrics.score_records` without translation.
    """

    strategy: str
    event_slug: str
    start_ts: int
    split: str
    #: capital allotted to this market -- the return denominator. Distinct from
    #: stake_deployed, which sums every entry's notional and so double-counts a
    #: position that was rolled rather than added to.
    stake: float
    pnl: float
    fees: float
    stake_deployed: float
    notional_traded: float
    n_trades: int
    n_fills: int
    entry_candle: int | None
    exit_candle: int | None
    early_exit: bool
    winner: str
    portfolio: Portfolio


def simulate_market(
    episode: pd.DataFrame,
    decide: DecideFn,
    *,
    strategy: str,
    split: str,
    config: ExecutionConfig | None = None,
) -> MarketResult:
    """Runs one policy over one market's candles through Portfolio.

    episode = one market, candle_index 0..59, sorted, with next_open/high/low columns
    (see load_split_candles). decide(row, portfolio, i) only sees row + earlier state,
    never next_*, so it can't peek ahead.

    Shared by the rule baselines and the XGB baseline; the Q-learning agent trains with
    its own step loop but evaluates through this same function.

    Trades are stamped with the candle they *fill* on, not the one that signalled
    them -- matching BaselineModels/backtest.py, which was already doing that.
    The two harnesses used to differ by one candle here, which quietly shifted
    every holding period between the tracks by the same amount.
    """
    config = config or ExecutionConfig()
    portfolio = Portfolio(config=config)
    n = len(episode)
    entry_candle: int | None = None
    exit_candle: int | None = None
    early_exit = False

    for i in range(n):
        row = episode.iloc[i]
        is_last = i == n - 1
        action = decide(row, portfolio, i)
        if is_last:
            # candle 59 has no next candle to fill against; only settlement can
            # close a position opened on or before candle 58.
            continue
        was = portfolio.side
        fill_candle = int(row.candle_index) + 1
        if not portfolio.apply(action, row.next_open, row.next_high, row.next_low, fill_candle):
            continue
        if action in (BUY_UP, BUY_DOWN):
            if entry_candle is None:
                entry_candle = fill_candle
        if was is not Side.FLAT and portfolio.side is Side.FLAT:
            exit_candle = fill_candle
            early_exit = True

    last = episode.iloc[-1]
    if portfolio.side is not Side.FLAT:
        exit_candle = SETTLEMENT_CANDLE
        early_exit = False
        portfolio.settle(str(last.winner), candle_index=SETTLEMENT_CANDLE)

    entries = [t for t in portfolio.trades if t.action in (BUY_UP, BUY_DOWN)]
    return MarketResult(
        strategy=strategy,
        event_slug=str(last.event_slug),
        start_ts=int(episode.iloc[0].start_ts) if "start_ts" in episode.columns else 0,
        split=split,
        stake=config.stake_dollars,
        pnl=portfolio.cash,
        fees=portfolio.fees_paid,
        stake_deployed=float(sum(t.shares * t.price for t in entries)),
        notional_traded=float(sum(t.shares * t.price for t in portfolio.trades)),
        n_trades=len(entries),
        n_fills=len(portfolio.trades),
        entry_candle=entry_candle,
        exit_candle=exit_candle,
        early_exit=early_exit,
        winner=str(last.winner),
        portfolio=portfolio,
    )


def results_to_frame(results: Iterable[MarketResult]) -> pd.DataFrame:
    rows = [
        {f: getattr(r, f) for f in MARKET_RECORD_FIELDS}
        for r in results
    ]
    return pd.DataFrame(rows, columns=list(MARKET_RECORD_FIELDS))


# -- metrics --------------------------------------------------------------
def score(mk: pd.DataFrame) -> dict:
    """Score one strategy on one split. Delegates to :func:`sim.metrics.score_records`.

    ``mk`` needs :data:`MARKET_RECORD_FIELDS` and exactly one strategy + one
    split -- group by both before calling.

    This function used to carry its own arithmetic, which disagreed with
    ``BaselineModels.metrics.trading_metrics`` on nearly every quantity they
    shared: drawdown came out negative here and positive there, Sharpe was taken
    over a different set of markets, and fee drag, turnover and holding period
    were not computed at all because the old CSV schema did not carry the
    columns they need. Both now call the same function, so the four-way
    comparison the proposal asks for is arithmetic rather than translation.
    """
    if mk.strategy.nunique() > 1:
        raise ValueError("score() takes one strategy at a time; got " f"{sorted(mk.strategy.unique())}")
    if mk.split.nunique() > 1:
        raise ValueError("score() takes one split at a time; got " f"{sorted(mk.split.unique())}")

    return {
        "strategy": mk.strategy.iloc[0],
        "split": mk.split.iloc[0],
        **score_records(mk),
    }
