"""
Shared evaluation harness so strategies/, QLearning/ and BaselineModels/ are actually
comparable: same train/val/test split (market_slugs), same fill engine (simulate_market,
which drives sim.execution.Portfolio), same metrics (score() on MARKET_RECORD_FIELDS).
Point every model at this instead of reimplementing any of the three per model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from BaselineModels.data_loader import compute_bounds, load_split, market_universe
from sim.execution import ExecutionConfig, Portfolio, Side

# last live candle index (0..59 once pre-open rows are dropped)
LAST_INDEX = 59

# columns every model's markets.csv has to have, this is what score() reads
MARKET_RECORD_FIELDS = (
    "strategy",
    "event_slug",
    "start_ts",
    "split",
    "stake",
    "pnl",
    "return_pct",
    "traded",
    "n_legs",
    "winner",
)


def market_slugs(split: str, dataset: str = "candles_15s", *, allow_test: bool = False) -> set[str]:
    """event_slugs belonging to `split`, per BaselineModels.data_loader's split."""
    universe = market_universe(dataset)
    bounds = compute_bounds(universe, dataset)
    lo, hi = bounds.range_for(split)
    keep = universe[(universe.start_ts >= lo) & (universe.start_ts < hi)]
    keep = keep[keep.winner.isin(("Up", "Down"))]
    return set(keep.event_slug)


def load_split_candles(split: str, *, allow_test: bool = False) -> pd.DataFrame:
    """live-window candles for a split, with next_open/high/low added for fills."""
    df = load_split(split, dataset="candles_15s", allow_test=allow_test)
    df = df[df.candle_index >= 0].sort_values(["event_slug", "candle_index"], ignore_index=True)
    g = df.groupby("event_slug", sort=False)
    for col in ("open", "high", "low"):
        df[f"next_{col}"] = g[col].shift(-1)
    return df


DecideFn = Callable[[pd.Series, Portfolio, int], int]


@dataclass
class MarketResult:
    strategy: str
    event_slug: str
    start_ts: int
    split: str
    stake: float
    pnl: float
    return_pct: float
    traded: bool
    n_legs: int
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
    """
    config = config or ExecutionConfig()
    portfolio = Portfolio(config=config)
    n = len(episode)
    for i in range(n):
        row = episode.iloc[i]
        is_last = i == n - 1
        action = decide(row, portfolio, i)
        if not is_last:
            portfolio.apply(
                action, row.next_open, row.next_high, row.next_low, int(row.candle_index)
            )
        # candle 59 has no next candle to fill against; only settlement can
        # close a position opened on or before candle 58.
    last = episode.iloc[-1]
    if portfolio.side is not Side.FLAT:
        portfolio.settle(last.winner, candle_index=LAST_INDEX)

    stake = config.stake_dollars
    pnl = portfolio.cash
    return MarketResult(
        strategy=strategy,
        event_slug=str(last.event_slug),
        start_ts=int(episode.iloc[0].start_ts) if "start_ts" in episode.columns else 0,
        split=split,
        stake=stake,
        pnl=pnl,
        return_pct=pnl / stake * 100.0,
        traded=bool(portfolio.trades),
        n_legs=len(portfolio.trades),
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

MARKETS_PER_YEAR = 4 * 24 * 365


def _bootstrap_ci(x: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    if len(x) < 2:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _max_drawdown_dollars(cum_pnl: np.ndarray) -> float:
    if len(cum_pnl) == 0:
        return 0.0
    peak = np.maximum.accumulate(np.concatenate([[0.0], cum_pnl]))[1:]
    return float(np.min(cum_pnl - peak))


def score(mk: pd.DataFrame) -> dict:
    """Headline metrics (ROI, Sharpe, t-stat, drawdown, CI) for one strategy/split's rows.

    mk needs MARKET_RECORD_FIELDS and exactly one strategy + one split -- group by both
    before calling. This is the one place these numbers get computed for cross-model
    comparisons; analyze_trades.py adds its own extra diagnostics on top separately.
    """
    if mk.strategy.nunique() > 1:
        raise ValueError("score() takes one strategy at a time; got " f"{sorted(mk.strategy.unique())}")
    if mk.split.nunique() > 1:
        raise ValueError("score() takes one split at a time; got " f"{sorted(mk.split.unique())}")

    mk = mk.sort_values("start_ts").reset_index(drop=True)
    n = len(mk)
    stake = float(mk.stake.iloc[0])
    pnl = mk.pnl.to_numpy()
    ret = mk.return_pct.to_numpy() / 100.0
    cum_pnl = np.cumsum(pnl)

    wins, losses = pnl[pnl > 0], pnl[pnl < 0]
    sd = ret.std(ddof=1) if n > 1 else 0.0
    sharpe = ret.mean() / sd if sd > 0 else 0.0
    tstat = sharpe * math.sqrt(n) if sd > 0 else 0.0
    lo, hi = _bootstrap_ci(ret)

    return {
        "strategy": mk.strategy.iloc[0],
        "split": mk.split.iloc[0],
        "markets": n,
        "markets_traded": int(mk.traded.sum()),
        "participation_%": mk.traded.mean() * 100 if n else 0.0,
        "total_staked": stake * n,
        "total_pnl": float(pnl.sum()),
        "roi_on_stake_%": float(pnl.sum() / (stake * n) * 100) if n else 0.0,
        "avg_pnl_per_market": float(pnl.mean()) if n else 0.0,
        "median_pnl": float(np.median(pnl)) if n else 0.0,
        "avg_return_%": float(ret.mean() * 100) if n else 0.0,
        "return_ci95_low_%": lo * 100,
        "return_ci95_high_%": hi * 100,
        "win_rate_%": float((pnl > 0).mean() * 100) if n else 0.0,
        "profit_factor": (
            float(wins.sum() / abs(losses.sum())) if len(losses) and losses.sum() else float("inf")
        ),
        "sharpe_per_market": float(sharpe),
        "sharpe_annualized": float(sharpe * math.sqrt(MARKETS_PER_YEAR)),
        "t_stat": float(tstat),
        "max_drawdown_$": _max_drawdown_dollars(cum_pnl),
        "max_drawdown_stakes": _max_drawdown_dollars(cum_pnl) / stake if stake else 0.0,
    }
