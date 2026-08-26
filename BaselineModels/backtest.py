"""Run a policy over a set of markets and score it, using the shared cost model.

All the baselines and the Q-learning agent should go through here. PnL after
fees is only comparable across models if the fills, fee rate and slippage are
identical, so this module owns *how* a trade costs money. It has no opinion on
*what* to trade -- that's the policy's job.

The rule it enforces: a decision made on candle c fills against candle c+1,
using the next_open / next_high / next_low columns that features.py puts on
every row. Policies get a Step, which holds no future data and no reference to
the DataFrame, so there's no way to fill at a price you just looked at.

Policies are plain functions of a Step. Everything a policy might need to
remember (current position, entries used so far) is on the Step, so you can
reuse the same policy across runs without state leaking between them.

    from BaselineModels.backtest import backtest, threshold_policy
    from BaselineModels.metrics import trading_metrics

    results = backtest(val_features, threshold_policy(0.05), p_hat=preds)
    print(trading_metrics(results))
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from BaselineModels.features import LAST_CANDLE
from BaselineModels.metrics import MarketResult
from sim.execution import (
    BUY_DOWN,
    BUY_UP,
    CLOSE,
    HOLD,
    ExecutionConfig,
    Portfolio,
    Side,
)

# Candle index we record settlements at, one past the last live candle.
SETTLEMENT_CANDLE = LAST_CANDLE + 1

_STEP_COLUMNS = [
    "candle_index",
    "p_mkt",
    "candles_remaining",
    "can_trade",
    "next_open",
    "next_high",
    "next_low",
]


@dataclass(frozen=True)
class Step:
    """Everything a policy gets to see at one decision point.

    A flat frozen record instead of the DataFrame row, so a policy has no way
    to reach back into the frame and accidentally index into the future.
    """

    event_slug: str
    candle_index: int
    p_mkt: float
    p_hat: float
    candles_remaining: int
    side: Side
    can_trade: bool
    # Entries already made in this market. Lets a policy cap its own turnover
    # without having to keep state between calls.
    n_entries: int


Policy = Callable[[Step], int]


def run_market(
    market: pd.DataFrame,
    policy: Policy,
    config: ExecutionConfig | None = None,
    p_hat: np.ndarray | None = None,
) -> MarketResult:
    """Play one market start to finish and return the result.

    Always settles at the end, free, whether or not the policy closed early.
    That asymmetry is the whole point: closing early pays a second taker fee,
    redeeming at settlement doesn't.
    """
    config = config or ExecutionConfig()
    port = Portfolio(config=config)

    slug = str(market.event_slug.iloc[0])
    winner = str(market.winner.iloc[0])
    if p_hat is None:
        p_hat = np.full(len(market), np.nan)
    p_hat = np.asarray(p_hat, dtype=float)
    if len(p_hat) != len(market):
        raise ValueError(f"p_hat has {len(p_hat)} rows, market {slug} has {len(market)}")

    entry_candle: int | None = None
    exit_candle: int | None = None
    early_exit = False
    n_entries = 0

    cols = market[_STEP_COLUMNS].to_numpy(dtype=float)

    for i in range(len(cols)):
        ci, p_mkt, rem, can_trade, nxt_o, nxt_h, nxt_l = cols[i]
        step = Step(
            event_slug=slug,
            candle_index=int(ci),
            p_mkt=float(p_mkt),
            p_hat=float(p_hat[i]),
            candles_remaining=int(rem),
            side=port.side,
            can_trade=bool(can_trade),
            n_entries=n_entries,
        )
        action = policy(step)
        if action == HOLD or not step.can_trade:
            continue

        was = port.side
        # Fill on the next candle, the one the Step couldn't see.
        if not port.apply(action, nxt_o, nxt_h, nxt_l, candle_index=int(ci) + 1):
            continue
        if action in (BUY_UP, BUY_DOWN):
            # Adding to a position we already hold still counts. It stakes more
            # money and pays another fee, so max_entries should catch it.
            n_entries += 1
            if entry_candle is None:
                entry_candle = int(ci) + 1
        if was is not Side.FLAT and port.side is Side.FLAT:
            exit_candle = int(ci) + 1
            early_exit = True

    # Snapshot the trades before settling. Settlement is an exit, not a trade,
    # and counting it would inflate turnover.
    traded_before_settle = list(port.trades)
    if port.side is not Side.FLAT:
        exit_candle = SETTLEMENT_CANDLE
        early_exit = False
    port.settle(winner, candle_index=SETTLEMENT_CANDLE)

    entries = [t for t in traded_before_settle if t.action in (BUY_UP, BUY_DOWN)]
    return MarketResult(
        event_slug=slug,
        pnl=port.cash,
        fees=port.fees_paid,
        # Capital allotted to this market. stake_deployed below sums every
        # entry's notional, which is the right denominator for turnover but not
        # for return: a policy that rolls one position through several entries
        # never had more than one stake at risk.
        stake=config.stake_dollars,
        stake_deployed=float(sum(t.shares * t.price for t in entries)),
        notional_traded=float(sum(t.shares * t.price for t in traded_before_settle)),
        n_trades=len(entries),
        entry_candle=entry_candle,
        exit_candle=exit_candle,
        winner=winner,
        early_exit=early_exit,
    )


def backtest(
    features: pd.DataFrame,
    policy: Policy,
    config: ExecutionConfig | None = None,
    p_hat: np.ndarray | str | None = None,
) -> list[MarketResult]:
    """Run `policy` over every market in `features`.

    p_hat can be an array lined up row-for-row with features, or the name of a
    column on it. Policies that don't use a model (no-trade, buy-and-hold,
    random) can leave it as None.

    Markets play in chronological order so the cumulative PnL path, and the
    drawdown that comes off it, match the order results actually arrived in.
    """
    required = {"event_slug", "candle_index", "p_mkt", "can_trade", "next_open", "winner"}
    missing = required - set(features.columns)
    if missing:
        raise ValueError(f"features frame is missing columns: {sorted(missing)}")

    df = features.copy()
    if isinstance(p_hat, str):
        if p_hat not in df.columns:
            raise ValueError(f"p_hat column {p_hat!r} is not on the frame")
        df["_p_hat"] = df[p_hat].astype(float)
    elif p_hat is None:
        df["_p_hat"] = np.nan
    else:
        preds = np.asarray(p_hat, dtype=float)
        if len(preds) != len(features):
            raise ValueError(f"p_hat has {len(preds)} rows, features has {len(features)}")
        # Attach before sorting, otherwise the predictions and the rows can
        # drift apart and you get a silently wrong backtest.
        df["_p_hat"] = preds

    order = ["start_ts", "event_slug", "candle_index"] if "start_ts" in df.columns else ["event_slug", "candle_index"]
    df = df.sort_values(order, ignore_index=True)

    preds = df["_p_hat"].to_numpy(dtype=float)
    return [
        run_market(df.iloc[idx], policy, config, preds[idx])
        for idx in df.groupby("event_slug", sort=False).indices.values()
    ]


# -- policies ------------------------------------------------------------
def no_trade_policy(step: Step) -> int:
    """Do nothing. The floor to beat: no PnL, no fees, no risk."""
    return HOLD


def buy_and_hold_policy(*, tie: str = "down", entry_candle: int = 0) -> Policy:
    """Back whichever side the market favours at the open, then hold to the end.

    The proposal's buy-and-hold: decide from the opening price alone, never
    look at anything else, never sell. Pays exactly one taker fee plus entry
    slippage, then rides to free settlement.

    Up if p_mkt > 0.5, Down if p_mkt < 0.5. At exactly 0.5 there is no favoured
    side, and `tie` decides: "down" (the default, matching the spec's "Down
    otherwise"), "up", or "skip" to sit the market out. This is not a rare
    branch -- 109 of 1,330 validation markets (8.2%) open at exactly 0.500,
    because the book quotes in whole cents -- so whichever way it is set, it is
    choosing the side for one market in twelve, at the price where the fee is
    at its maximum.

    Entry is the first tradable candle at or after `entry_candle`. For the 13
    validation markets whose tape starts after candle 0 that is their first
    available row rather than the true open; they are still traded so the
    market sample stays identical to the other policies.
    """
    if tie not in ("down", "up", "skip"):
        raise ValueError(f"tie must be 'down', 'up' or 'skip', got {tie!r}")

    def policy(step: Step) -> int:
        if step.side is not Side.FLAT or step.n_entries > 0:
            return HOLD
        if step.candle_index < entry_candle:
            return HOLD
        if not np.isfinite(step.p_mkt):
            return HOLD
        if step.p_mkt > 0.5:
            return BUY_UP
        if step.p_mkt < 0.5:
            return BUY_DOWN
        if tie == "skip":
            return HOLD
        return BUY_UP if tie == "up" else BUY_DOWN

    return policy


def threshold_policy(
    theta: float,
    *,
    max_entries: int = 1,
    min_candles_remaining: int = 0,
) -> Policy:
    """Buy when the model disagrees with the market by more than `theta`.

    Holds to resolution instead of taking profit, since redemption is free and
    closing early pays the fee twice. max_entries=1 caps it at one position per
    market -- without that the rule fires again on every candle where the edge
    is still there, which is most of them, and turnover blows up for no extra
    information.

    min_candles_remaining optionally blocks late entries. The market is
    sharpest right before settlement (AUC 0.993 in the last minute), so an edge
    we think we see there is probably noise.
    """
    if theta < 0:
        raise ValueError(f"theta must be non-negative, got {theta}")

    def policy(step: Step) -> int:
        if step.side is not Side.FLAT:
            return HOLD
        if step.n_entries >= max_entries:
            return HOLD
        if step.candles_remaining < min_candles_remaining:
            return HOLD
        if not np.isfinite(step.p_hat):
            return HOLD
        edge = step.p_hat - step.p_mkt
        if abs(edge) <= theta:
            return HOLD
        return BUY_UP if edge > 0 else BUY_DOWN

    return policy
