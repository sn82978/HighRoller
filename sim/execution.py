"""Fees, fills and position accounting for Polymarket BTC Up/Down 15m markets.

Price convention
----------------
Every price in this module is an **Up-equivalent** probability in (0, 1), the
same convention the 15s candle dataset uses: a Down fill at 0.40 is recorded as
an Up-equivalent 0.60. One Up share pays $1 if the market resolves Up; one Down
share pays $1 if it resolves Down and costs ``1 - p`` to buy.

Timing convention
-----------------
Features for candle ``c`` are built from candles ``<= c``; the action they
trigger executes against candle ``c + 1``. ``features.py`` carries the next
candle's OHLC on each row (``next_open`` / ``next_high`` / ``next_low``) so a
caller physically cannot fill at a price it used as an input. Nothing in this
module ever sees a price beyond the candle it was handed.

One assumption to state rather than bury: slippage is sized from the *fill*
candle's own high-low range, which is only fully known once that candle closes.
A live trader would size it from the spread quoted at the moment of the fill
instead. This is information from the fill candle, but it is spent exclusively
against the trader -- ``fill_price`` always displaces the price adversely -- so
it can make a strategy look worse than it is and never better. It is a cost
assumption, not a source of edge.

Fees
----
Polymarket charges the taker ``fee = C * r * p * (1 - p)`` where ``C`` is share
count and ``p`` the contract price; the crypto category carries ``r = 0.07``,
the highest rate on the platform. The fee is symmetric in ``p`` (so Up and Down
pay the same at mirrored prices), peaks at the money, and -- critically --
**redemption at settlement is free**. Holding to resolution is therefore
strictly cheaper than closing early, which is the asymmetry the whole project
turns on. Do not "simplify" that away.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Literal

# Polymarket taker fee rate for the crypto category.
CRYPTO_FEE_RATE = 0.07


class Side(IntEnum):
    """Which leg a position is on. FLAT means no open position."""

    FLAT = 0
    UP = 1
    DOWN = -1


# Action space, matching the proposal: buy YES, buy NO, hold, or sell.
HOLD = 0
BUY_UP = 1
BUY_DOWN = 2
CLOSE = 3
ACTIONS = (HOLD, BUY_UP, BUY_DOWN, CLOSE)
ACTION_NAMES = {HOLD: "hold", BUY_UP: "buy_up", BUY_DOWN: "buy_down", CLOSE: "close"}


@dataclass(frozen=True)
class ExecutionConfig:
    """Knobs for the cost model. The RL agent must use the same instance."""

    fee_rate: float = CRYPTO_FEE_RATE
    #: Adverse slippage as a fraction of the candle's high-low range. The 15s
    #: candles show a median high-low of 0.04 (p90 0.12) driven by bid-ask
    #: bounce in a wide book, so 0.25 charges roughly half the half-spread.
    #: Set to 0.0 for a frictionless upper bound on strategy performance.
    slippage_frac: float = 0.25
    #: Dollars committed per entry. PnL is reported per $1,000 deployed, so this
    #: only sets the granularity, not the headline number.
    stake_dollars: float = 100.0
    #: Refuse fills outside this band; the tape does touch 0.001/0.999 and
    #: dividing by a near-zero price produces absurd share counts.
    min_price: float = 0.01
    max_price: float = 0.99


def taker_fee(shares: float, price: float, fee_rate: float = CRYPTO_FEE_RATE) -> float:
    """Polymarket taker fee in dollars: ``C * r * p * (1 - p)``.

    Symmetric in ``price``, so it is identical whether you pass the Up price or
    the Down price of the same trade.
    """
    if shares <= 0:
        return 0.0
    return shares * fee_rate * price * (1.0 - price)


def fill_price(
    mid: float,
    high: float,
    low: float,
    direction: Literal[-1, 1],
    slippage_frac: float = 0.25,
) -> float:
    """Up-equivalent fill price after adverse slippage.

    ``direction`` is +1 when the trade *adds* Up-equivalent exposure (buying Up,
    or selling Down) and -1 when it *reduces* it (buying Down, or selling Up).
    The price always moves against the trader, so a round trip pays the slippage
    twice on top of the two fees.
    """
    if direction not in (-1, 1):
        raise ValueError(f"direction must be +1 or -1, got {direction!r}")
    span = 0.0
    if high is not None and low is not None:
        if not (math.isnan(high) or math.isnan(low)):
            span = max(0.0, high - low)
    return mid + direction * slippage_frac * span


@dataclass
class Trade:
    """One executed fill, for the turnover / fee / holding-period metrics."""

    candle_index: int
    action: int
    side: Side
    shares: float
    price: float  # price of the contract actually transacted, not Up-equivalent
    cash_delta: float
    fee: float


@dataclass
class Portfolio:
    """Position and cash for a single market, marked to market every step.

    The per-step reward the Q-learning agent needs is the change in
    :meth:`value` across a step, which already nets out any fee paid during it.
    """

    config: ExecutionConfig = field(default_factory=ExecutionConfig)
    cash: float = 0.0
    shares_up: float = 0.0
    shares_down: float = 0.0
    fees_paid: float = 0.0
    trades: list[Trade] = field(default_factory=list)
    settled: bool = False

    # -- state -----------------------------------------------------------
    @property
    def side(self) -> Side:
        if self.shares_up > 0:
            return Side.UP
        if self.shares_down > 0:
            return Side.DOWN
        return Side.FLAT

    @property
    def shares(self) -> float:
        """Size of the open position, whichever leg it is on."""
        return self.shares_up + self.shares_down

    def value(self, price_up: float) -> float:
        """Mark-to-market value: cash plus both legs at the current price."""
        return self.cash + self.shares_up * price_up + self.shares_down * (1.0 - price_up)

    def exposure(self, price_up: float) -> float:
        """Dollars currently at risk in the market (excludes cash)."""
        return self.shares_up * price_up + self.shares_down * (1.0 - price_up)

    # -- trading ---------------------------------------------------------
    def _tradable(self, price_up: float) -> bool:
        return (
            not self.settled
            and price_up is not None
            and not math.isnan(price_up)
            and self.config.min_price <= price_up <= self.config.max_price
        )

    def buy(
        self,
        side: Side,
        mid: float,
        high: float,
        low: float,
        candle_index: int,
        stake: float | None = None,
    ) -> bool:
        """Open or add to a position on ``side``. Returns whether it filled.

        A fill is refused rather than clamped when the price is outside the
        tradable band, so a rejected trade shows up as a no-op in the log
        instead of silently executing somewhere it could not have.
        """
        if side not in (Side.UP, Side.DOWN):
            raise ValueError(f"cannot buy side {side!r}")
        direction = 1 if side is Side.UP else -1
        fill_up = fill_price(mid, high, low, direction, self.config.slippage_frac)
        if not self._tradable(fill_up):
            return False

        price = fill_up if side is Side.UP else 1.0 - fill_up
        if price <= 0.0:
            return False
        stake = self.config.stake_dollars if stake is None else stake
        shares = stake / price
        fee = taker_fee(shares, price, self.config.fee_rate)

        self.cash -= shares * price + fee
        self.fees_paid += fee
        if side is Side.UP:
            self.shares_up += shares
        else:
            self.shares_down += shares
        self.trades.append(
            Trade(
                candle_index=candle_index,
                action=BUY_UP if side is Side.UP else BUY_DOWN,
                side=side,
                shares=shares,
                price=price,
                cash_delta=-(shares * price + fee),
                fee=fee,
            )
        )
        return True

    def close(self, mid: float, high: float, low: float, candle_index: int) -> bool:
        """Liquidate the open position early, paying the taker fee again.

        This is the expensive path. :meth:`settle` is the free one.
        """
        side = self.side
        if side is Side.FLAT:
            return False
        # Selling Up reduces Up exposure (-1); selling Down adds it (+1).
        direction = -1 if side is Side.UP else 1
        fill_up = fill_price(mid, high, low, direction, self.config.slippage_frac)
        if not self._tradable(fill_up):
            return False

        price = fill_up if side is Side.UP else 1.0 - fill_up
        shares = self.shares
        fee = taker_fee(shares, price, self.config.fee_rate)

        self.cash += shares * price - fee
        self.fees_paid += fee
        self.shares_up = 0.0
        self.shares_down = 0.0
        self.trades.append(
            Trade(
                candle_index=candle_index,
                action=CLOSE,
                side=side,
                shares=shares,
                price=price,
                cash_delta=shares * price - fee,
                fee=fee,
            )
        )
        return True

    def settle(self, winner: str, candle_index: int = 60) -> float:
        """Redeem at resolution. **No fee** -- this is the free exit.

        Returns the cash collected.

        Settlement is deliberately *not* recorded in :attr:`trades`. It is a
        redemption, not a fill: it pays no fee, crosses no spread, and counting
        it would inflate turnover. It used to be appended only when the payout
        was positive, which made ``len(trades)`` one higher on markets that
        happened to win -- a fill count that depended on the outcome. The cash
        collected is this method's return value and is visible in :attr:`cash`.
        """
        if self.settled:
            raise RuntimeError("portfolio already settled")
        if winner not in ("Up", "Down"):
            raise ValueError(f"winner must be 'Up' or 'Down', got {winner!r}")

        payout = self.shares_up if winner == "Up" else self.shares_down
        self.cash += payout
        self.shares_up = 0.0
        self.shares_down = 0.0
        self.settled = True
        return payout

    def apply(
        self,
        action: int,
        mid: float,
        high: float,
        low: float,
        candle_index: int,
        stake: float | None = None,
    ) -> bool:
        """Dispatch one action from :data:`ACTIONS`. Returns whether it filled.

        Buying the leg opposite an open position closes it first rather than
        holding both legs, which would be a guaranteed-$1 box paying two fees.
        If that close is refused the flip is abandoned whole.

        Today that guard never fires: closing Up and buying Down both *reduce*
        Up-equivalent exposure, so the two legs of a flip compute the same fill
        price and always agree on tradability. The check is here so the
        all-or-nothing invariant survives anyone tightening :meth:`buy`'s guards
        independently of :meth:`close`'s -- at which point the coincidence
        breaks and a half-executed flip would leave both legs open, which
        :attr:`side` cannot represent and no downstream metric would notice.
        """
        if action == HOLD:
            return False
        if action == CLOSE:
            return self.close(mid, high, low, candle_index)
        if action in (BUY_UP, BUY_DOWN):
            want = Side.UP if action == BUY_UP else Side.DOWN
            if self.side is not Side.FLAT and self.side is not want:
                if not self.close(mid, high, low, candle_index):
                    return False
            return self.buy(want, mid, high, low, candle_index, stake)
        raise ValueError(f"unknown action {action!r}")
