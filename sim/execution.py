"""Fees, fills and position tracking for the Polymarket BTC Up/Down 15m markets.

Prices
------
Every price here is "Up-equivalent" -- a probability in (0, 1), same as the 15s
candle dataset. So a Down fill at 0.40 gets stored as Up-equivalent 0.60. One Up
share pays $1 if the market resolves Up. One Down share pays $1 if it resolves
Down and costs 1 - p.

Timing
------
Features for candle c only use candles <= c, and whatever action they trigger
fills against candle c + 1. features.py puts the next candle's OHLC on each row
(next_open / next_high / next_low), so you physically can't fill at a price you
also used as an input. Nothing in here ever looks past the candle it was given.

One assumption worth calling out instead of hiding: we size slippage from the
fill candle's own high-low range, which you'd only know after that candle
closes. A real trader would use the spread quoted at the moment of the fill.
Technically that's information from the fill candle, but fill_price() always
moves the price against us, so it can only make a strategy look worse than it
is, never better. It's a cost assumption, not free edge.

Fees
----
Polymarket charges takers fee = C * r * p * (1 - p), where C is the number of
shares and p is the price. Crypto markets use r = 0.07, the highest rate they
have. The fee is symmetric in p (Up and Down pay the same at mirrored prices)
and is largest at 50/50. The important part: redeeming at settlement is FREE.
That makes holding to resolution strictly cheaper than closing early, which is
the asymmetry this whole project is about. Don't "simplify" it away.
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
    """Settings for the cost model. The RL agent has to use the same one."""

    fee_rate: float = CRYPTO_FEE_RATE
    #: Slippage against us, as a fraction of the candle's high-low range. Median
    #: high-low on the 15s candles is 0.04 (p90 is 0.12), mostly bid-ask bounce
    #: in a wide book, so 0.25 works out to about half the half-spread. Set it to
    #: 0.0 if you want a frictionless best case.
    slippage_frac: float = 0.25
    #: Dollars per entry. We report PnL per $1,000 deployed, so this just sets
    #: the granularity and doesn't change the headline number.
    stake_dollars: float = 100.0
    #: Don't fill outside this range. The tape really does hit 0.001/0.999, and
    #: dividing by a price that small gives you a ridiculous number of shares.
    min_price: float = 0.01
    max_price: float = 0.99


def taker_fee(shares: float, price: float, fee_rate: float = CRYPTO_FEE_RATE) -> float:
    """Polymarket taker fee in dollars: C * r * p * (1 - p).

    Symmetric in price, so you get the same answer whether you pass the Up price
    or the Down price for the same trade.
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
    """Up-equivalent fill price once slippage has been charged against us.

    direction is +1 when the trade adds Up-equivalent exposure (buying Up or
    selling Down) and -1 when it reduces it (buying Down or selling Up). The
    price always moves the wrong way for us, so a round trip eats the slippage
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
    """A single fill. We need these to compute turnover, fees and holding time."""

    candle_index: int
    action: int
    side: Side
    shares: float
    price: float  # price of the contract actually transacted, not Up-equivalent
    cash_delta: float
    fee: float


@dataclass
class Portfolio:
    """Cash and position for one market, marked to market on every step.

    The reward the Q-learning agent wants is just the change in value() over a
    step, which already has any fee paid during that step subtracted out.
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
        """How many shares we're holding, whichever leg they're on."""
        return self.shares_up + self.shares_down

    def value(self, price_up: float) -> float:
        """What the account is worth right now: cash plus both legs at current price."""
        return self.cash + self.shares_up * price_up + self.shares_down * (1.0 - price_up)

    def exposure(self, price_up: float) -> float:
        """Dollars actually at risk in the market. Doesn't count cash."""
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
        """Open or add to a position on `side`. Returns True if it actually filled.

        If the price is outside the tradable band we refuse the fill instead of
        clamping it. That way a rejected trade shows up as a no-op in the log,
        rather than quietly executing at a price it never could have.
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
        """Dump the position early, which means paying the taker fee a second time.

        This is the expensive way out. settle() is the free one.
        """
        side = self.side
        if side is Side.FLAT:
            return False
        # Selling Up reduces Up exposure (-1), selling Down adds it (+1).
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
        """Redeem at resolution. No fee -- this is the free exit.

        Returns the cash we collected.

        On purpose, settlement does NOT get added to self.trades. It's a
        redemption, not a fill: no fee, no spread crossed, and counting it would
        inflate turnover. It used to get appended, but only when the payout was
        positive, so len(trades) came out one higher on markets that happened to
        win. A fill count that depends on whether you won is obviously wrong.
        The cash is the return value and also shows up in self.cash.
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
        """Run one action from ACTIONS. Returns True if something filled.

        If you buy the leg opposite an open position, we close the old one
        first instead of holding both. Holding both would be a guaranteed $1 box
        that costs two fees to build, which is never what you want. If that
        close gets refused we bail on the whole flip.

        Right now that guard can't actually trigger: closing Up and buying Down
        both reduce Up-equivalent exposure, so both legs compute the same fill
        price and always agree on whether it's tradable. It's here so the
        all-or-nothing behaviour survives if someone later tightens buy()'s
        checks without matching close(). Once that coincidence breaks, a
        half-done flip would leave both legs open, and `side` has no way to
        represent that, so nothing downstream would catch it.
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
