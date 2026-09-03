"""Tests for the fee / fill / accounting model.

The numbers checked here come straight from the proposal, so a failure means
either the code drifted or the proposal's cost assumptions need revisiting.
"""

import math

import pytest

from sim.execution import (
    BUY_DOWN,
    BUY_UP,
    CLOSE,
    CRYPTO_FEE_RATE,
    ExecutionConfig,
    Portfolio,
    Side,
    fill_price,
    taker_fee,
)

FRICTIONLESS = ExecutionConfig(slippage_frac=0.0, stake_dollars=100.0)


# -- fees ----------------------------------------------------------------
def test_fee_at_the_money_is_1_75_cents_per_share():
    assert taker_fee(1, 0.50) == pytest.approx(0.0175)


def test_fee_at_the_tails_is_0_63_cents_per_share():
    assert taker_fee(1, 0.10) == pytest.approx(0.0063)


def test_fee_is_symmetric_in_price():
    """Up at p and Down at 1-p are the same trade and must cost the same."""
    for p in (0.1, 0.25, 0.5, 0.73):
        assert taker_fee(100, p) == pytest.approx(taker_fee(100, 1 - p))


def test_fee_peaks_at_the_money():
    prices = [0.05, 0.2, 0.5, 0.8, 0.95]
    fees = [taker_fee(1, p) for p in prices]
    assert max(fees) == fees[prices.index(0.5)]


def test_fee_scales_with_share_count():
    assert taker_fee(200, 0.4) == pytest.approx(2 * taker_fee(100, 0.4))


# -- fills ---------------------------------------------------------------
def test_slippage_is_always_adverse():
    """Adding exposure fills higher, reducing it fills lower."""
    up = fill_price(0.50, 0.54, 0.46, direction=1, slippage_frac=0.25)
    down = fill_price(0.50, 0.54, 0.46, direction=-1, slippage_frac=0.25)
    assert up > 0.50 > down
    assert up == pytest.approx(0.52)
    assert down == pytest.approx(0.48)


def test_zero_slippage_fills_at_mid():
    assert fill_price(0.37, 0.9, 0.1, direction=1, slippage_frac=0.0) == pytest.approx(0.37)


def test_fill_tolerates_missing_range():
    assert fill_price(0.4, float("nan"), float("nan"), direction=1) == pytest.approx(0.4)


def test_fill_rejects_bad_direction():
    with pytest.raises(ValueError):
        fill_price(0.5, 0.6, 0.4, direction=0)


# -- accounting ----------------------------------------------------------
def test_buy_up_costs_price_plus_fee():
    p = Portfolio(config=FRICTIONLESS)
    assert p.buy(Side.UP, 0.50, 0.50, 0.50, candle_index=0)
    shares = 100.0 / 0.50
    assert p.shares_up == pytest.approx(shares)
    assert p.cash == pytest.approx(-(100.0 + taker_fee(shares, 0.50)))
    assert p.side is Side.UP


def test_buy_down_pays_one_minus_p():
    """A Down share at Up-equivalent 0.60 costs 0.40."""
    p = Portfolio(config=FRICTIONLESS)
    p.buy(Side.DOWN, 0.60, 0.60, 0.60, candle_index=0)
    assert p.shares_down == pytest.approx(100.0 / 0.40)
    assert p.side is Side.DOWN


def test_mark_to_market_at_entry_price_loses_exactly_the_fee():
    """Frictionless entry then immediate mark: the only loss is the fee."""
    p = Portfolio(config=FRICTIONLESS)
    p.buy(Side.UP, 0.50, 0.50, 0.50, candle_index=0)
    assert p.value(0.50) == pytest.approx(-p.fees_paid)


def test_round_trip_at_the_money_costs_about_3_5_cents_per_share():
    """The proposal's headline cost: ~3.5c of probability per share, no edge."""
    p = Portfolio(config=FRICTIONLESS)
    p.buy(Side.UP, 0.50, 0.50, 0.50, candle_index=0)
    shares = p.shares_up
    p.close(0.50, 0.50, 0.50, candle_index=1)
    assert p.cash / shares == pytest.approx(-0.035, abs=1e-4)


def test_settlement_charges_no_fee():
    """Redemption is free; that asymmetry is the point of the project."""
    p = Portfolio(config=FRICTIONLESS)
    p.buy(Side.UP, 0.50, 0.50, 0.50, candle_index=0)
    entry_fee = p.fees_paid
    p.settle("Up")
    assert p.fees_paid == pytest.approx(entry_fee), "settle() must not charge a fee"
    # Settlement adds no fill at all, so the entry is still the last trade.
    assert [t.action for t in p.trades] == [BUY_UP]


def test_closing_early_costs_strictly_more_than_holding():
    """Same position, same terminal price -- the early exit pays a second fee."""
    held = Portfolio(config=FRICTIONLESS)
    held.buy(Side.UP, 0.50, 0.50, 0.50, candle_index=0)
    held.settle("Up")

    sold = Portfolio(config=FRICTIONLESS)
    sold.buy(Side.UP, 0.50, 0.50, 0.50, candle_index=0)
    assert sold.close(0.90, 0.90, 0.90, candle_index=59)
    sold.settle("Up")  # flat by now, so this is a no-op payout

    assert held.fees_paid < sold.fees_paid
    assert held.cash > sold.cash


def test_close_is_refused_at_a_near_certain_price():
    """At 0.995 an early exit is strictly worse than a free redemption.

    The price band blocks the fill rather than letting a policy burn a fee to
    capture the last half-cent, so `close` returns False and the caller holds.
    """
    p = Portfolio(config=FRICTIONLESS)
    p.buy(Side.UP, 0.50, 0.50, 0.50, candle_index=0)
    assert p.close(0.995, 0.995, 0.995, candle_index=59) is False
    assert p.side is Side.UP


def test_winning_position_held_to_resolution():
    p = Portfolio(config=FRICTIONLESS)
    p.buy(Side.UP, 0.50, 0.50, 0.50, candle_index=0)
    shares = p.shares_up
    p.settle("Up")
    # Paid $100 + fee for `shares`, redeemed at $1 each.
    assert p.cash == pytest.approx(shares - 100.0 - taker_fee(shares, 0.50))
    assert p.cash > 0


def test_losing_position_held_to_resolution_loses_the_stake():
    p = Portfolio(config=FRICTIONLESS)
    p.buy(Side.UP, 0.50, 0.50, 0.50, candle_index=0)
    p.settle("Down")
    assert p.cash == pytest.approx(-(100.0 + p.fees_paid))


def test_settle_pays_the_down_leg_when_down_wins():
    p = Portfolio(config=FRICTIONLESS)
    p.buy(Side.DOWN, 0.60, 0.60, 0.60, candle_index=0)
    shares = p.shares_down
    p.settle("Down")
    assert p.cash == pytest.approx(shares - 100.0 - taker_fee(shares, 0.40))


def test_settle_twice_raises():
    p = Portfolio(config=FRICTIONLESS)
    p.settle("Up")
    with pytest.raises(RuntimeError):
        p.settle("Up")


def test_settle_rejects_unresolved_winner():
    with pytest.raises(ValueError):
        Portfolio(config=FRICTIONLESS).settle(None)


# -- action dispatch -----------------------------------------------------
def test_hold_is_a_noop():
    p = Portfolio(config=FRICTIONLESS)
    assert p.apply(0, 0.5, 0.5, 0.5, candle_index=0) is False
    assert p.cash == 0 and p.side is Side.FLAT


def test_flipping_sides_closes_the_old_leg_first():
    """Never hold both legs: that is a guaranteed-$1 box paying two fees."""
    p = Portfolio(config=FRICTIONLESS)
    p.apply(BUY_UP, 0.50, 0.50, 0.50, candle_index=0)
    p.apply(BUY_DOWN, 0.50, 0.50, 0.50, candle_index=1)
    assert p.shares_up == 0
    assert p.shares_down > 0
    assert len(p.trades) == 3  # buy, close, buy


def test_fill_count_does_not_depend_on_the_outcome():
    """Settlement is a redemption, not a fill, so it never enters `trades`.

    Recording it only when the payout was positive made `len(trades)` -- the
    `n_legs` column in every markets.csv -- one higher on markets that happened
    to resolve in our favour. Two identical policies would then look like they
    traded different amounts purely because one got lucky.
    """
    counts = {}
    for winner in ("Up", "Down"):
        p = Portfolio(config=FRICTIONLESS)
        p.buy(Side.UP, 0.50, 0.50, 0.50, candle_index=0)
        p.settle(winner)
        counts[winner] = len(p.trades)
    assert counts["Up"] == counts["Down"] == 1
    # The payout still lands in cash; only the bookkeeping entry is gone.
    won = Portfolio(config=FRICTIONLESS)
    won.buy(Side.UP, 0.50, 0.50, 0.50, candle_index=0)
    shares = won.shares
    cash_before = won.cash
    assert won.settle("Up") == pytest.approx(shares)
    assert won.cash == pytest.approx(cash_before + shares)


def test_unfillable_flip_leaves_the_position_untouched():
    """A flip either fully happens or doesn't. It never half-executes into a box.

    `side` returns Up whenever shares_up > 0, so a portfolio holding both legs
    would look like a normal Up position to every policy and metric downstream.
    Right now this works out for free -- the close and the buy in a flip share a
    direction, so they get the same fill price and refuse together -- but that's
    a coincidence. These two tests pin it down so changing one leg's guards
    later can't quietly break it.
    """
    p = Portfolio(config=FRICTIONLESS)
    p.apply(BUY_UP, 0.50, 0.50, 0.50, candle_index=0)
    shares_before, cash_before = p.shares_up, p.cash

    # 0.999 is outside the tradable band, so the close leg cannot fill.
    assert p.apply(BUY_DOWN, 0.999, 0.999, 0.999, candle_index=1) is False
    assert p.shares_down == 0
    assert p.shares_up == shares_before
    assert p.cash == cash_before
    assert len(p.trades) == 1  # the original buy, nothing else


def test_unfillable_flip_leaves_the_position_untouched_on_nan():
    """Same thing from the other direction: a NaN candle has no fill price."""
    p = Portfolio(config=FRICTIONLESS)
    p.apply(BUY_DOWN, 0.50, 0.50, 0.50, candle_index=0)
    shares_before = p.shares_down

    assert p.apply(BUY_UP, math.nan, math.nan, math.nan, candle_index=1) is False
    assert p.shares_up == 0
    assert p.shares_down == shares_before
    assert len(p.trades) == 1


def test_close_when_flat_is_a_noop():
    p = Portfolio(config=FRICTIONLESS)
    assert p.apply(CLOSE, 0.5, 0.5, 0.5, candle_index=0) is False
    assert p.trades == []


def test_unknown_action_raises():
    with pytest.raises(ValueError):
        Portfolio(config=FRICTIONLESS).apply(99, 0.5, 0.5, 0.5, candle_index=0)


# -- guards --------------------------------------------------------------
def test_refuses_to_fill_outside_the_price_band():
    """The tape touches 0.001; dividing a stake by that is not a real trade."""
    p = Portfolio(config=FRICTIONLESS)
    assert p.buy(Side.UP, 0.001, 0.001, 0.001, candle_index=0) is False
    assert p.trades == []


def test_refuses_to_fill_on_nan_price():
    p = Portfolio(config=FRICTIONLESS)
    assert p.buy(Side.UP, float("nan"), float("nan"), float("nan"), candle_index=0) is False


def test_no_trading_after_settlement():
    p = Portfolio(config=FRICTIONLESS)
    p.settle("Up")
    assert p.buy(Side.UP, 0.5, 0.5, 0.5, candle_index=0) is False


# -- the economics the project hinges on ---------------------------------
def test_a_correct_but_small_edge_still_loses_money_after_fees():
    """A model 2c better calibrated than the market loses on a round trip.

    This is the proposal's central claim -- a good probability estimate is not
    a profitable policy -- pinned down as a test.
    """
    cfg = ExecutionConfig(slippage_frac=0.0, stake_dollars=100.0)
    p = Portfolio(config=cfg)
    p.buy(Side.UP, 0.50, 0.50, 0.50, candle_index=0)
    p.close(0.52, 0.52, 0.52, candle_index=30)  # market moved 2c our way
    assert p.cash < 0, "2c of edge should not survive a round trip at the money"


def test_the_same_edge_survives_if_held_to_resolution():
    """Same edge, free exit: now it pays. Holding is structurally cheaper."""
    cfg = ExecutionConfig(slippage_frac=0.0, stake_dollars=100.0)
    wins, losses = 0.0, 0.0
    for winner in ("Up", "Down"):
        p = Portfolio(config=cfg)
        p.buy(Side.UP, 0.50, 0.50, 0.50, candle_index=0)
        p.settle(winner)
        if winner == "Up":
            wins = p.cash
        else:
            losses = p.cash
    # A 52% true probability against a 50c price, redeemed free.
    ev = 0.52 * wins + 0.48 * losses
    assert ev > 0, "2c of edge should survive when the exit is free"
