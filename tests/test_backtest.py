"""Tests for the backtest harness.

These are mostly about *when* a fill happens, not what it costs -- the cost
model is already covered in test_execution.py. The two things worth pinning
here are that a decision on candle c executes against candle c+1, and that a
policy can't see anything it shouldn't.
"""

import numpy as np
import pandas as pd
import pytest

from BaselineModels.backtest import (
    SETTLEMENT_CANDLE,
    Step,
    backtest,
    buy_and_hold_policy,
    no_trade_policy,
    run_market,
    threshold_policy,
)
from BaselineModels.features import LAST_CANDLE, build_features
from BaselineModels.metrics import trading_metrics
from sim.execution import BUY_DOWN, BUY_UP, CLOSE, HOLD, ExecutionConfig, Side

FRICTIONLESS = ExecutionConfig(slippage_frac=0.0, stake_dollars=100.0)


@pytest.fixture
def feats(candles):
    return build_features(candles)


def always(action):
    """Fires every candle, so it ends up adding to the position 59 times."""
    return lambda step: action


def enter_once(action):
    """Take one position and hold it. What the real policies here do."""
    return lambda step: action if step.n_entries == 0 else HOLD


# -- the no-trade floor --------------------------------------------------
def test_no_trade_policy_costs_nothing(feats):
    m = trading_metrics(backtest(feats, no_trade_policy))
    assert m["total_pnl"] == 0.0
    assert m["total_fees"] == 0.0
    assert m["n_traded"] == 0
    assert m["capital_deployed"] == 0.0


def test_no_trade_fee_fraction_is_nan_not_zero(feats):
    """Tells apart "paid no fees" from "never measured fees"."""
    m = trading_metrics(backtest(feats, no_trade_policy))
    assert np.isnan(m["fee_fraction_gross_pnl"])


def test_every_market_is_played_exactly_once(feats):
    results = backtest(feats, no_trade_policy)
    assert len(results) == feats.event_slug.nunique()
    assert len({r.event_slug for r in results}) == len(results)


# -- execution timing ----------------------------------------------------
def test_entry_fills_on_the_candle_after_the_decision(feats):
    """Decide on candle c, trade gets recorded at c+1."""
    one = feats[feats.event_slug == feats.event_slug.iloc[0]]
    r = run_market(one, always(BUY_UP), FRICTIONLESS)
    assert r.entry_candle == int(one.candle_index.iloc[0]) + 1


def test_fill_price_comes_from_the_next_candle_not_this_one(candles):
    """Check the price by arithmetic: the PnL only works out from next_open.

    Doing the same sum with p_mkt gives a different answer, so this breaks
    loudly if anyone ever drops the one-candle execution offset.
    """
    winners = candles[candles.winner == "Up"]
    feats = build_features(winners)
    one = feats[feats.event_slug == feats.event_slug.iloc[0]].reset_index(drop=True)
    first = one.iloc[0]
    assert first.next_open != pytest.approx(first.p_mkt), "test is vacuous if they match"

    r = run_market(one, enter_once(BUY_UP), FRICTIONLESS)
    shares = FRICTIONLESS.stake_dollars / first.next_open
    fee = shares * FRICTIONLESS.fee_rate * first.next_open * (1 - first.next_open)
    assert r.fees == pytest.approx(fee)
    assert r.pnl == pytest.approx(shares - FRICTIONLESS.stake_dollars - fee)

    # The same arithmetic against this candle's own price disagrees.
    wrong = FRICTIONLESS.stake_dollars / first.p_mkt
    assert r.pnl != pytest.approx(wrong - FRICTIONLESS.stake_dollars - fee)


def test_the_last_candle_cannot_trade(feats):
    """No candle 60 to fill against, so a signal there gets ignored."""
    one = feats[feats.event_slug == feats.event_slug.iloc[0]]
    late = one[one.candle_index == LAST_CANDLE]
    r = run_market(late, always(BUY_UP), FRICTIONLESS)
    assert r.n_trades == 0
    assert r.pnl == 0.0


def test_a_policy_cannot_reach_the_dataframe(feats):
    """Step is frozen and holds no reference to the frame or the outcome."""
    seen = []
    run_market(feats[feats.event_slug == feats.event_slug.iloc[0]], lambda s: seen.append(s) or HOLD)
    assert all(isinstance(s, Step) for s in seen)
    with pytest.raises(Exception):
        seen[0].p_mkt = 0.5  # frozen
    assert not any(hasattr(s, "winner") or hasattr(s, "y") for s in seen)


# -- settlement ----------------------------------------------------------
def test_a_held_position_settles_free_at_the_end(feats):
    one = feats[feats.event_slug == feats.event_slug.iloc[0]]
    r = run_market(one, enter_once(BUY_UP), FRICTIONLESS)
    assert r.exit_candle == SETTLEMENT_CANDLE
    assert not r.early_exit
    assert r.n_trades == 1
    # One entry, one fee. Settling adds neither a trade nor a cost.
    assert r.notional_traded == pytest.approx(r.stake_deployed)


def test_adding_to_a_position_is_charged_every_time(feats):
    """Re-buying each candle should stake more money and pay more fees."""
    one = feats[feats.event_slug == feats.event_slug.iloc[0]]
    once = run_market(one, enter_once(BUY_UP), FRICTIONLESS)
    many = run_market(one, always(BUY_UP), FRICTIONLESS)
    assert many.n_trades > once.n_trades
    assert many.stake_deployed > once.stake_deployed
    assert many.fees > once.fees


def test_holding_a_winner_pays_out(candles):
    winners = candles[candles.winner == "Up"]
    feats = build_features(winners)
    one = feats[feats.event_slug == feats.event_slug.iloc[0]]
    r = run_market(one, enter_once(BUY_UP), FRICTIONLESS)
    assert r.pnl > 0


def test_holding_a_loser_loses_the_stake_plus_the_fee(candles):
    losers = candles[candles.winner == "Down"]
    feats = build_features(losers)
    one = feats[feats.event_slug == feats.event_slug.iloc[0]]
    r = run_market(one, enter_once(BUY_UP), FRICTIONLESS)
    assert r.pnl == pytest.approx(-(r.stake_deployed + r.fees))


def test_backing_the_winner_beats_backing_the_loser(candles):
    """Perfect foresight makes money, perfectly wrong loses the stake."""
    feats = build_features(candles)
    oracle = backtest(
        feats,
        lambda s: HOLD if s.n_entries else (BUY_UP if _winner_of(feats, s) == "Up" else BUY_DOWN),
        FRICTIONLESS,
    )
    anti = backtest(
        feats,
        lambda s: HOLD if s.n_entries else (BUY_DOWN if _winner_of(feats, s) == "Up" else BUY_UP),
        FRICTIONLESS,
    )
    assert trading_metrics(oracle)["total_pnl"] > 0
    assert trading_metrics(anti)["total_pnl"] < 0
    assert trading_metrics(oracle)["win_rate"] == 1.0
    assert trading_metrics(anti)["win_rate"] == 0.0


def _winner_of(feats, step):
    """Cheats by design, for this test only. Real policies can't do this."""
    return feats.loc[feats.event_slug == step.event_slug, "winner"].iloc[0]


# -- early exit is the expensive path ------------------------------------
def test_closing_early_is_recorded_and_costs_a_second_fee(feats):
    one = feats[feats.event_slug == feats.event_slug.iloc[0]]

    def enter_then_leave(step):
        if step.side is Side.FLAT and step.n_entries == 0:
            return BUY_UP
        if step.side is not Side.FLAT and step.candles_remaining < 30:
            return CLOSE
        return HOLD

    held = run_market(one, enter_once(BUY_UP), FRICTIONLESS)
    flipped = run_market(one, enter_then_leave, FRICTIONLESS)
    assert flipped.early_exit
    assert flipped.n_trades == 1
    assert flipped.fees > held.fees
    assert flipped.notional_traded > held.notional_traded


# -- threshold policy ----------------------------------------------------
def test_threshold_policy_never_trades_when_it_agrees_with_the_market(feats):
    p = feats.p_mkt.to_numpy()
    m = trading_metrics(backtest(feats, threshold_policy(0.01), p_hat=p))
    assert m["n_traded"] == 0


def test_threshold_policy_buys_up_when_it_thinks_the_market_is_too_low(feats):
    p = np.ones(len(feats))
    results = backtest(feats, threshold_policy(0.05), FRICTIONLESS, p_hat=p)
    assert all(r.n_trades == 1 for r in results)
    frame = pd.DataFrame([{"slug": r.event_slug, "w": r.winner, "pnl": r.pnl} for r in results])
    assert (frame[frame.w == "Up"].pnl > 0).all()


def test_threshold_policy_buys_down_when_it_thinks_the_market_is_too_high(feats):
    p = np.zeros(len(feats))
    results = backtest(feats, threshold_policy(0.05), FRICTIONLESS, p_hat=p)
    frame = pd.DataFrame([{"w": r.winner, "pnl": r.pnl} for r in results])
    assert (frame[frame.w == "Down"].pnl > 0).all()


def test_a_higher_threshold_trades_no_more_often(feats):
    rng = np.random.default_rng(0)
    p = np.clip(feats.p_mkt.to_numpy() + rng.normal(0, 0.1, len(feats)), 0.01, 0.99)
    counts = [
        trading_metrics(backtest(feats, threshold_policy(t), p_hat=p))["n_traded"]
        for t in (0.01, 0.05, 0.20, 0.90)
    ]
    assert counts == sorted(counts, reverse=True)
    assert counts[-1] == 0


def test_max_entries_caps_positions_per_market(feats):
    p = np.ones(len(feats))

    def churn(step):
        if step.side is not Side.FLAT:
            return CLOSE
        return threshold_policy(0.05, max_entries=2)(step)

    results = backtest(feats, churn, FRICTIONLESS, p_hat=p)
    assert all(r.n_trades <= 2 for r in results)


def test_threshold_policy_ignores_nan_predictions(feats):
    p = np.full(len(feats), np.nan)
    m = trading_metrics(backtest(feats, threshold_policy(0.0), p_hat=p))
    assert m["n_traded"] == 0


def test_min_candles_remaining_blocks_late_entries(feats):
    p = np.ones(len(feats))
    results = backtest(feats, threshold_policy(0.05, min_candles_remaining=50), FRICTIONLESS, p_hat=p)
    for r in results:
        if r.entry_candle is not None:
            assert r.entry_candle <= LAST_CANDLE - 50 + 1


def test_negative_threshold_is_rejected():
    with pytest.raises(ValueError):
        threshold_policy(-0.01)


# -- buy and hold --------------------------------------------------------
def test_buy_and_hold_takes_exactly_one_position_per_market(feats):
    results = backtest(feats, buy_and_hold_policy(), FRICTIONLESS)
    assert len(results) == feats.event_slug.nunique()
    assert all(r.n_trades == 1 for r in results)


def test_buy_and_hold_never_sells_early(feats):
    for r in backtest(feats, buy_and_hold_policy(), FRICTIONLESS):
        assert not r.early_exit
        assert r.exit_candle == SETTLEMENT_CANDLE


def test_buy_and_hold_pays_exactly_one_fee(feats):
    """One entry fee, nothing at settlement -- notional equals the stake."""
    for r in backtest(feats, buy_and_hold_policy(), FRICTIONLESS):
        assert r.notional_traded == pytest.approx(r.stake_deployed)
        assert r.fees > 0


def test_buy_and_hold_enters_at_the_first_live_candle(feats):
    src = feats.groupby("event_slug").candle_index.min()
    for r in backtest(feats, buy_and_hold_policy(), FRICTIONLESS):
        assert r.entry_candle == int(src[r.event_slug]) + 1


def test_buy_and_hold_backs_the_favoured_side(candles):
    """Above 0.5 -> Up, below -> Down, judged on the opening price only."""
    up_first = _market_opening_at(candles, 0.80)
    down_first = _market_opening_at(candles, 0.20)

    r_up = run_market(build_features(up_first), buy_and_hold_policy(), FRICTIONLESS)
    r_dn = run_market(build_features(down_first), buy_and_hold_policy(), FRICTIONLESS)
    assert r_up.n_trades == 1 and r_dn.n_trades == 1
    # Winner is "Up" in both fixtures, so backing Up profits and Down does not.
    assert r_up.pnl > 0
    assert r_dn.pnl < 0


def test_buy_and_hold_tie_at_exactly_half_is_configurable(candles):
    """8.2% of real markets open at exactly 0.500, so this branch is load-bearing."""
    flat = _market_opening_at(candles, 0.50)
    feats = build_features(flat)

    down = run_market(feats, buy_and_hold_policy(tie="down"), FRICTIONLESS)
    up = run_market(feats, buy_and_hold_policy(tie="up"), FRICTIONLESS)
    skip = run_market(feats, buy_and_hold_policy(tie="skip"), FRICTIONLESS)

    assert down.n_trades == 1 and up.n_trades == 1
    assert skip.n_trades == 0 and skip.pnl == 0.0
    # Opposite sides of the same market: one wins, the other loses.
    assert (down.pnl > 0) != (up.pnl > 0)


def test_buy_and_hold_rejects_an_unknown_tie_rule():
    with pytest.raises(ValueError, match="tie must be"):
        buy_and_hold_policy(tie="coinflip")


def test_buy_and_hold_ignores_the_model(feats):
    """It must not consult p_hat, so predictions cannot change its behaviour."""
    a = backtest(feats, buy_and_hold_policy(), FRICTIONLESS)
    b = backtest(feats, buy_and_hold_policy(), FRICTIONLESS, p_hat=np.ones(len(feats)))
    assert [r.pnl for r in a] == pytest.approx([r.pnl for r in b])


def _market_opening_at(candles, price):
    """One market whose live window opens at `price`, winner Up."""
    one = candles[candles.event_slug == candles.event_slug.iloc[0]].copy()
    one["winner"] = "Up"
    live = one.candle_index >= 0
    for col in ("open", "high", "low", "close", "vwap"):
        one.loc[live, col] = price
    return one


# -- wiring guards -------------------------------------------------------
def test_predictions_stay_aligned_when_the_frame_is_reordered(feats):
    """backtest sorts internally, so predictions have to follow their rows."""
    shuffled = feats.sample(frac=1.0, random_state=0)
    p = np.where(shuffled.candle_index.to_numpy() % 2 == 0, 1.0, 0.0)
    a = backtest(shuffled, threshold_policy(0.05), FRICTIONLESS, p_hat=p)
    b = backtest(shuffled.assign(_p=p), threshold_policy(0.05), FRICTIONLESS, p_hat="_p")
    assert [r.pnl for r in a] == pytest.approx([r.pnl for r in b])


def test_mismatched_prediction_length_raises(feats):
    with pytest.raises(ValueError):
        backtest(feats, threshold_policy(0.05), p_hat=np.ones(len(feats) + 1))


def test_missing_columns_raise(feats):
    with pytest.raises(ValueError, match="missing columns"):
        backtest(feats.drop(columns=["next_open"]), no_trade_policy)


def test_unknown_p_hat_column_raises(feats):
    with pytest.raises(ValueError, match="not on the frame"):
        backtest(feats, threshold_policy(0.05), p_hat="nope")
