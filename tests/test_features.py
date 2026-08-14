"""Tests for feature construction -- above all, that nothing reads the future."""

import numpy as np
import pandas as pd
import pytest

from BaselineModels.data_loader import LEAKY_COLUMNS
from BaselineModels.features import (
    FEATURE_COLUMNS,
    LAST_CANDLE,
    RETURN_LAGS,
    add_book_asymmetry,
    assert_no_lookahead,
    build_features,
    feature_columns,
)


# -- leakage -------------------------------------------------------------
def test_no_lookahead_on_random_walks(candles):
    """Recomputing on truncated history must reproduce every feature exactly."""
    assert_no_lookahead(candles, n_markets=3, probe_indices=(0, 20, 45))


def test_no_lookahead_on_a_late_price_spike(spike_candles):
    """The hard case: a market that only reveals its outcome at candle 50."""
    assert_no_lookahead(spike_candles, n_markets=1, probe_indices=(0, 10, 30, 49))


def test_leaky_columns_are_not_features():
    """`volume` is the market's final total -- it must never reach a model."""
    for col in LEAKY_COLUMNS:
        assert col not in FEATURE_COLUMNS, f"{col!r} is a lookahead leak"


def test_market_total_volume_is_absent_from_the_frame(candles):
    feats = build_features(candles)
    assert "volume" not in feats.columns


def test_features_before_a_spike_do_not_see_it(spike_candles):
    """Concretely: at candle 40 the price is still 0.50 and momentum is flat."""
    feats = build_features(spike_candles, live_only=True)
    row = feats[feats.candle_index == 40].iloc[0]
    assert row.p_mkt == pytest.approx(0.50)
    for k in RETURN_LAGS:
        assert row[f"ret_{k}"] == pytest.approx(0.0)


# -- execution offset ----------------------------------------------------
def test_next_prices_are_the_following_candle(candles):
    """The row's fill price must come from a candle it did not read."""
    feats = build_features(candles, live_only=True, drop_warmup=False)
    src = candles.set_index(["event_slug", "candle_index"])
    for _, row in feats[feats.can_trade].head(50).iterrows():
        nxt = src.loc[(row.event_slug, row.candle_index + 1)]
        assert row.next_open == pytest.approx(nxt.open)
        assert row.next_high == pytest.approx(nxt.high)
        assert row.next_low == pytest.approx(nxt.low)


def test_last_candle_cannot_trade(candles):
    """There is no candle 60 to fill against, so the final row is untradable."""
    feats = build_features(candles)
    last = feats[feats.candle_index == LAST_CANDLE]
    assert len(last) == 3
    assert not last.can_trade.any()


def test_next_prices_never_cross_market_boundaries(candles):
    """Market A's last candle must not fill against market B's first."""
    feats = build_features(candles, live_only=True)
    for slug, grp in feats.groupby("event_slug"):
        assert not grp[grp.candle_index == LAST_CANDLE].can_trade.any(), slug


# -- per-market isolation ------------------------------------------------
def test_returns_do_not_bleed_across_markets(candles):
    """Rolling windows are grouped, so a market's history starts at its own open."""
    feats = build_features(candles, live_only=False, drop_warmup=False)
    first = feats.groupby("event_slug").head(1)
    assert first.ret_1.isna().all()
    assert first.ret_16.isna().all()


def test_row_count_is_one_per_market_per_live_candle(candles):
    feats = build_features(candles, live_only=True, drop_warmup=False)
    assert len(feats) == 3 * (LAST_CANDLE + 1)
    assert feats.duplicated(["event_slug", "candle_index"]).sum() == 0


def test_warmup_rows_are_dropped_by_default(candles):
    """No feature should silently be a shorter window than its name claims."""
    feats = build_features(candles, live_only=False, drop_warmup=True)
    starts = feats.groupby("event_slug").candle_index.min()
    assert (starts >= -40 + 16).all()


# -- label ---------------------------------------------------------------
def test_label_matches_the_winner(candles):
    feats = build_features(candles)
    for slug, grp in feats.groupby("event_slug"):
        expected = 1 if grp.winner.iloc[0] == "Up" else 0
        assert (grp.y == expected).all(), slug


def test_label_is_constant_within_a_market(candles):
    feats = build_features(candles)
    assert (feats.groupby("event_slug").y.nunique() == 1).all()


# -- null handling -------------------------------------------------------
def test_price_is_forward_filled_never_backward(candles):
    """A market whose tape starts late must not have its opening back-filled."""
    df = candles.copy()
    mask = (df.event_slug == df.event_slug.iloc[0]) & (df.candle_index < -20)
    df.loc[mask, ["open", "high", "low", "close", "vwap"]] = np.nan
    feats = build_features(df, live_only=False, drop_warmup=False)
    kept = feats[feats.event_slug == candles.event_slug.iloc[0]]
    assert kept.candle_index.min() >= -20


def test_features_are_finite_where_present(candles):
    feats = build_features(candles)
    block = feats[feature_columns(feats)].to_numpy(dtype=float)
    assert not np.isinf(block).any()


def test_vwap_gap_is_zero_on_untraded_candles(candles):
    """vwap is undefined at zero volume; the gap must not become NaN."""
    feats = build_features(candles)
    assert feats.vwap_gap.notna().all()


# -- clock ---------------------------------------------------------------
def test_candles_remaining_counts_down_to_zero(candles):
    feats = build_features(candles)
    assert feats.candles_remaining.min() == 0
    at_open = feats[feats.candle_index == 0]
    assert (at_open.candles_remaining == LAST_CANDLE).all()
    assert np.allclose(at_open.minutes_remaining, 14.75)


# -- book asymmetry ------------------------------------------------------
def test_book_asymmetry_joins_and_lags(candles):
    """Up + Down - 1 comes from the minute file and is read strictly late."""
    rows = []
    for slug in candles.event_slug.unique():
        for mi in range(-10, 16):
            rows.append({"event_slug": slug, "minute_index": mi, "outcome": "Up", "price": 0.50})
            rows.append(
                {"event_slug": slug, "minute_index": mi, "outcome": "Down", "price": 0.503}
            )
    minute = pd.DataFrame(rows)

    feats = build_features(candles, live_only=True)
    joined = add_book_asymmetry(feats, minute)
    assert "book_asym" in joined.columns
    assert "book_asym" in feature_columns(joined)
    # The allowlist itself is untouched -- the join is per-frame, not global.
    assert "book_asym" not in FEATURE_COLUMNS
    assert "book_asym" not in feature_columns(feats)
    live = joined[joined.candle_index >= 4]
    assert live.book_asym.abs().gt(0).all()
    assert live.book_asym.iloc[0] == pytest.approx(0.003, abs=1e-9)


def test_book_asymmetry_rejects_a_one_sided_minute_frame(candles):
    minute = pd.DataFrame(
        [{"event_slug": "x", "minute_index": 0, "outcome": "Up", "price": 0.5}]
    )
    with pytest.raises(ValueError):
        add_book_asymmetry(build_features(candles), minute)


# -- sample cuts must announce themselves --------------------------------
def test_a_late_starting_market_is_dropped_with_a_warning(candles):
    """18 real markets have no trades until candle 45; they cannot be warmed up.

    Dropping them is right, but it shrinks the sample, so it must be loud.
    """
    late = candles[candles.event_slug == candles.event_slug.iloc[0]].copy()
    late["event_slug"] = "btc-updown-15m-late"
    for col in ("open", "high", "low", "close", "vwap"):
        late.loc[late.candle_index < 45, col] = float("nan")

    with pytest.warns(UserWarning, match="starts too late"):
        out = build_features(late)
    assert "btc-updown-15m-late" not in set(out.event_slug)


def test_no_warning_when_every_market_survives(candles):
    import warnings as _w

    with _w.catch_warnings():
        _w.simplefilter("error")
        build_features(candles)


# -- guards --------------------------------------------------------------
def test_missing_columns_raise(candles):
    with pytest.raises(ValueError, match="missing columns"):
        build_features(candles.drop(columns=["winner"]))
