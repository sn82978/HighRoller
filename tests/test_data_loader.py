"""Tests for the temporal split.

The split is the project's main defence against a fake result, so these tests
run against the real dataset rather than a synthetic stand-in -- a split that
works on toy data and leaks on the real thing would be worthless.
"""

import pandas as pd
import pytest

from BaselineModels.data_loader import (
    LEAKY_COLUMNS,
    SPLIT_NAMES,
    SplitBounds,
    compute_bounds,
    dataset_files,
    load_split,
    market_universe,
    split_summary,
)

DATASET = "candles_15s"


@pytest.fixture(scope="module")
def universe():
    return market_universe(DATASET)


@pytest.fixture(scope="module")
def bounds(universe):
    return compute_bounds(universe, DATASET)


# -- the test-split guard -------------------------------------------------
def test_test_split_refuses_to_load_by_default():
    """The proposal budgets one look at the test block. Make it deliberate."""
    with pytest.raises(PermissionError, match="Refusing to load the test split"):
        load_split("test", DATASET)


def test_test_split_loads_when_explicitly_allowed(bounds):
    df = load_split(
        "test", DATASET, allow_test=True, columns=["event_slug", "start_ts", "candle_index", "close", "winner", "truncated"]
    )
    assert len(df) > 0


def test_unknown_split_raises():
    with pytest.raises(ValueError, match="unknown split"):
        load_split("holdout", DATASET)


def test_unknown_dataset_raises():
    with pytest.raises(ValueError, match="unknown dataset"):
        dataset_files("candles_5s")


# -- bounds ---------------------------------------------------------------
def test_bounds_are_strictly_ordered(bounds):
    assert bounds.train_start < bounds.train_end < bounds.val_end < bounds.test_end


def test_bounds_snap_to_utc_midnight(bounds):
    """Split edges line up with the daily parquet partitioning."""
    for ts in (bounds.train_start, bounds.train_end, bounds.val_end):
        assert ts % 86_400 == 0


def test_fractions_must_sum_to_one(universe):
    with pytest.raises(ValueError, match="must sum to 1"):
        compute_bounds(universe, DATASET, fractions=(0.5, 0.3, 0.3))


def test_bounds_cover_every_market(universe, bounds):
    """No market falls in a gap between splits."""
    assigned = 0
    for split in SPLIT_NAMES:
        lo, hi = bounds.range_for(split)
        assigned += ((universe.start_ts >= lo) & (universe.start_ts < hi)).sum()
    assert assigned == len(universe)


def test_ranges_are_contiguous_and_non_overlapping(bounds):
    train_lo, train_hi = bounds.range_for("train")
    val_lo, val_hi = bounds.range_for("val")
    test_lo, _ = bounds.range_for("test")
    assert train_hi == val_lo
    assert val_hi == test_lo


# -- disjointness ---------------------------------------------------------
def test_no_market_appears_in_two_splits(universe, bounds):
    seen: dict[str, str] = {}
    for split in SPLIT_NAMES:
        lo, hi = bounds.range_for(split)
        for slug in universe[(universe.start_ts >= lo) & (universe.start_ts < hi)].event_slug:
            assert slug not in seen, f"{slug} in both {seen.get(slug)} and {split}"
            seen[slug] = split
    assert len(seen) == len(universe)


def test_loaded_split_contains_only_its_own_markets(bounds):
    df = load_split("val", DATASET, columns=["event_slug", "start_ts", "candle_index", "close", "winner", "truncated"])
    lo, hi = bounds.range_for("val")
    assert df.start_ts.min() >= lo
    assert df.start_ts.max() < hi


def test_train_is_strictly_earlier_than_val():
    cols = ["event_slug", "start_ts", "candle_index", "close", "winner", "truncated"]
    train = load_split("train", DATASET, columns=cols)
    val = load_split("val", DATASET, columns=cols)
    assert train.start_ts.max() < val.start_ts.min()


# -- content --------------------------------------------------------------
def test_universe_has_one_row_per_market(universe):
    assert universe.event_slug.is_unique
    assert universe.start_ts.is_monotonic_increasing


def test_only_resolved_markets_are_returned():
    df = load_split("val", DATASET, columns=["event_slug", "start_ts", "candle_index", "close", "winner", "truncated"])
    assert df.winner.isin(("Up", "Down")).all()


def test_truncated_markets_keep_their_live_window_only():
    """18 markets lose part of the pre-open tape; their live window is intact."""
    cols = ["event_slug", "start_ts", "candle_index", "close", "winner", "truncated"]
    df = load_split("train", DATASET, columns=cols)
    trunc = df[df.truncated.astype(bool)]
    if len(trunc):
        assert trunc.candle_index.min() >= 0


def test_rows_are_sorted_within_each_market():
    df = load_split("val", DATASET, columns=["event_slug", "start_ts", "candle_index", "close", "winner", "truncated"])
    assert df.groupby("event_slug").candle_index.apply(lambda s: s.is_monotonic_increasing).all()


def test_split_summary_reports_all_three(bounds):
    s = split_summary(DATASET)
    assert list(s.split) == list(SPLIT_NAMES)
    assert (s.markets > 0).all()
    # A base rate far from 0.5 in one split would flatter or punish a model for
    # reasons that have nothing to do with the model.
    assert s.base_rate_up.between(0.40, 0.60).all()


# -- the leak catalogue ---------------------------------------------------
def test_leaky_columns_are_documented():
    assert "volume" in LEAKY_COLUMNS
    assert "winner" in LEAKY_COLUMNS


def test_market_total_volume_really_is_constant_per_market():
    """The reason `volume` is on the leak list, pinned to the real data."""
    df = load_split("val", DATASET, columns=["event_slug", "candle_index", "volume"])
    assert (df.groupby("event_slug").volume.nunique() == 1).all()


# -- disjointness assertion actually fires --------------------------------
def test_overlapping_bounds_cannot_be_constructed(universe):
    """Train swallowing val must be unrepresentable, not merely undetected.

    A load validates itself against the bounds it was handed, so degenerate
    bounds would otherwise validate happily against their own definition.
    """
    good = compute_bounds(universe, DATASET)
    with pytest.raises(ValueError, match="strictly increasing"):
        SplitBounds(
            train_start=good.train_start,
            train_end=good.val_end,  # train now swallows val
            val_end=good.val_end,
            test_end=good.test_end,
        )


def test_out_of_order_bounds_are_rejected():
    with pytest.raises(ValueError, match="strictly increasing"):
        SplitBounds(train_start=300, train_end=200, val_end=400, test_end=500)


def test_skewed_fractions_that_collapse_a_split_are_rejected(universe):
    """A validation block thinner than a day snaps to an empty range."""
    with pytest.raises(ValueError, match="strictly increasing"):
        compute_bounds(universe, DATASET, fractions=(0.999, 0.0005, 0.0005))


def test_loader_rejects_bounds_that_exclude_the_data(universe):
    """Bounds pointing at an empty stretch of calendar must fail, not return 0 rows."""
    far_future = 2_000_000_000
    bounds = SplitBounds(
        train_start=far_future,
        train_end=far_future + 86_400,
        val_end=far_future + 2 * 86_400,
        test_end=far_future + 3 * 86_400,
    )
    with pytest.raises(ValueError, match="zero markets"):
        load_split("train", DATASET, bounds=bounds, columns=["event_slug", "start_ts"])
