"""Tests for the XGBoost baseline.

Mostly guards around the theta selection. The sweep is the only place in this
project where we *pick* a number instead of just reporting one, so it's the
most likely spot for a bogus result to sneak in.
"""

import numpy as np
import pandas as pd
import pytest

from BaselineModels.xgb_baseline import (
    MARKET_PRICE_FEATURES,
    MIN_TRADES_FOR_SELECTION,
    best_theta,
    compare_to_market,
    fit,
)


def _sweep(rows):
    return pd.DataFrame(rows)


# -- theta selection -----------------------------------------------------
def test_a_single_lucky_trade_cannot_win_the_sweep():
    """Numbers from the actual first run: 1 trade scored +$1005 per $1k."""
    sweep = _sweep(
        [
            {"theta": 0.03, "n_traded": 729, "pnl_per_1k_deployed": -43.97},
            {"theta": 0.10, "n_traded": 1, "pnl_per_1k_deployed": 1005.12},
        ]
    )
    assert best_theta(sweep) == pytest.approx(0.03)


def test_selection_requires_a_minimum_number_of_trades():
    sweep = _sweep(
        [
            {"theta": 0.05, "n_traded": MIN_TRADES_FOR_SELECTION - 1, "pnl_per_1k_deployed": 900.0},
            {"theta": 0.02, "n_traded": MIN_TRADES_FOR_SELECTION, "pnl_per_1k_deployed": -1.0},
        ]
    )
    assert best_theta(sweep) == pytest.approx(0.02)


def test_no_tradeable_theta_raises_rather_than_relaxing_the_floor():
    sweep = _sweep([{"theta": 0.2, "n_traded": 2, "pnl_per_1k_deployed": 500.0}])
    with pytest.raises(ValueError, match="tradeable sample"):
        best_theta(sweep)


def test_selection_maximises_the_criterion_among_eligible_thetas():
    sweep = _sweep(
        [
            {"theta": 0.01, "n_traded": 500, "pnl_per_1k_deployed": -10.0},
            {"theta": 0.02, "n_traded": 400, "pnl_per_1k_deployed": 5.0},
            {"theta": 0.03, "n_traded": 300, "pnl_per_1k_deployed": 2.0},
        ]
    )
    assert best_theta(sweep) == pytest.approx(0.02)


def test_selection_criterion_is_configurable():
    sweep = _sweep(
        [
            {"theta": 0.01, "n_traded": 500, "pnl_per_1k_deployed": 10.0, "sharpe": 0.1},
            {"theta": 0.02, "n_traded": 400, "pnl_per_1k_deployed": 5.0, "sharpe": 9.0},
        ]
    )
    assert best_theta(sweep, criterion="sharpe") == pytest.approx(0.02)


# -- fitting -------------------------------------------------------------
@pytest.fixture
def split(candles):
    from BaselineModels.features import build_features

    feats = build_features(candles)
    slugs = sorted(feats.event_slug.unique())
    tr = feats[feats.event_slug.isin(slugs[:2])]
    va = feats[feats.event_slug.isin(slugs[2:])]
    return tr, va


def test_fit_returns_probabilities_in_range(split):
    tr, va = split
    model = fit(tr, va, num_boost_round=10, early_stopping_rounds=5)
    p = model.predict(va)
    assert len(p) == len(va)
    assert ((p >= 0) & (p <= 1)).all()


def test_the_label_is_never_a_feature(split):
    tr, va = split
    model = fit(tr, va, num_boost_round=10, early_stopping_rounds=5)
    for banned in ("y", "winner", "volume", "next_open", "next_high", "next_low"):
        assert banned not in model.features


def test_ablation_removes_the_market_price(split):
    tr, va = split
    model = fit(tr, va, num_boost_round=10, early_stopping_rounds=5, drop_features=MARKET_PRICE_FEATURES)
    for f in MARKET_PRICE_FEATURES:
        assert f not in model.features
    assert len(model.features) > 0


def test_dropping_every_feature_raises(split):
    tr, va = split
    from BaselineModels.features import feature_columns

    with pytest.raises(ValueError, match="no features left"):
        fit(tr, va, drop_features=tuple(feature_columns(tr)))


def test_predicting_on_a_frame_without_the_features_raises(split):
    tr, va = split
    model = fit(tr, va, num_boost_round=10, early_stopping_rounds=5)
    with pytest.raises(ValueError, match="missing model features"):
        model.predict(va.drop(columns=["p_mkt"]))


# -- reporting -----------------------------------------------------------
def test_comparison_includes_the_market_as_a_competitor(split):
    _, va = split
    tbl = compare_to_market(va, np.full(len(va), 0.5))
    assert set(tbl.forecaster) == {"market", "model", "always_0.5"}


def test_an_uninformative_model_scores_like_a_coin_flip(split):
    _, va = split
    tbl = compare_to_market(va, np.full(len(va), 0.5)).set_index("forecaster")
    assert tbl.loc["model", "log_loss"] == pytest.approx(np.log(2))
    assert tbl.loc["model", "log_loss"] == pytest.approx(tbl.loc["always_0.5", "log_loss"])


# -- theta selection must never see the reported split -------------------
def test_theta_is_selected_on_val_only():
    """Only validation may choose the threshold.

    On --split test the driver used to read theta off the test sweep, which
    turns the held-out number into "the best of eleven thresholds tried on the
    held-out data". Train is excluded for the same reason in reverse: a
    threshold picked on the data the model was fitted to is not a threshold.
    """
    from BaselineModels.run_baselines import SELECTION_SPLIT, selects_on_eval_split

    assert selects_on_eval_split("val")
    assert not selects_on_eval_split("test")
    assert not selects_on_eval_split("train")
    assert SELECTION_SPLIT == "val"


def test_a_theta_chosen_on_test_would_have_been_a_different_theta():
    """The guard above is not decorative -- the two sweeps disagree.

    Built so that val's best eligible theta (0.02) is not test's (0.05). If
    selection ran on the reported split, the reported theta would change.
    """
    from BaselineModels.xgb_baseline import best_theta

    val_sweep = pd.DataFrame({
        "theta": [0.01, 0.02, 0.05],
        "n_traded": [900, 400, 120],
        "pnl_per_1k_deployed": [-30.0, -5.0, -12.0],
    })
    test_sweep = pd.DataFrame({
        "theta": [0.01, 0.02, 0.05],
        "n_traded": [880, 380, 110],
        "pnl_per_1k_deployed": [-28.0, -19.0, +4.0],
    })
    assert best_theta(val_sweep) == 0.02
    assert best_theta(test_sweep) == 0.05
