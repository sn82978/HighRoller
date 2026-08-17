"""Tests for scoring.

A few of these exist to pin down definitions that are easy to get wrong in a
way that flatters the results.
"""

import numpy as np
import pandas as pd
import pytest

from BaselineModels.metrics import (
    MARKETS_PER_YEAR,
    MarketResult,
    brier,
    calibration_table,
    comparison_table,
    expected_calibration_error,
    log_loss,
    paired_bootstrap_logloss,
    probability_metrics,
    results_frame,
    trading_metrics,
)


def _res(slug, pnl, fees=0.0, stake=100.0, notional=100.0, n=1, entry=1, exit_=60, early=False):
    return MarketResult(
        event_slug=slug,
        pnl=pnl,
        fees=fees,
        stake_deployed=stake,
        notional_traded=notional,
        n_trades=n,
        entry_candle=entry,
        exit_candle=exit_,
        winner="Up",
        early_exit=early,
    )


# -- probability metrics -------------------------------------------------
def test_log_loss_of_a_coin_flip_is_ln_2():
    assert log_loss(np.array([0, 1]), np.array([0.5, 0.5])) == pytest.approx(np.log(2))


def test_brier_of_a_coin_flip_is_a_quarter():
    assert brier(np.array([0, 1]), np.array([0.5, 0.5])) == pytest.approx(0.25)


def test_perfect_forecast_scores_zero():
    y = np.array([0, 1, 1, 0])
    p = np.array([0.0, 1.0, 1.0, 0.0])
    assert log_loss(y, p) == pytest.approx(0.0, abs=1e-10)
    assert brier(y, p) == pytest.approx(0.0)


def test_log_loss_is_finite_on_a_confidently_wrong_prediction():
    """Without clipping this is inf, which poisons any average it lands in."""
    assert np.isfinite(log_loss(np.array([1]), np.array([0.0])))


def test_a_perfectly_calibrated_forecaster_has_near_zero_ece():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.05, 0.95, 200_000)
    y = (rng.uniform(size=p.size) < p).astype(int)
    assert expected_calibration_error(y, p) < 0.01


def test_a_systematically_overconfident_forecaster_has_large_ece():
    y = np.zeros(1000, dtype=int)
    p = np.full(1000, 0.9)  # says 90%, happens 0% of the time
    assert expected_calibration_error(y, p) == pytest.approx(0.9, abs=1e-6)


def test_calibration_table_recovers_the_observed_rate():
    y = np.array([1, 1, 0, 0, 1, 1, 1, 1])
    p = np.full(8, 0.75)
    tbl = calibration_table(y, p, bins=10)
    row = tbl[tbl.n > 0].iloc[0]
    assert row.n == 8
    assert row.observed_rate == pytest.approx(0.75)


def test_auc_is_nan_when_only_one_class_is_present():
    m = probability_metrics(np.ones(10), np.full(10, 0.7))
    assert np.isnan(m["auc"])


# -- clustered significance ----------------------------------------------
def test_bootstrap_detects_a_genuinely_better_model():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(400), 60)
    truth = rng.uniform(0.1, 0.9, 400)
    y = (rng.uniform(size=400) < truth).astype(int).repeat(60)
    p_model = truth.repeat(60)
    p_market = np.full(y.size, 0.5)
    out = paired_bootstrap_logloss(y, p_model, p_market, groups, n_boot=500)
    assert out["mean_improvement"] > 0
    assert out["ci_lo"] > 0
    assert out["p_model_better"] > 0.95


def test_bootstrap_reports_no_edge_for_an_identical_model():
    rng = np.random.default_rng(1)
    groups = np.repeat(np.arange(300), 60)
    y = rng.integers(0, 2, 300).repeat(60)
    p = np.full(y.size, 0.5)
    out = paired_bootstrap_logloss(y, p, p, groups, n_boot=500)
    assert out["mean_improvement"] == pytest.approx(0.0, abs=1e-12)
    assert out["ci_lo"] <= 0 <= out["ci_hi"]


def test_bootstrap_counts_markets_not_rows():
    """60 candles from one market shouldn't count as 60 observations."""
    groups = np.repeat(np.arange(50), 60)
    y = np.tile(np.repeat([0, 1], 60), 25)
    p = np.full(y.size, 0.5)
    out = paired_bootstrap_logloss(y, p, p, groups, n_boot=100)
    assert out["n_markets"] == 50


# -- trading metrics -----------------------------------------------------
def test_pnl_per_1k_normalises_by_capital_actually_deployed():
    """Not total PnL / 1000, which ignores how much was risked to get it."""
    small = trading_metrics([_res("a", 10.0, stake=100.0)])
    big = trading_metrics([_res("b", 100.0, stake=1000.0)])
    assert small["pnl_per_1k_deployed"] == pytest.approx(100.0)
    assert big["pnl_per_1k_deployed"] == pytest.approx(100.0)


def test_fee_fraction_is_nan_when_gross_pnl_is_not_positive():
    """A hard 0.0 here usually means the fee accumulator was never wired up."""
    m = trading_metrics([_res("a", -50.0, fees=3.5)])
    assert np.isnan(m["fee_fraction_gross_pnl"])


def test_fee_fraction_is_the_share_of_gross_pnl_eaten_by_fees():
    m = trading_metrics([_res("a", 6.5, fees=3.5)])  # gross 10, fees 3.5
    assert m["fee_fraction_gross_pnl"] == pytest.approx(0.35)


def test_a_policy_that_never_trades_scores_zero_not_nan_pnl():
    m = trading_metrics([_res("a", 0.0, stake=0.0, notional=0.0, n=0, entry=None, exit_=None)])
    assert m["total_pnl"] == 0.0
    assert m["n_traded"] == 0
    assert m["pnl_per_1k_deployed"] == 0.0
    assert np.isnan(m["win_rate"])


def test_max_drawdown_measures_the_worst_peak_to_trough():
    results = [_res("a", 10.0), _res("b", -30.0), _res("c", 5.0)]
    assert trading_metrics(results)["max_drawdown"] == pytest.approx(30.0)


def test_max_drawdown_is_zero_for_a_monotonically_rising_curve():
    results = [_res("a", 1.0), _res("b", 2.0), _res("c", 3.0)]
    assert trading_metrics(results)["max_drawdown"] == pytest.approx(0.0)


def test_turnover_is_notional_over_capital_not_a_step_count():
    """One entry held to settlement is turnover 1, not 60."""
    m = trading_metrics([_res("a", 5.0, stake=100.0, notional=100.0)])
    assert m["turnover"] == pytest.approx(1.0)


def test_a_round_trip_doubles_turnover():
    m = trading_metrics([_res("a", 5.0, stake=100.0, notional=200.0)])
    assert m["turnover"] == pytest.approx(2.0)


def test_sharpe_is_annualised_over_markets_per_year():
    results = [_res(f"m{i}", p) for i, p in enumerate([10.0, -5.0, 7.0, -2.0, 4.0])]
    rets = np.array([r / 100.0 for r in (10.0, -5.0, 7.0, -2.0, 4.0)])
    expected = rets.mean() / rets.std(ddof=1) * np.sqrt(MARKETS_PER_YEAR)
    assert trading_metrics(results)["sharpe"] == pytest.approx(expected)


def test_sharpe_is_nan_with_no_variation():
    assert np.isnan(trading_metrics([_res("a", 5.0)])["sharpe"])


def test_holding_period_excludes_markets_that_never_entered():
    results = [_res("a", 0.0, stake=0.0, n=0, entry=None, exit_=None), _res("b", 5.0, entry=10, exit_=60)]
    assert trading_metrics(results)["avg_holding_candles"] == pytest.approx(50.0)


def test_win_rate_counts_only_markets_that_traded():
    results = [_res("a", 0.0, stake=0.0, n=0, entry=None, exit_=None), _res("b", 5.0), _res("c", -1.0)]
    assert trading_metrics(results)["win_rate"] == pytest.approx(0.5)


def test_scoring_an_empty_run_raises_rather_than_returning_zeros():
    with pytest.raises(ValueError):
        trading_metrics([])


# -- assembly ------------------------------------------------------------
def test_results_frame_has_one_row_per_market():
    df = results_frame([_res("a", 1.0), _res("b", 2.0)])
    assert len(df) == 2
    assert set(df.event_slug) == {"a", "b"}


def test_comparison_table_stacks_policies():
    a = trading_metrics([_res("a", 1.0)])
    b = trading_metrics([_res("b", 2.0)])
    tbl = comparison_table({"x": a, "y": b})
    assert list(tbl.policy) == ["x", "y"]
    assert tbl.loc[tbl.policy == "y", "total_pnl"].iloc[0] == pytest.approx(2.0)
