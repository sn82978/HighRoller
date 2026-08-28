"""Tests for the cross-model comparison.

The comparison is the proposal's headline deliverable -- no-trade floor,
buy-and-hold, XGBoost at its swept threshold, and the Q-learning agent, "on
identical markets under identical fees". The fees are identical because every
track now goes through sim.execution; the *markets* are only identical if
something makes them so, which is what these tests cover.
"""

import pandas as pd
import pytest

from sim.compare_models import align_to_common_markets


def _rows(strategy, slugs, pnl=1.0):
    return pd.DataFrame([
        {"strategy": strategy, "event_slug": s, "split": "val", "start_ts": 100 + i,
         "stake": 100.0, "pnl": pnl, "fees": 0.5, "stake_deployed": 100.0,
         "notional_traded": 100.0, "n_trades": 1, "n_fills": 1,
         "entry_candle": 1, "exit_candle": 60, "early_exit": False, "winner": "Up"}
        for i, s in enumerate(slugs)
    ])


def test_alignment_restricts_everyone_to_the_shared_markets():
    """Different sample cuts must not become different denominators.

    On the real val split the baselines cover 1,343 markets and the rule
    strategies 1,332, because they drop for different reasons -- a warmup
    window versus a complete live window. Totalling PnL across those two sets
    and putting the results in one table compares different populations.
    """
    mk = pd.concat([
        _rows("baseline", ["a", "b", "c", "d"]),
        _rows("rule", ["b", "c", "d", "e"]),
    ], ignore_index=True)

    aligned, common, dropped = align_to_common_markets(mk)
    assert common == {"b", "c", "d"}
    assert dropped == {"baseline": 1, "rule": 1}
    assert aligned.groupby("strategy").event_slug.nunique().unique().tolist() == [3]


def test_alignment_is_a_noop_when_coverage_already_matches():
    mk = pd.concat([_rows("a", ["x", "y"]), _rows("b", ["x", "y"])], ignore_index=True)
    aligned, common, dropped = align_to_common_markets(mk)
    assert dropped == {}
    assert len(aligned) == len(mk)


def test_alignment_refuses_when_models_share_nothing():
    """Better to stop than to emit a table over an empty intersection."""
    mk = pd.concat([_rows("a", ["x"]), _rows("b", ["y"])], ignore_index=True)
    with pytest.raises(SystemExit, match="share no markets"):
        align_to_common_markets(mk)


def test_alignment_changes_the_reported_total():
    """If it did not, the tests above would be checking nothing that matters."""
    from sim.metrics import score_records

    mk = pd.concat([
        _rows("baseline", ["a", "b"], pnl=1.0),
        _rows("rule", ["b"], pnl=1.0),
    ], ignore_index=True)
    unaligned = score_records(mk[mk.strategy == "baseline"])["total_pnl"]
    aligned, _, _ = align_to_common_markets(mk)
    assert score_records(aligned[aligned.strategy == "baseline"])["total_pnl"] < unaligned


# -- identical fees, checked rather than assumed -------------------------
def _cost_rows(strategy, slugs, slippage):
    df = _rows(strategy, slugs)
    df["slippage_frac"] = slippage
    return df


def test_comparing_across_cost_models_is_refused():
    """The claim is 'identical markets under identical fees'. Check the fees.

    generate_trades.py defaulted --slippage 0.0 while every other track
    defaulted 0.25, so running each track's own documented command produced a
    table whose rows were priced differently -- worth 196 per $1k on
    momentum_flip, and a sign flip on its gross edge. Nothing detected it,
    because the cost model was not recorded anywhere in the outputs.
    """
    from sim.compare_models import check_one_cost_model

    mk = pd.concat([
        _cost_rows("rule", ["a", "b"], 0.0),
        _cost_rows("baseline", ["a", "b"], 0.25),
    ], ignore_index=True)
    with pytest.raises(SystemExit, match="different cost models"):
        check_one_cost_model(mk)


def test_one_shared_cost_model_passes_and_is_returned():
    from sim.compare_models import check_one_cost_model

    mk = pd.concat([
        _cost_rows("rule", ["a", "b"], 0.25),
        _cost_rows("baseline", ["a", "b"], 0.25),
    ], ignore_index=True)
    assert check_one_cost_model(mk) == 0.25


def test_a_track_with_two_cost_models_in_one_file_is_refused():
    """A half-regenerated markets.csv is worse than a stale one."""
    from sim.compare_models import check_one_cost_model

    mk = pd.concat([
        _cost_rows("rule", ["a"], 0.0),
        _cost_rows("rule", ["b"], 0.25),
    ], ignore_index=True)
    with pytest.raises(SystemExit, match="more than one"):
        check_one_cost_model(mk)


def test_missing_cost_column_warns_instead_of_passing_silently(capsys):
    from sim.compare_models import check_one_cost_model

    assert check_one_cost_model(_rows("rule", ["a", "b"])) is None
    assert "[warn]" in capsys.readouterr().out
