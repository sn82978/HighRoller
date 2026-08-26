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
