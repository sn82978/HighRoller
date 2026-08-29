"""Tests for the Q-learning track.

This track had zero tests, which is how a hardcoded allow_test=True, an unseeded
RNG and a blind append all managed to survive in it at the same time.

The reproducibility tests are the important ones here. The report averages 30
"independent runs", but until these passed, independent just meant "drawn from
the unseeded global np.random", so none of the reported spread could be
reproduced -- not on another machine, and not twice on the same one.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

QL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "QLearning")
if QL_DIR not in sys.path:
    sys.path.insert(0, QL_DIR)

from make_environment import FLAT, LONG_DOWN, LONG_UP, TradingEnv  # noqa: E402
from q_learning_agent import QLearningAgent  # noqa: E402
from sim.execution import BUY_DOWN, BUY_UP, CLOSE, ExecutionConfig, HOLD, Side  # noqa: E402

from conftest import make_market  # noqa: E402

FRICTIONLESS = ExecutionConfig(slippage_frac=0.0, stake_dollars=100.0)


def _episodes(n=3):
    """Live-window episodes in the shape data_preparation.prepare_episodes emits."""
    out = []
    for i in range(n):
        m = make_market(f"btc-updown-15m-{i}", 1_000_000_000 + 900 * i,
                        winner="Up" if i % 2 else "Down", seed=i)
        m = m[m.candle_index >= 0].reset_index(drop=True)
        for col in ("open", "high", "low"):
            m[f"next_{col}"] = m[col].shift(-1)
        m["price_up"] = m["close"]
        m["price_down"] = 1.0 - m["price_up"]
        out.append(m)
    return out


def _train(seed, n_episodes=40, config=FRICTIONLESS):
    eps = _episodes()
    env = TradingEnv(episodes=eps, config=config, seed=seed)
    agent = QLearningAgent(
        state_shape=env.state_space_size(), n_actions=env.n_actions(), seed=seed + 10_000
    )
    for _ in range(n_episodes):
        state = env.reset()
        done = False
        while not done:
            action = agent.select_action(state, valid_actions=env.get_valid_actions())
            nxt, reward, done, _ = env.step(action)
            agent.update(state, action, reward, nxt, done,
                         next_valid_actions=env.get_valid_actions())
            state = nxt
        agent.decay_epsilon()
    return agent.q_table


# -- reproducibility -------------------------------------------------------
def test_same_seed_reproduces_the_same_q_table():
    a, b = _train(0), _train(0)
    assert np.array_equal(a, b), "a seeded run must be reproducible"
    assert (a != 0).sum() > 0, "a run that learned nothing would match trivially"


def test_different_seeds_diverge():
    assert not np.array_equal(_train(0), _train(1))


def test_env_episode_sampling_is_seeded():
    """Which markets a run happens to visit is part of that run."""
    def visited(seed):
        env = TradingEnv(episodes=_episodes(8), config=FRICTIONLESS, seed=seed)
        return [env.reset() and env._ep.event_slug.iloc[0] for _ in range(20)]

    assert visited(3) == visited(3)
    assert visited(3) != visited(4)


def test_agent_does_not_touch_global_numpy_random():
    """Seeding does nothing if some other agent is messing with the global stream."""
    np.random.seed(1234)
    before = np.random.rand()
    np.random.seed(1234)
    _train(7, n_episodes=5)
    assert np.random.rand() == before


# -- action masking --------------------------------------------------------
def test_masking_offers_only_legal_actions():
    env = TradingEnv(episodes=_episodes(1), config=FRICTIONLESS, seed=0)
    env.reset(episode_index=0)
    assert set(env.get_valid_actions()) == {HOLD, BUY_UP, BUY_DOWN}
    env.step(BUY_UP)
    assert set(env.get_valid_actions()) == {HOLD, CLOSE}, "cannot open a second leg"


def test_agent_never_selects_a_masked_action():
    env = TradingEnv(episodes=_episodes(1), config=FRICTIONLESS, seed=0)
    agent = QLearningAgent(
        state_shape=env.state_space_size(), n_actions=env.n_actions(), seed=0
    )
    state = env.reset(episode_index=0)
    done = False
    while not done:
        valid = env.get_valid_actions()
        action = agent.select_action(state, valid_actions=valid)
        assert action in valid
        state, _, done, _ = env.step(action)


# -- execution -------------------------------------------------------------
def test_entry_price_is_the_filled_price_not_the_unslipped_mid():
    """The pnl_bucket the agent sees has to include the spread it just paid.

    Recording next_open here told the agent it was flat on a position it had
    just paid slippage to open, so the neutral PnL bucket absorbed trades that
    were already down by the half-spread.
    """
    config = ExecutionConfig(slippage_frac=0.25, stake_dollars=100.0)
    env = TradingEnv(episodes=_episodes(1), config=config, seed=0)
    env.reset(episode_index=0)
    row = env._ep.iloc[0]
    env.step(BUY_UP)
    assert env.portfolio.side is Side.UP
    assert env._entry_price == pytest.approx(env.portfolio.trades[-1].price)
    assert env._entry_price > row["next_open"], "an Up buy fills above the mid"


def test_trades_are_stamped_with_the_candle_they_fill_on():
    env = TradingEnv(episodes=_episodes(1), config=FRICTIONLESS, seed=0)
    env.reset(episode_index=0)
    signal_candle = int(env._ep.iloc[0].candle_index)
    env.step(BUY_UP)
    assert env.portfolio.trades[-1].candle_index == signal_candle + 1


def test_settlement_is_free_and_forced_at_the_last_candle():
    env = TradingEnv(episodes=_episodes(1), config=FRICTIONLESS, seed=0)
    env.reset(episode_index=0)
    env.step(BUY_UP)
    fees_after_entry = env.portfolio.fees_paid
    done = False
    while not done:
        _, _, done, _ = env.step(HOLD)
    assert env.portfolio.settled
    assert env.portfolio.side is Side.FLAT
    assert env.portfolio.fees_paid == pytest.approx(fees_after_entry), "redemption is free"


def test_gross_pnl_is_not_clamped_to_positive():
    """The old clamp threw away losing steps but kept their fees.

    That is the 21.5x fee-drag discrepancy the progress report flags in 3.3.
    """
    env = TradingEnv(episodes=_episodes(1), config=FRICTIONLESS, seed=0)
    env.reset(episode_index=0)
    infos = [env.step(BUY_UP)[3]]
    done = False
    while not done:
        _, _, done, info = env.step(HOLD)
        infos.append(info)
    assert any(i.gross_pnl < 0 for i in infos), (
        "no losing step in this episode -- the clamp would be untestable here"
    )


def test_empty_episode_list_raises_instead_of_printing():
    with pytest.raises(ValueError, match="no episodes"):
        TradingEnv(episodes=[], config=FRICTIONLESS)


# -- the held-out split has to be asked for out loud ---------------------
def test_evaluate_split_refuses_test_without_the_flag():
    """You only get one scored pass, so you have to ask for it on purpose.

    evaluate_split.py exists so you can score the test split without retraining,
    which also makes it the easiest place in the repo to accidentally burn the
    one held-out run we get.
    """
    import evaluate_split

    with pytest.raises(SystemExit, match="allow-test"):
        evaluate_split.main(["--split", "test"])


def test_evaluate_split_refuses_to_rescore_a_split_it_already_scored(tmp_path, monkeypatch):
    """A second pass would append and double every market in the table.

    And it'd do it silently -- score() would just average each market twice and
    nothing downstream checks for that. Same thing two concurrent training
    sweeps did before run_sweep started taking a lock.
    """
    import evaluate_split

    models = tmp_path / "models"
    models.mkdir()
    np.save(models / "qlearning_seed00.npy", np.zeros((10, 60, 3, 5, 4)))
    metrics = tmp_path / "metrics"
    metrics.mkdir()
    (metrics / "qlearning_val.csv").write_text("already scored\n")

    monkeypatch.setattr(evaluate_split, "MODELS_DIR", str(models))
    monkeypatch.setattr(evaluate_split, "metrics_path",
                        lambda fam, split: str(metrics / f"{fam}_{split}.csv"))

    with pytest.raises(SystemExit, match="already exists"):
        evaluate_split.main(["--split", "val"])
