import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from data_preparation import get_eval_data, get_test_data, get_training_data
from make_environment import TradingEnv
from q_learning_agent import QLearningAgent
from make_environment import StepInfo

from sim.evaluation import SETTLEMENT_CANDLE, score
from sim.execution import ACTION_NAMES, ExecutionConfig, HOLD, BUY_UP, BUY_DOWN, CLOSE

# repo-relative now instead of hardcoded to shreya's machine
SAVE_DIR = os.path.join(ROOT, "QLearning/figs")
METRICS_DIR = os.path.join(ROOT, "QLearning/metrics")
MODELS_DIR = os.path.join(ROOT, "QLearning/models")
OUT_DIR = os.path.join(ROOT, "QLearning/output")


def evaluate(env: TradingEnv, agent: QLearningAgent, save_to_csv: bool, env_name: str,
             model_name: str, iteration: int, seed=None):
    # greedy pass (epsilon=0) over every episode, builds rows in the same schema
    # the other models use so it can go through sim.evaluation.score() too
    market_rows = []
    all_ep_actions = []  # track actions across eval episodes

    for i in range(len(env.episodes)):
        state = env.reset(episode_index=i)
        done = False
        ep_actions = []

        while not done:
            valid_actions = env.get_valid_actions()
            action = agent.select_action(state, valid_actions=valid_actions, greedy=True)
            state, reward, done, info = env.step(action)
            ep_actions.append(action)

        ep = env._ep
        last = ep.iloc[-1]
        # Same interchange schema every other track writes (sim.metrics), so
        # compare_models.py scores the agent with the identical function.
        trades = env.portfolio.trades
        entries = [t for t in trades if t.action in (BUY_UP, BUY_DOWN)]
        closes = [t for t in trades if t.action == CLOSE]
        market_rows.append(
            dict(
                strategy=model_name,
                event_slug=str(last.event_slug),
                start_ts=int(ep.iloc[0].start_ts),
                split=env_name,
                stake=env.config.stake_dollars,
                pnl=env.portfolio.cash,
                fees=env.portfolio.fees_paid,
                stake_deployed=float(sum(t.shares * t.price for t in entries)),
                notional_traded=float(sum(t.shares * t.price for t in trades)),
                n_trades=len(entries),
                n_fills=len(trades),
                entry_candle=entries[0].candle_index if entries else None,
                exit_candle=(
                    closes[-1].candle_index if closes
                    else (SETTLEMENT_CANDLE if entries else None)
                ),
                early_exit=bool(closes),
                winner=str(last.winner),
                # Stamps the cost model this row was simulated under, so
                # compare_models.py can verify that "identical fees" actually
                # held rather than trusting that every track was launched with
                # the same flag. See sim.metrics.COST_MODEL_FIELD.
                slippage_frac=env.config.slippage_frac,
            )
        )
        all_ep_actions.append(ep_actions)

    mk = pd.DataFrame(market_rows)

    win_rate = float((mk.pnl > 0).mean() * 100) if len(mk) else 0.0
    print(f"win rate: {win_rate}")
    print(f"num markets: {len(mk)}")
    print(f"avg pnl per market: {mk.pnl.mean() if len(mk) else 0.0}")
    print(f"total pnl: {mk.pnl.sum() if len(mk) else 0.0}")

    if save_to_csv and len(mk):
        os.makedirs(METRICS_DIR, exist_ok=True)
        os.makedirs(OUT_DIR, exist_ok=True)

        s = score(mk)
        s["model"] = model_name
        s["iteration"] = iteration
        # The run seed, not agent.seed -- the agent's is derived from it, and
        # recording the derived one makes the column useless for re-running.
        s["seed"] = seed
        _append_row(metrics_path(model_family(model_name), env_name), pd.DataFrame([s]))

        # same schema as the other models so compare_models.py can just concat these
        _append_row(os.path.join(OUT_DIR, "markets.csv"), mk)

    return all_ep_actions


def model_family(model_name):
    """'qlearning_seed07' -> 'qlearning'. The per-run suffix drops off."""
    return model_name.split("_seed")[0].split("ITER")[0]


def metrics_path(family, env_name):
    return os.path.join(METRICS_DIR, f"{family}_{env_name}.csv")


def _append_row(path, df):
    """Append, but refuse to append a different schema onto an existing file.

    These files are written 30 times per sweep in append mode. pandas writes
    values in the frame's own column order without checking them against the
    header already on disk, so appending a changed schema silently produces a
    file whose columns do not mean what its header says. That is exactly what
    the committed pre-refactor metrics CSVs would have become when score()'s
    output changed.
    """
    if os.path.exists(path):
        header = list(pd.read_csv(path, nrows=0).columns)
        if header != list(df.columns):
            raise SystemExit(
                f"{path} already exists with a different schema.\n"
                f"  on disk: {header}\n"
                f"  writing: {list(df.columns)}\n"
                "Move the old file aside (or run reset_outputs) before re-running."
            )
    df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def reset_outputs(family):
    """Clear one model family's outputs so a sweep starts from empty.

    Every run of the 30-iteration sweep appended to the same markets.csv and
    metrics CSVs without ever truncating them, so a second sweep silently
    doubled every row and compare_models.py would score each market twice.
    """
    removed = []
    for env_name in ("train", "val", "test"):
        p = metrics_path(family, env_name)
        if os.path.exists(p):
            os.remove(p)
            removed.append(p)
    p = os.path.join(OUT_DIR, "markets.csv")
    if os.path.exists(p):
        os.remove(p)
        removed.append(p)
    for p in removed:
        print(f"  cleared {p}")


def main(model_name, iteration=0, NUM_EPISODES=5000, save_to_csv=False,
         config: ExecutionConfig | None = None, seed=None, episodes=None,
         eval_episodes=None):
    config = config or ExecutionConfig()
    if seed is None:
        seed = iteration

    # data prep. Passed in by run_sweep so 30 runs don't re-read the same
    # parquet 30 times; loaded here when main() is called on its own.
    if episodes is None:
        print("getting training data")
        episodes = get_training_data()

    # make env and agent. Two seeds off one run seed: the env samples which
    # markets get played, the agent samples exploration and tie-breaks.
    print(f"making environment and agent (seed {seed})")
    env = TradingEnv(episodes=episodes, config=config, seed=seed)
    agent = QLearningAgent(
        state_shape=env.state_space_size(), n_actions=env.n_actions(), seed=seed + 10_000
    )

    # Lists to track metrics over training
    rewards_history = []
    epsilon_history = []
    action_counts_history = [] # track distribution of actions
    train_action_sequences = [] # track full action sequence per training ep

    # training loop
    print("training begun")
    for ep in range(NUM_EPISODES):
        state = env.reset()
        done = False
        ep_reward = 0.0
        ep_actions = [] # actions in each episode

        while not done:
            valid_actions = env.get_valid_actions()
            action = agent.select_action(state, valid_actions=valid_actions)

            next_state, reward, done, _ = env.step(action)

            # get the valid actions for the state we just transitioned INTO
            next_valid_actions = env.get_valid_actions()

            agent.update(state, action, reward, next_state, done, next_valid_actions=next_valid_actions)

            state = next_state
            ep_reward += reward
            ep_actions.append(action)

        rewards_history.append(ep_reward)
        epsilon_history.append(agent.epsilon)

        # cnt frequency of each action (0, 1, 2, 3)
        n_actions = env.n_actions()
        ep_counts = [ep_actions.count(a) for a in range(n_actions)]
        action_counts_history.append(ep_counts)
        train_action_sequences.append(ep_actions)

        agent.decay_epsilon()

        if (ep + 1) % 500 == 0 or ep == 0:
            print(f"Episode {ep + 1}/{NUM_EPISODES}; Epsilon: {agent.epsilon}")

    # plotting & saving episodes
    os.makedirs(SAVE_DIR, exist_ok=True)
    episodes_axis = range(1, NUM_EPISODES + 1)

    # rwards over eps and lr curve
    plt.figure(figsize=(10, 5))
    plt.plot(episodes_axis, rewards_history, alpha=0.3, color="blue", label="Episode Reward")

    # smooth with a 100-episode moving average
    window = 100
    if len(rewards_history) >= window:
        moving_avg = pd.Series(rewards_history).rolling(window=window).mean()
        plt.plot(episodes_axis, moving_avg, color="red", linewidth=2, label=f"Learning Curve ({window}-Ep Avg)")

    plt.title(f"Training Rewards: {model_name}")
    plt.xlabel("Episode")
    plt.ylabel("Reward ($)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f"{model_name}_rewards.png"), dpi=300)
    plt.close()

    # epsilon decay plot
    plt.figure(figsize=(10, 4))
    plt.plot(episodes_axis, epsilon_history, color="purple")
    plt.title(f"Epsilon Decay: {model_name}")
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f"{model_name}_epsilon.png"), dpi=300)
    plt.close()

    # distr/freq of actions plot
    action_counts_arr = np.array(action_counts_history)
    plt.figure(figsize=(10, 5))
    for a in range(env.n_actions()):
        plt.plot(episodes_axis, action_counts_arr[:, a], label=ACTION_NAMES.get(a, f"Action {a}"), alpha=0.7)

    plt.title(f"Action Frequencies During Training: {model_name}")
    plt.xlabel("Episode")
    plt.ylabel("Action Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f"{model_name}_action_frequencies.png"), dpi=300)
    plt.close()

    print("saving q table")
    os.makedirs(MODELS_DIR, exist_ok=True)
    agent.save(f'{MODELS_DIR}/{model_name}.npy')  # save q table

    print("evaluating agent on training data")
    train_eval_actions = evaluate(env, agent, save_to_csv, "train", model_name, iteration, seed)

    print("evaluating agent")
    if eval_episodes is None:
        eval_episodes = get_eval_data()
    eval_env = TradingEnv(episodes=eval_episodes, config=config, seed=seed)
    eval_actions = evaluate(eval_env, agent, save_to_csv, "val", model_name, iteration, seed)

    # plot seq of actions in evaluation as a heatmap
    max_len = max(len(seq) for seq in eval_actions)
    seq_matrix = np.full((len(eval_actions), max_len), np.nan)
    for idx, seq in enumerate(eval_actions):
        seq_matrix[idx, :len(seq)] = seq

    plt.figure(figsize=(10, 6))
    plt.imshow(seq_matrix, aspect="auto", cmap="tab10", vmin=-0.5, vmax=env.n_actions() - 0.5)

    cbar = plt.colorbar(ticks=range(env.n_actions()))
    cbar.ax.set_yticklabels([ACTION_NAMES.get(a, f"Action {a}") for a in range(env.n_actions())])

    plt.title(f"Evaluation Action Sequences: {model_name}")
    plt.xlabel("Step")
    plt.ylabel("Episode")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f"{model_name}_eval_action_sequences.png"), dpi=300)
    plt.close()

    return agent


#: One run per seed. The report averaged 30 runs across two configurations it
#: called "70/20" and "80/20", but those names never described anything the code
#: does: get_training_data()/get_eval_data() read the canonical 70/15/15 temporal
#: split from BaselineModels.data_loader, and "0.8T_0.2E" was only ever a
#: filename prefix. There is no split-ratio parameter anywhere in this track.
#: The old ratios came from split_data.py, which partitioned day-files at random
#: -- the non-temporal split the proposal specifically ruled out -- and is marked
#: SUPERSEDED. So the two configurations are retired rather than reimplemented,
#: and the spread is measured across seeds on the one split every model shares.
DEFAULT_RUNS = 30
DEFAULT_EPISODES = 6789
MODEL_FAMILY = "qlearning"


def run_sweep(n_runs=DEFAULT_RUNS, num_episodes=DEFAULT_EPISODES,
              family=MODEL_FAMILY, config=None, fresh=True):
    """Train `n_runs` agents, one per seed, and score each on train and val.

    Holds a lock for the duration. Two sweeps writing the same family append
    into the same CSVs line by line, and one calling reset_outputs part-way
    through the other truncates it mid-run: a real overlap here produced 56 rows
    in a 30-run file and 188,708 duplicate market rows, with no error anywhere.
    Nothing downstream would have caught it -- score() would happily have
    averaged each market twice.
    """
    lock = os.path.join(OUT_DIR, f".{family}.sweep.lock")
    os.makedirs(OUT_DIR, exist_ok=True)
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise SystemExit(
            f"another sweep for family {family!r} is already running "
            f"(lock: {lock}).\nIf that is stale -- no python process is training "
            f"-- delete the file and retry."
        )
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    try:
        _run_sweep_locked(n_runs, num_episodes, family, config, fresh)
    finally:
        try:
            os.remove(lock)
        except OSError:
            pass


def _run_sweep_locked(n_runs, num_episodes, family, config, fresh):
    if fresh:
        reset_outputs(family)

    # Loaded once and shared: the episode frames are read-only, and re-reading
    # the parquet for each of 30 runs was most of the wall clock.
    print("loading episodes")
    train_episodes = get_training_data()
    eval_episodes = get_eval_data()

    for seed in range(n_runs):
        model_name = f"{family}_seed{seed:02d}"
        print(f"\n=== run {seed + 1}/{n_runs}: {model_name} ===")
        main(
            model_name,
            iteration=seed,
            seed=seed,
            NUM_EPISODES=num_episodes,
            save_to_csv=True,
            config=config,
            episodes=train_episodes,
            eval_episodes=eval_episodes,
        )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Train the Q-learning sweep.")
    ap.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    ap.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    ap.add_argument("--family", default=MODEL_FAMILY)
    ap.add_argument("--slippage", type=float, default=0.25,
                    help="ExecutionConfig.slippage_frac; project default is 0.25")
    ap.add_argument("--append", action="store_true",
                    help="keep existing outputs instead of clearing them first")
    args = ap.parse_args()

    run_sweep(
        n_runs=args.runs,
        num_episodes=args.episodes,
        family=args.family,
        config=ExecutionConfig(slippage_frac=args.slippage),
        fresh=not args.append,
    )
