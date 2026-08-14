import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from data_preparation import get_eval_data, get_test_data, get_training_data
from make_environment import TradingEnv
from q_learning_agent import QLearningAgent

SAVE_DIR = "/Users/shreyanakum/Documents/HighRoller/QLearning/figs"
METRICS_DIR = "/Users/shreyanakum/Documents/HighRoller/QLearning/metrics"
MODELS_DIR = "/Users/shreyanakum/Documents/HighRoller/QLearning/models"

def evaluate(env: TradingEnv, agent: QLearningAgent, save_to_csv: bool, env_name: str,model_name: str,iteration: int):
    # run a deterministic eval loop where epsilon is 0 across all the episodes
    total_pnls = []
    holding_periods = []
    total_fees = 0.0
    gross_pnl = 0.0

    for i in range(len(env.episodes)):
        state = env.reset(episode_index=i)
        done = False
        ep_pnl = 0.0
        steps = 0

        while not done:
            action = agent.select_action(state, greedy=True)
            state, reward, done, info = env.step(action)
            ep_pnl += reward
            steps += 1

            if isinstance(info, dict):
                total_fees += info.get("fee", 0.0)
                gross_pnl += info.get("gross_pnl", max(0.0, reward))

        total_pnls.append(ep_pnl)
        holding_periods.append(steps)

    pnls = np.array(total_pnls)
    total_pnl = float(sum(pnls))
    win_rate = float((pnls > 0).mean() * 100)
    avg_pnl_per_market = float(np.mean(pnls)) if len(pnls) > 0 else 0.0

    # Proposal Metrics
    pnl_per_1k_capital = (total_pnl / 1000.0)  # Cumulative PnL per $1,000 capital
    sharpe_ratio = float(avg_pnl_per_market / np.std(pnls)) if np.std(pnls) > 0 else 0.0

    # Max Drawdown calculation
    cum_pnls = np.cumsum(pnls)
    running_max = np.maximum.accumulate(cum_pnls)
    drawdowns = running_max - cum_pnls
    max_drawdown = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

    avg_holding_period = float(np.mean(holding_periods))
    turnover = float(np.sum(holding_periods))
    fee_fraction_gross_pnl = float(total_fees / gross_pnl if gross_pnl > 0 else 0.0)

    print(f"win rate: {win_rate}")
    print(f"num markets: {len(total_pnls)}")
    print(f"avg pnl per market: {avg_pnl_per_market}")
    print(f"total pnl: {total_pnl}")

    if save_to_csv:
        csv_path = f"{METRICS_DIR}/{model_name.split('ITER')[0]}_{env_name}.csv"
        metrics_df = pd.DataFrame(
            [
                {
                    "model": model_name,
                    "iteration": iteration,
                    "env_name": env_name,
                    "win_rate": win_rate,
                    "num_markets": len(total_pnls),
                    "avg_pnl_per_market": avg_pnl_per_market,
                    "total_pnl": total_pnl,
                    "pnl_per_1k_capital": pnl_per_1k_capital,
                    "sharpe_ratio": sharpe_ratio,
                    "max_drawdown": max_drawdown,
                    "turnover": turnover,
                    "total_fees_paid": total_fees,
                    "fee_fraction_gross_pnl": fee_fraction_gross_pnl,
                    "avg_holding_period": avg_holding_period,
                }
            ]
        )

        file_exists = os.path.exists(csv_path)
        metrics_df.to_csv(csv_path, mode="a", header=not file_exists, index=False)


def main(model_name, iteration=0, NUM_EPISODES=5000, save_to_csv=False):
    # data prep
    print("getting training data")
    episodes = get_training_data()

    # make env and agent
    print("making environment and agent")
    env = TradingEnv(episodes=episodes)
    agent = QLearningAgent(state_shape=env.state_space_size(), n_actions=env.n_actions())

    # Lists to track metrics over training
    rewards_history = []
    epsilon_history = []

    # training loop
    print("training begun")
    for ep in range(NUM_EPISODES):
        state = env.reset()
        done = False
        ep_reward = 0.0

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)

            agent.update(state, action, reward, next_state, done)

            state = next_state
            ep_reward += reward

        rewards_history.append(ep_reward)
        epsilon_history.append(agent.epsilon)

        agent.decay_epsilon()

        if (ep + 1) % 500 == 0 or ep == 0:
            print(f"Episode {ep + 1}/{NUM_EPISODES}; Epsilon: {agent.epsilon}")

    # --- Plotting & Saving Visuals ---
    os.makedirs(SAVE_DIR, exist_ok=True)
    episodes_axis = range(1, NUM_EPISODES + 1)

    # 1. Rewards over Episodes & Learning Curve (Moving Average)
    plt.figure(figsize=(10, 5))
    plt.plot(episodes_axis, rewards_history, alpha=0.3, color="blue", label="Episode Reward")
    
    # Smooth with a 100-episode moving average
    window = 100
    if len(rewards_history) >= window:
        moving_avg = pd.Series(rewards_history).rolling(window=window).mean()
        plt.plot(episodes_axis, moving_avg, color="red", linewidth=2, label=f"Learning Curve ({window}-Ep Avg)")

    plt.title(f"Training Rewards - {model_name}")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f"{model_name}_rewards.png"), dpi=300)
    plt.close()

    # 2. Epsilon Decay Plot
    plt.figure(figsize=(10, 4))
    plt.plot(episodes_axis, epsilon_history, color="purple")
    plt.title(f"Epsilon Decay - {model_name}")
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f"{model_name}_epsilon.png"), dpi=300)
    plt.close()

    print("saving q table")
    agent.save(f'{MODELS_DIR}/{model_name}')  # save q table

    print("evaluating agent on training data")
    evaluate(env, agent, save_to_csv, "training", model_name, iteration)

    print("evaluating agent")
    eval_episodes = get_eval_data()
    eval_env = TradingEnv(episodes=eval_episodes)
    evaluate(eval_env, agent, save_to_csv, "eval", model_name, iteration)

    # print('testing agent')
    # test_episodes = get_test_data()
    # test_env = TradingEnv(episodes=test_episodes)
    # evaluate(test_env, agent, save_to_csv, 'test', model_name, iteration)


if __name__ == "__main__":
    base_modelname = "0.7T_0.1E_qtable"
    for i in range(30):
        modelname = f"{base_modelname}ITER{i}"
        main(modelname, iteration=i, NUM_EPISODES=5000, save_to_csv=True)