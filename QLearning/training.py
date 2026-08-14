'''
train and evaluate the q learning agent
'''

import numpy as np
import pandas as pd
from data_preparation import prepare_episodes
from make_environment import TradingEnv
from q_learning_agent import QLearningAgent

DATA_DIR = "/Users/shreyanakum/Documents/HighRoller/data/polymarket/btc_updown_15m_candles_15s"

def evaluate(env: TradingEnv, agent: QLearningAgent):
    # run a deterministic eval loop where epsilon is 0 across all the episodes
    total_pnls = []

    for i in range(len(env.episodes)):
        state = env.reset(episode_index=i)
        done = False
        ep_pnl = 0.0

        while not done:
            action = agent.select_action(state, greedy=True)
            state, reward, done, _ = env.step(action)
            ep_pnl += reward

        total_pnls.append(ep_pnl)

    win_rate = (np.array(total_pnls) > 0).mean() * 100
    print(f'win rate: {win_rate}')
    print(f"num markets: {len(total_pnls)}")
    print(f"avg pnl per market: {np.mean(total_pnls)}")
    print(f"total pnl: {sum(total_pnls)}")

def main(DATA_FILE, NUM_EPISODES=5000):
    # data prep
    raw_df = pd.read_csv(DATA_FILE)
    episodes = prepare_episodes(raw_df)

    # make env and agent
    env = TradingEnv(episodes=episodes)
    agent = QLearningAgent(state_shape=env.state_space_size(), n_actions=env.n_actions())

    # training loop

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

        agent.decay_epsilon()

        if (ep + 1) % 500 == 0 or ep == 0:
            print(f"Episode {ep + 1}/{NUM_EPISODES}; Epsilon: {agent.epsilon}")


    agent.save() # save q table

    evaluate(env, agent)

# for file in os.listdir(DATA_DIR):
#     if not file.endswith('.csv'):
#         continue
file = f"{DATA_DIR}/btc_updown_15m_candles_15s_2026-02-26.csv"
main(file, 20)