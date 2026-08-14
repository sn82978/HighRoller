'''
train and evaluate the q learning agent
'''

import numpy as np
import pandas as pd
from data_preparation import get_training_data, get_eval_data, get_test_data
from make_environment import TradingEnv
from q_learning_agent import QLearningAgent

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

def main(NUM_EPISODES=5000):
    # data prep
    print('getting training data')
    episodes = get_training_data()

    # make env and agent
    print('making environment and agent')
    env = TradingEnv(episodes=episodes)
    agent = QLearningAgent(state_shape=env.state_space_size(), n_actions=env.n_actions())

    # training loop
    print('training begun')
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

    print('saving q table')
    agent.save() # save q table

    print('evaluating agent')
    eval_episodes = get_eval_data()
    eval_env = TradingEnv(episodes=eval_episodes)
    evaluate(eval_env, agent)

    # print('testing agent')
    # test_episodes = get_test_data()
    # test_env = TradingEnv(episodes=test_episodes)
    # evaluate(test_env, agent)

if __name__ == '__main__':
    main(NUM_EPISODES=5000)