'''
tabular q-learning agent
'''

import numpy as np

class QLearningAgent:
    def __init__(self, state_shape, n_actions=4, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.02):
        self.q_table = np.zeros(state_shape + (n_actions,))
        self.alpha = alpha # lr
        self.gamma = gamma # how much agent cares ab future rewards v immediate rewards
        self.epsilon = epsilon # initial exploration rate
        self.epsilon_decay = epsilon_decay # multiplier used to gradually decrease epsilon after each training ep
        self.min_epsilon = min_epsilon # lowest allowed val for epsilon
        self.n_actions = n_actions # total actions we can take

    def select_action(self, state, greedy=False):
        # select an action using epsilon greedy

        if not greedy and np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions) # random exploration choice

        return int(np.argmax(self.q_table[state])) # choose the largest q-val  

    def update(self, state, action, reward, next_state, done):
        # std q learning td update rule
        best_next_action = np.argmax(self.q_table[next_state])
        target = reward
        if not done:
            target += self.gamma * self.q_table[next_state][best_next_action]

        td_error = target - self.q_table[state][action]

        self.q_table[state][action] += self.alpha * td_error

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def save(self, filepath: str = "q_table.npy"):
        np.save(filepath, self.q_table)

    def load(self, filepath: str = "q_table.npy"):
        self.q_table = np.load(filepath)
    