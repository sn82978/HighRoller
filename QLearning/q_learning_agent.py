import numpy as np

class QLearningAgent:
    """Tabular Q-learning, with actions masked by what position we're holding.

    Every random draw goes through self.rng, a seeded Generator, instead of the
    global np.random. Exploration, tie-breaking and the env's episode sampling
    all used to run off the unseeded global, which meant the "30 independent
    runs" the report averages couldn't be reproduced at all -- not on another
    machine, and not even twice on the same one.
    """

    def __init__(self, state_shape, n_actions=4, alpha=0.1, gamma=0.99, epsilon=1.0,
                 epsilon_decay=0.995, min_epsilon=0.02, seed=None):
        self.q_table = np.zeros(state_shape + (n_actions,))
        self.alpha = alpha # lr
        self.gamma = gamma # how much agent cares ab future rewards v immediate rewards
        self.epsilon = epsilon # initial exploration rate
        self.epsilon_decay = epsilon_decay # multiplier used to gradually decrease epsilon after each training ep
        self.min_epsilon = min_epsilon # lowest allowed val for epsilon
        self.n_actions = n_actions # total actions we can take
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def select_action(self, state, valid_actions=None, greedy=False):
        # mask invalid actions if provided
        if valid_actions is None:
            valid_actions = list(range(self.n_actions))

        if not greedy and self.rng.random() < self.epsilon:
            return int(self.rng.choice(valid_actions)) # pick only from valid actions

        # choose largest q-val among valid actions
        q_vals = self.q_table[state]
        masked_q_vals = np.full_like(q_vals, -np.inf)
        masked_q_vals[valid_actions] = q_vals[valid_actions]

        # breaking ties randomly prevents the agent from getting stuck always choosing action 0 (HOLD) when all Q-vals are 0.0
        max_val = np.max(masked_q_vals)
        best_actions = [a for a in valid_actions if masked_q_vals[a] == max_val]
        return int(self.rng.choice(best_actions))

    def update(self, state, action, reward, next_state, done, next_valid_actions=None):
        # mask the next state's actions so we don't bootstrap from impossible actions
        if next_valid_actions is None:
            next_valid_actions = list(range(self.n_actions))
            
        target = reward
        if not done:
            # Only consider valid actions for the next state's max Q-value
            next_q_vals = self.q_table[next_state]
            max_next_q = np.max(next_q_vals[next_valid_actions])
            target += self.gamma * max_next_q

        td_error = target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def save(self, filepath= "q_table.npy"):
        np.save(filepath, self.q_table)

    def load(self, filepath="q_table.npy"):
        self.q_table = np.load(filepath)