'''
takes the data_preparation episodes list and makes environments for the Q-learning agent to train in.

q-table:
    shape = (price buckets, time buckets, positions, pnl buckets, actions) = (10, 60, 3, 5, 4) = 9000 states
    so for each state the q table stores 4 number that estimates the value of holding, buying up, buying down, and selling

    - price_bucket (there's 10 total this is j a snippet):
        0-0.09: heavy losing bet
        0.40-0.49: toss up (slight below 50%)
        0.50-0.59: toss up (slight above 50%)
        0.90-1.00: heavy winning bet

    - time_buckets is based on candle pos (0 to 59)

    - positions: FLAT, LONG_UP, LONG_DOWN

    - pnl_bucket:
        <= -0.30: heavy loss 
        -0.30 to -0.10: slight loss
        0: neutral
        +0.10 to +0.30: slight profit
        >= +0.30: heavy profit 

    - actions: HOLD, BUY_UP, BUY_DOWN, SELL

https://gymnasium.farama.org/introduction/basic_usage/
'''

import numpy as np
import pandas as pd

# actions we can take
HOLD = 0
BUY_UP = 1
BUY_DOWN = 2
SELL = 3

ACTION_NAMES = {HOLD: "Hold", BUY_UP: "Buy Up", BUY_DOWN : "Buy Down", SELL: "Sell"}

# current market position (whether we hold a trade or not)
FLAT = 0 # we hold no shares
LONG_UP = 1 # we bought up shares
LONG_DOWN = 2 # we bought down shares

N_PRICE_BUCKETS = 10 # divides the share price (0 to 1) into 10 bins
N_PNL_BUCKETS = 5 # divides ou unrealized pnl into 5 bins when we hold a position
MAX_TIME_BUCKET = 59 # how much time is left in the market in 15 * 4 = 60 candles

# internal class to keep info going on in episodes
class StepInfo:
    def __init__(self, action_name, was_valid, position, realized_pnl):
        self.action_name = action_name
        self.was_valid = was_valid
        self.position = position
        self.realized_pnl = realized_pnl

    def __repr__(self):
        return f"StepInfo(action_name={self.action_name}, was_valid={self.was_valid}, position={self.position}, realized_pnl={self.realized_pnl})"

    def __eq__(self, other):
        if not isinstance(other, StepInfo):
            return False

        return (self.action_name, self.was_valid, self.position, self.realized_pnl) == (other.action_name, other.was_valid, other.position, other.self.realized_pnl)

class TradingEnv:

    def __init__(self, episodes):
        if not episodes:
            print("no episodes")

        self.episodes = episodes
        self._ep = None # the current market df being played
        self._i = 0 # which candle row are we in rn
        self.position = FLAT # we have no shares at first
        self.entry_price= 0.0 # track the buy-in price of pos we r in rn
        self._rng = np.random.default_rng() # initalizes the rng to pick a random market next time

    def _get_state(self):
        row = self._ep.iloc[self._i]
        
        # time bucket safely bounded [0, 59]
        time_bucket = max(0, min(int(row["candle_index"]), MAX_TIME_BUCKET))

        # cut the price of curr position
        if self.position == LONG_UP:
            cur_price = row["price_up"]
        elif self.position == LONG_DOWN:
            cur_price = row["price_down"]
        else:
            cur_price = row["price_up"]

        # calculate the price bucket
        price_bucket = min(int(cur_price * N_PRICE_BUCKETS), N_PRICE_BUCKETS - 1)
        price_bucket = max(0, price_bucket)

        # calculate pnl bucket
        if self.position == FLAT:
            pnl_bucket = N_PNL_BUCKETS//2 # bucket 2 is neutral
        else:
            unrealized = cur_price - self.entry_price
            clipped = max(-0.5, min(0.5, unrealized)) # range: [-0.5, +0.5]
            
            # scale [-0.5, 0.5] -> [0.0, 1.0], multiply by 5, convert to int
            normalized_pnl = (clipped + 0.5)/1.0
            pnl_bucket = int(normalized_pnl * N_PNL_BUCKETS)
            
            # clamp bounds strictly to [0, 4]
            pnl_bucket = min(max(0, pnl_bucket), N_PNL_BUCKETS - 1)

        return (price_bucket, time_bucket, self.position, pnl_bucket)

    def reset(self, episode_index):
        index = episode_index
        if episode_index is None:
            index = self._rng.integers(0, len(self.episodes))

        self._ep = self.episodes[index].reset_index(drop=True)
        self._i =0
        self.position = FLAT
        self.entry_price = 0.0
        return self._get_state()

    def step(self, action):
        row = self._ep.iloc[self._i]
        price_up, price_down = row["price_up"], row["price_down"]
        is_last_step = self._i == len(self._ep)-1

        reward = 0.0
        was_valid = True

        if action == BUY_UP:
            if self.position == FLAT:
                self.position = LONG_UP
                self.entry_price = price_up
            else:
                was_valid = False

        elif action == BUY_DOWN:
            if self.position == FLAT:
                self.position = LONG_DOWN
                self.entry_price = price_down
            else:
                was_valid = False

        elif action == SELL:
            if self.position == LONG_UP:
                reward = price_up - self.entry_price
                self.position, self.entry_price = FLAT, 0.0
            elif self.position == LONG_DOWN:
                reward = price_down - self.entry_price
                self.position, self.entry_price = FLAT, 0.0
            else:
                was_valid = False

        # forced to settle at episode end if holding position
        if is_last_step and self.position != FLAT:
            winner = str(row["winner"]).strip().capitalize
            if self.position == LONG_UP:
                held_outcome = "Up"
            else:
                held_outcome = "Down"

            if held_outcome == winner:
                payout = 1.0
            else:
                payout = 0.0

            reward = payout - self.entry_price

            self.position = FLAT
            self.entry_price = 0.0

        info = StepInfo(ACTION_NAMES[action], was_valid, self.position, reward)
        done = is_last_step

        if not done:
            self._i += 1

        next_state = self._get_state()
        return next_state, reward, done, info