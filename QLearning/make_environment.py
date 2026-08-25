'''
takes the data_preparation episodes list and makes environments for the Q-learning agent to train in.

fills/fees/slippage go through sim.execution.Portfolio now (same engine the other models
use) instead of the old hand-rolled fee math, which also filled trades on the same candle
close it signaled on -- basically lookahead. now it fills on the next candle's open like
everything else, and reward is just the $ change in portfolio value per step.

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

    - actions: HOLD, BUY_UP, BUY_DOWN, SELL (CLOSE)

https://gymnasium.farama.org/introduction/basic_usage/
'''

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd

from sim.execution import ACTIONS, BUY_DOWN, BUY_UP, CLOSE, ExecutionConfig, HOLD, Portfolio, Side

ACTION_NAMES = {HOLD: "Hold", BUY_UP: "Buy Up", BUY_DOWN: "Buy Down", CLOSE: "Sell"}

# position bucket ints used by the q-table's state index (not sim.execution.Side,
# whose DOWN=-1 can't index an array).
FLAT = 0
LONG_UP = 1
LONG_DOWN = 2
_SIDE_TO_BUCKET = {Side.FLAT: FLAT, Side.UP: LONG_UP, Side.DOWN: LONG_DOWN}

N_PRICE_BUCKETS = 10 # divides the share price (0 to 1) into 10 bins
N_PNL_BUCKETS = 5 # divides ou unrealized pnl into 5 bins when we hold a position
MAX_TIME_BUCKET = 59 # how much time is left in the market in 15 * 4 = 60 candles


# internal class to keep info going on in episodes
class StepInfo:
    def __init__(self, action_name, was_valid, position, realized_pnl, fee=0.0, gross_pnl=0.0):
        self.action_name = action_name
        self.was_valid = was_valid
        self.position = position
        self.realized_pnl = realized_pnl
        self.fee = fee
        self.gross_pnl = gross_pnl

    def get(self, key, default=None):
        return getattr(self, key, default)

    def __getitem__(self, item):
        return getattr(self, item)

class TradingEnv:

    def __init__(self, episodes, config: ExecutionConfig | None = None):
        if not episodes:
            print("no episodes")

        self.episodes = episodes
        self.config = config or ExecutionConfig()
        self._ep = None # the current market df being played
        self._i = 0 # which candle row are we in rn
        self.portfolio = Portfolio(config=self.config)
        self._entry_price = 0.0 # Up-equivalent price the open position was entered at
        self._rng = np.random.default_rng() # initalizes the rng to pick a random market next time

    def _get_state(self):
        row = self._ep.iloc[self._i]
        position = _SIDE_TO_BUCKET[self.portfolio.side]

        # time bucket safely bounded [0, 59]
        time_bucket = max(0, min(int(row["candle_index"]), MAX_TIME_BUCKET))

        # current mark price of whichever leg (if any) we hold
        if position == LONG_UP:
            curr_price = row["price_up"]
        elif position == LONG_DOWN:
            curr_price = row["price_down"]
        else:
            curr_price = row["price_up"]

        # calculate the price bucket
        price_bucket = min(int(curr_price * N_PRICE_BUCKETS), N_PRICE_BUCKETS - 1)
        price_bucket = max(0, price_bucket)

        # calculate pnl bucket
        if position == FLAT:
            pnl_bucket = N_PNL_BUCKETS // 2 # bucket 2 is neutral
        else:
            unrealized = curr_price - self._entry_price
            clipped = max(-0.5, min(0.5, unrealized)) # range: [-0.5, +0.5]

            # scale [-0.5, 0.5] -> [0.0, 1.0], multiply by 5, convert to int
            normalized_pnl = (clipped + 0.5) / 1.0
            pnl_bucket = int(normalized_pnl * N_PNL_BUCKETS)

            # clamp bounds strictly to [0, 4]
            pnl_bucket = min(max(0, pnl_bucket), N_PNL_BUCKETS - 1)

        return (price_bucket, time_bucket, position, pnl_bucket)

    @staticmethod
    def state_space_size():
        return (N_PRICE_BUCKETS, MAX_TIME_BUCKET + 1, 3, N_PNL_BUCKETS)

    @staticmethod
    def n_actions():
        return len(ACTIONS)

    def get_valid_actions(self):
        # returns list of allowed actions based on current position
        if self.portfolio.side is Side.FLAT:
            return [HOLD, BUY_UP, BUY_DOWN]
        return [HOLD, CLOSE]

    def reset(self, episode_index=None):
        index = episode_index
        if episode_index is None:
            index = self._rng.integers(0, len(self.episodes))

        self._ep = self.episodes[index].reset_index(drop=True)
        self._i = 0
        self.portfolio = Portfolio(config=self.config)
        self._entry_price = 0.0
        return self._get_state()

    def step(self, action):
        row = self._ep.iloc[self._i]
        is_last_step = self._i == len(self._ep) - 1

        value_before = self.portfolio.value(row["price_up"])
        was_valid = True
        fee_before = self.portfolio.fees_paid

        if action == HOLD:
            pass
        elif action == CLOSE:
            if self.portfolio.side is Side.FLAT:
                was_valid = False
            elif is_last_step or pd.isna(row.get("next_open")):
                was_valid = False  # no next candle to fill a close against
            else:
                was_valid = self.portfolio.close(
                    row["next_open"], row["next_high"], row["next_low"], int(row["candle_index"])
                )
                if was_valid:
                    self._entry_price = 0.0
        elif action in (BUY_UP, BUY_DOWN):
            if self.portfolio.side is not Side.FLAT:
                was_valid = False
            elif is_last_step or pd.isna(row.get("next_open")):
                was_valid = False  # no next candle to fill an entry against
            else:
                side = Side.UP if action == BUY_UP else Side.DOWN
                was_valid = self.portfolio.buy(
                    side, row["next_open"], row["next_high"], row["next_low"], int(row["candle_index"])
                )
                if was_valid:
                    self._entry_price = row["next_open"] if side is Side.UP else 1.0 - row["next_open"]
        else:
            raise ValueError(f"unknown action {action!r}")

        fee = self.portfolio.fees_paid - fee_before

        # forced settlement at episode end -- no exit taker fee on resolution
        if is_last_step and self.portfolio.side is not Side.FLAT:
            self.portfolio.settle(str(row["winner"]).strip().capitalize())
            self._entry_price = 0.0

        if is_last_step:
            value_after = self.portfolio.cash  # settled: no open position left
        else:
            next_row = self._ep.iloc[self._i + 1]
            value_after = self.portfolio.value(next_row["price_up"])

        reward = value_after - value_before

        info = StepInfo(
            ACTION_NAMES[action],
            was_valid,
            _SIDE_TO_BUCKET[self.portfolio.side],
            reward,
            fee=fee,
            gross_pnl=max(0.0, reward + fee),
        )
        done = is_last_step

        if not done:
            self._i += 1

        next_state = self._get_state()
        return next_state, reward, done, info
