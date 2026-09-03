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

import math

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

    def __init__(self, episodes, config: ExecutionConfig | None = None, seed=None):
        if not episodes:
            raise ValueError(
                "TradingEnv got no episodes. This used to just print 'no episodes' "
                "and keep going, and then reset() blew up later on an empty-range "
                "integers() call that had nothing to do with the real problem."
            )

        self.episodes = episodes
        self.config = config or ExecutionConfig()
        self._ep = None # the current market df being played
        self._i = 0 # which candle row are we in rn
        self.portfolio = Portfolio(config=self.config)
        self._entry_price = 0.0 # Up-equivalent price the open position was entered at
        self._c = None # numpy views of the current episode, see _columns()
        self._cols = {} # index -> those views, built once per episode
        # Seeded on purpose. Which markets a run visits is part of the run, and
        # leaving default_rng() unseeded meant nobody could reproduce the 30-run
        # spread we were reporting.
        self._rng = np.random.default_rng(seed)

    def _columns(self, index):
        """Grab the columns the step loop needs as numpy arrays, once per episode.

        The loop used to call self._ep.iloc[i]["price_up"] several times every
        single step. A full 30-seed sweep is around 25 million steps, so that
        pandas indexing was basically the entire runtime -- 2 hours, almost none
        of it actual math. The episode frames never change, so we pull the
        columns out once and cache them.
        """
        cached = self._cols.get(index)
        if cached is None:
            ep = self.episodes[index]
            get = lambda c: (
                ep[c].to_numpy(dtype=float) if c in ep.columns
                else np.full(len(ep), np.nan)
            )
            cached = {
                "candle_index": ep["candle_index"].to_numpy(dtype=np.int64),
                "price_up": get("price_up"),
                "price_down": get("price_down"),
                "next_open": get("next_open"),
                "next_high": get("next_high"),
                "next_low": get("next_low"),
                "winner": str(ep["winner"].iloc[0]).strip().capitalize(),
                "n": len(ep),
            }
            self._cols[index] = cached
        return cached

    def _get_state(self):
        c = self._c
        i = self._i
        position = _SIDE_TO_BUCKET[self.portfolio.side]

        # time bucket safely bounded [0, 59]
        time_bucket = max(0, min(int(c["candle_index"][i]), MAX_TIME_BUCKET))

        # current mark price of whichever leg (if any) we hold
        if position == LONG_DOWN:
            curr_price = c["price_down"][i]
        else:
            curr_price = c["price_up"][i]

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

        self._ep = self.episodes[index]
        self._c = self._columns(index)
        self._i = 0
        self.portfolio = Portfolio(config=self.config)
        self._entry_price = 0.0
        return self._get_state()

    def step(self, action):
        c = self._c
        i = self._i
        is_last_step = i == c["n"] - 1
        nxt_o = c["next_open"][i]
        fill_candle = int(c["candle_index"][i]) + 1

        value_before = self.portfolio.value(c["price_up"][i])
        was_valid = True
        fee_before = self.portfolio.fees_paid

        if action == HOLD:
            pass
        elif action == CLOSE:
            if self.portfolio.side is Side.FLAT:
                was_valid = False
            elif is_last_step or math.isnan(nxt_o):
                was_valid = False  # no next candle to fill a close against
            else:
                # Stamped with the candle the trade FILLS on, not the one that
                # triggered it. Same as backtest.py and sim.evaluation, so
                # holding periods mean the same thing everywhere.
                was_valid = self.portfolio.close(
                    nxt_o, c["next_high"][i], c["next_low"][i], fill_candle,
                )
                if was_valid:
                    self._entry_price = 0.0
        elif action in (BUY_UP, BUY_DOWN):
            if self.portfolio.side is not Side.FLAT:
                was_valid = False
            elif is_last_step or math.isnan(nxt_o):
                was_valid = False  # no next candle to fill an entry against
            else:
                side = Side.UP if action == BUY_UP else Side.DOWN
                was_valid = self.portfolio.buy(
                    side, nxt_o, c["next_high"][i], c["next_low"][i], fill_candle,
                )
                if was_valid:
                    # The price we actually paid, slippage and all -- not the
                    # raw next_open. The agent's pnl_bucket is measured against
                    # this, so using the un-slipped mid would tell it it was
                    # already up on a position it just paid the spread to open.
                    entry = self.portfolio.trades[-1]
                    self._entry_price = entry.price
        else:
            raise ValueError(f"unknown action {action!r}")

        fee = self.portfolio.fees_paid - fee_before

        # forced settlement at episode end -- no exit taker fee on resolution
        if is_last_step and self.portfolio.side is not Side.FLAT:
            self.portfolio.settle(c["winner"])
            self._entry_price = 0.0

        if is_last_step:
            value_after = self.portfolio.cash  # settled: no open position left
        else:
            value_after = self.portfolio.value(c["price_up"][i + 1])

        reward = value_after - value_before

        info = StepInfo(
            ACTION_NAMES[action],
            was_valid,
            _SIDE_TO_BUCKET[self.portfolio.side],
            reward,
            fee=fee,
            # No clamp here. This used to be max(0.0, reward + fee), which
            # threw away every losing step from the gross-PnL total but kept
            # their fees. The fee-drag ratio built on that read 6.9% when the
            # honest number was 147.3% -- 21.5x off on the exact same trades,
            # and flattering right when the policy was doing worst. Fee drag is
            # computed once now, in sim.metrics, off net gross PnL.
            gross_pnl=reward + fee,
        )
        done = is_last_step

        if not done:
            self._i += 1

        next_state = self._get_state()
        return next_state, reward, done, info
