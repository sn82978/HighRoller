"""
Turns a train/val/test split into a list of per-market episode DataFrames for TradingEnv.

This used to build off a random split done locally in split_data.py, but that
split didn't share any markets with BaselineModels', so you couldn't compare
results across models at all. Now it just pulls from
sim.evaluation.load_split_candles like everything else.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pandas as pd

from sim.evaluation import load_split_candles


def prepare_episodes(df: pd.DataFrame) -> list[pd.DataFrame]:
    """Split a multi-market candle frame into one DataFrame per market."""
    episodes = []
    df = df.sort_values(["event_slug", "candle_index"]).copy()

    for slug, ep_df in df.groupby("event_slug", sort=False):
        ep_df = ep_df.dropna(subset=["close"]).copy()
        if ep_df.empty:
            continue

        ep_df["price_up"] = ep_df["close"]
        ep_df["price_down"] = 1.0 - ep_df["price_up"]
        ep_df = ep_df.reset_index(drop=True)
        episodes.append(ep_df)

    return episodes


def get_training_data() -> list[pd.DataFrame]:
    df = load_split_candles("train")
    episodes = prepare_episodes(df)
    print(f"{len(episodes)} training episodes")
    return episodes


def get_eval_data() -> list[pd.DataFrame]:
    df = load_split_candles("val")
    episodes = prepare_episodes(df)
    print(f"{len(episodes)} eval episodes")
    return episodes


def get_test_data(*, allow_test: bool = False) -> list[pd.DataFrame]:
    """Held-out episodes. You have to pass allow_test explicitly.

    This had allow_test=True hardcoded, so anything that called it read the
    held-out data whether or not that was what you meant. Right now the only
    caller is a commented-out block in training.py, which is exactly the sort of
    line someone uncomments without reading it first.
    """
    df = load_split_candles("test", allow_test=allow_test)
    episodes = prepare_episodes(df)
    print(f"{len(episodes)} test episodes")
    return episodes
