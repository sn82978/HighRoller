'''
take the df of all market rows, clean the timestamps, and filter out neg pre-market candles

make a list of df's w/ one 15-minute market
'''

import pandas as pd
import os
import numpy as np

DATA_DIR = "/Users/shreyanakum/Documents/HighRoller/QLearning/data/train"
TEST_DIR = "/Users/shreyanakum/Documents/HighRoller/QLearning/data/test"
EVAL_DIR = "/Users/shreyanakum/Documents/HighRoller/QLearning/data/val"

def prepare_episodes(df: pd.DataFrame):
    episodes = []

    df = df.sort_values(["event_slug", "t"]).copy()

    for i, ep_df in df.groupby("event_slug"):
        ep_df = ep_df[ep_df["candle_index"] >= 0].copy() # before the market is when the candle_index is less than 0
        ep_df = ep_df.dropna(subset=["close"]).copy()

        # print(ep_df.head())

        if ep_df.empty:
            print("empty")
            continue

        ep_df["price_up"] = ep_df["close"] # ir we buy_up, entry price is price_up 
        ep_df["price_down"] = 1.0 - ep_df["price_up"] # if we buy_down, entry price is price_down

        ep_df = ep_df.reset_index(drop=True) # iloc indexing in env matches exactly
        episodes.append(ep_df)

    return episodes

def get_training_data():
    all_episodes = []
    for file in os.listdir(DATA_DIR):
        # if not file.endswith('.csv'):
        #     continue
        # df = pd.read_csv(f"{DATA_DIR}/btc_updown_15m_candles_15s_2025-11-28.csv")
        # df = pd.read_csv(f"{DATA_DIR}/{file}")
        df = pd.read_parquet(f"{DATA_DIR}/{file}")
        episodes = prepare_episodes(df)
        all_episodes += episodes
        # print(f"{len(episodes)} episodes")

    print(f"{len(all_episodes)} training episodes")
    return all_episodes

def get_eval_data():
    all_episodes = []
    for file in os.listdir(EVAL_DIR):
        # if not file.endswith('.csv'):
        #     continue
        # df = pd.read_csv(f"{DATA_DIR}/btc_updown_15m_candles_15s_2025-11-28.csv")
        df = pd.read_parquet(f"{EVAL_DIR}/{file}")
        episodes = prepare_episodes(df)
        all_episodes += episodes
        # print(f"{len(episodes)} episodes")

    print(f"{len(all_episodes)} eval episodes")
    return all_episodes

def get_test_data():
    all_episodes = []
    for file in os.listdir(TEST_DIR):
        # if not file.endswith('.csv'):
        #     continue
        # df = pd.read_csv(f"{DATA_DIR}/btc_updown_15m_candles_15s_2025-11-28.csv")
        df = pd.read_parquet(f"{TEST_DIR}/{file}")
        episodes = prepare_episodes(df)
        all_episodes += episodes
        # print(f"{len(episodes)} episodes")

    print(f"{len(all_episodes)} test episodes")
    return all_episodes