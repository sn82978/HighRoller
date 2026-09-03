"""
Score Q-tables we already trained on a given split, without training again.

training.py trains 30 agents and scores each on train and val. Scoring those
same agents on the held-out test split has to be its own step, for two reasons:

  - If we retrained to get to test we'd have 30 different agents, and then the
    val number and the test number wouldn't describe the same thing. It'd be two
    unrelated runs, not a generalisation gap.
  - The test split is only supposed to be scored once. If that were bundled into
    the training entry point, every future `python QLearning/training.py` would
    burn that one shot again.

So this loads QLearning/models/<family>_seed<NN>.npy, plays each agent greedily
over whichever split you ask for, and appends rows in the same shared schema
every other track uses.

    python QLearning/evaluate_split.py --split test --allow-test

Refuses --split test without --allow-test, same as everything else here.
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from data_preparation import prepare_episodes
from make_environment import TradingEnv
from q_learning_agent import QLearningAgent

from sim.evaluation import load_split_candles
from sim.execution import ExecutionConfig
from training import (
    METRICS_DIR,
    MODELS_DIR,
    MODEL_FAMILY,
    OUT_DIR,
    evaluate,
    metrics_path,
)


def saved_models(family: str) -> list[tuple[str, int, str]]:
    """(model_name, seed, path) for every saved Q-table in this family."""
    if not os.path.isdir(MODELS_DIR):
        raise SystemExit(f"no models directory at {MODELS_DIR} -- run training.py first")
    found = []
    for name in sorted(os.listdir(MODELS_DIR)):
        stem, ext = os.path.splitext(name)
        if ext != ".npy" or not stem.startswith(f"{family}_seed"):
            continue
        found.append((stem, int(stem.split("_seed")[1]), os.path.join(MODELS_DIR, name)))
    if not found:
        raise SystemExit(
            f"no {family}_seed*.npy under {MODELS_DIR} -- run "
            f"`python QLearning/training.py` first"
        )
    return found


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--family", default=MODEL_FAMILY)
    ap.add_argument("--slippage", type=float, default=0.25,
                    help="ExecutionConfig.slippage_frac; project default is 0.25")
    args = ap.parse_args(argv)

    if args.split == "test" and not args.allow_test:
        raise SystemExit(
            "refusing --split test without --allow-test. The test split is "
            "budgeted for exactly one scored pass, at the very end."
        )

    models = saved_models(args.family)
    out_csv = metrics_path(args.family, args.split)
    markets_csv = os.path.join(OUT_DIR, "markets.csv")

    # Appending onto a split we already scored would double every market in the
    # table. Better to refuse than to quietly write a 60-row "30-run" sweep,
    # which is exactly what two concurrent sweeps did before training.py started
    # taking a lock.
    if os.path.exists(out_csv):
        raise SystemExit(
            f"{out_csv} already exists. That split has been scored. Delete the "
            f"file deliberately if you really mean to score it again -- and if "
            f"the split is test, say so in the writeup, because it is no longer "
            f"one pass."
        )

    print(f"loading {args.split} episodes")
    episodes = prepare_episodes(
        load_split_candles(args.split, allow_test=args.allow_test)
    )
    config = ExecutionConfig(slippage_frac=args.slippage)

    for model_name, seed, path in models:
        print(f"\n=== {model_name} on {args.split} ===")
        agent = QLearningAgent(
            state_shape=TradingEnv.state_space_size(),
            n_actions=TradingEnv.n_actions(),
            seed=seed,
        )
        agent.load(path)
        env = TradingEnv(episodes=episodes, config=config, seed=seed)
        evaluate(env, agent, True, args.split, model_name, seed, seed)

    print(f"\nwrote {out_csv}")
    print(f"appended {len(models)} runs to {markets_csv}")
    print(f"\nnext: python sim/compare_models.py --split {args.split}"
          + (" --allow-test" if args.split == "test" else ""))


if __name__ == "__main__":
    main()
