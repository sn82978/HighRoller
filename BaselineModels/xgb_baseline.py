"""XGBoost baseline: predict the outcome, then trade where we disagree.

Two steps, kept separate on purpose.

1. Predict P(Up) from the features at candle c. The thing to beat isn't a coin
   flip, it's the market price, which gets log loss 0.442 / AUC 0.870 on train
   for free. Note that p_mkt is one of the features, so the model can just copy
   the market and score ~0.442 without learning anything. compare_to_market()
   is there to catch that.

2. Buy when |p_hat - p_mkt| > theta, then hold to resolution (selling early
   pays the taker fee twice). Pick theta on validation PnL after fees, not on
   accuracy.

Keeping the policy fixed and dumb is what makes this a baseline for the RL
agent. If the agent does better, it's because of the timing, not the forecast.

    python -m BaselineModels.xgb_baseline
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import xgboost as xgb

from BaselineModels.backtest import backtest, no_trade_policy, threshold_policy
from BaselineModels.data_loader import load_split
from BaselineModels.features import build_features, feature_columns
from BaselineModels.metrics import (
    auc_by_horizon,
    calibration_table,
    comparison_table,
    paired_bootstrap_logloss,
    probability_metrics,
    trading_metrics,
)
from sim.execution import ExecutionConfig

# Kept conservative on purpose. 354k training rows sounds like a lot, but all
# 60 candles in a market share one label, so there are really only ~5.9k
# independent examples. Shallow trees + big min_child_weight so we don't just
# memorise individual markets.
DEFAULT_PARAMS: dict[str, object] = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "max_depth": 4,
    "eta": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 200,
    "reg_lambda": 5.0,
    "tree_method": "hist",
    "seed": 0,
}

DEFAULT_ROUNDS = 2000
EARLY_STOPPING = 50

# Thresholds to try, in probability units. A round trip at the money costs
# about 3.5c, so we don't expect anything under ~0.035 to clear costs. Sweep it
# anyway so the curve shows it.
DEFAULT_THETAS = (0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.10, 0.15, 0.20)


@dataclass
class FittedModel:
    booster: xgb.Booster
    features: list[str]
    best_iteration: int

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Probability of Up for each row of ``df``."""
        missing = set(self.features) - set(df.columns)
        if missing:
            raise ValueError(f"frame is missing model features: {sorted(missing)}")
        dm = xgb.DMatrix(df[self.features], feature_names=self.features)
        return self.booster.predict(dm, iteration_range=(0, self.best_iteration + 1))


# -- fitting -------------------------------------------------------------
# Drop these to fit without the market's own opinion. Answers the obvious
# question from the gain table: do the other features know anything on their
# own, or is the model just repeating p_mkt back to us?
MARKET_PRICE_FEATURES = ("p_mkt", "logit_p")


def fit(
    train: pd.DataFrame,
    val: pd.DataFrame,
    params: dict | None = None,
    num_boost_round: int = DEFAULT_ROUNDS,
    early_stopping_rounds: int = EARLY_STOPPING,
    verbose: bool = False,
    drop_features: tuple[str, ...] = (),
) -> FittedModel:
    """Fit on train, early-stop on validation log loss.

    Early stopping is the only thing validation is used for during fitting.
    Theta gets tuned later, separately, on validation PnL. Test isn't touched.
    """
    feats = [c for c in feature_columns(train) if c not in drop_features]
    if not feats:
        raise ValueError("no features left after drop_features")
    missing = set(feats) - set(val.columns)
    if missing:
        raise ValueError(f"val frame lacks train features: {sorted(missing)}")

    dtrain = xgb.DMatrix(train[feats], label=train.y, feature_names=feats)
    dval = xgb.DMatrix(val[feats], label=val.y, feature_names=feats)
    booster = xgb.train(
        {**DEFAULT_PARAMS, **(params or {})},
        dtrain,
        num_boost_round=num_boost_round,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=100 if verbose else False,
    )
    return FittedModel(booster=booster, features=feats, best_iteration=booster.best_iteration)


# -- forecast evaluation -------------------------------------------------
def compare_to_market(df: pd.DataFrame, p_hat: np.ndarray) -> pd.DataFrame:
    """Score the model and the market side by side on the same rows."""
    rows = {
        "market": probability_metrics(df.y.to_numpy(), df.p_mkt.to_numpy()),
        "model": probability_metrics(df.y.to_numpy(), p_hat),
        "always_0.5": probability_metrics(df.y.to_numpy(), np.full(len(df), 0.5)),
    }
    return pd.DataFrame(rows).T.rename_axis("forecaster").reset_index()


def forecast_report(df: pd.DataFrame, p_hat: np.ndarray, seed: int = 0) -> dict:
    """All the forecast-quality tables, plus a check on whether the edge is real.

    The bootstrap resamples markets, not rows. Candles in the same market share
    a label, so a row-level interval comes out about sqrt(60) too narrow and
    makes noise look significant.
    """
    return {
        "comparison": compare_to_market(df, p_hat),
        "calibration": calibration_table(df.y.to_numpy(), p_hat),
        "auc_by_horizon": auc_by_horizon(
            df.y.to_numpy(), p_hat, df.candles_remaining.to_numpy()
        ),
        "significance": paired_bootstrap_logloss(
            df.y.to_numpy(),
            p_hat,
            df.p_mkt.to_numpy(),
            df.event_slug.to_numpy(),
            seed=seed,
        ),
    }


# -- policy evaluation ---------------------------------------------------
def sweep_theta(
    df: pd.DataFrame,
    p_hat: np.ndarray,
    thetas=DEFAULT_THETAS,
    config: ExecutionConfig | None = None,
    **policy_kwargs,
) -> pd.DataFrame:
    """Backtest the threshold rule at each theta. Run this on validation.

    Scored on pnl_per_1k_deployed, i.e. PnL over the capital actually committed,
    so a theta that trades 40 markets is comparable to one that trades 4,000.
    """
    rows = []
    for theta in thetas:
        results = backtest(df, threshold_policy(theta, **policy_kwargs), config, p_hat)
        rows.append({"theta": theta, **trading_metrics(results)})
    return pd.DataFrame(rows)


# How many trades a theta needs before we'll let it win the sweep. This isn't
# hypothetical: without a floor the first run picked theta=0.10, where exactly
# 1 market out of 1,343 cleared the threshold and that one lucky trade scored
# +$1,005 per $1,000 deployed. 30 is the usual "you can take a mean of this"
# cutoff.
MIN_TRADES_FOR_SELECTION = 30


def best_theta(
    sweep: pd.DataFrame,
    criterion: str = "pnl_per_1k_deployed",
    min_trades: int = MIN_TRADES_FOR_SELECTION,
) -> float:
    """Best theta by `criterion`, among the ones that traded enough to count.

    Raises if nothing clears the floor rather than falling back to the best of
    a few one-trade flukes. "No tradeable sample" is a fine thing to report.
    """
    live = sweep[sweep.n_traded >= min_trades]
    if live.empty:
        raise ValueError(
            f"no theta produced at least {min_trades} trades "
            f"(best was {int(sweep.n_traded.max())}); the rule has no tradeable "
            "sample at these thresholds, which is itself the result"
        )
    return float(live.loc[live[criterion].idxmax(), "theta"])


# -- pipeline ------------------------------------------------------------
def load_features(split: str, allow_test: bool = False) -> pd.DataFrame:
    """Load a split and build features. Prints the dropped-market warning."""
    candles = load_split(split, allow_test=allow_test)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        feats = build_features(candles)
    for w in caught:
        print(f"  [{split}] {w.message}")
    return feats


def _show(title: str, obj) -> None:
    print(f"\n{title}\n{'-' * len(title)}")
    if isinstance(obj, pd.DataFrame):
        print(obj.to_string(index=False))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            print(f"  {k:<24} {v}")
    else:
        print(obj)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    ap.add_argument("--slippage", type=float, default=0.25, help="fraction of candle range charged adversely")
    ap.add_argument("--min-candles-remaining", type=int, default=0)
    ap.add_argument("--save", type=str, default=None, help="path to save the booster")
    ap.add_argument(
        "--ablate-market",
        action="store_true",
        help="also fit without p_mkt/logit_p, to see what the other features know",
    )
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    config = ExecutionConfig(slippage_frac=args.slippage)

    print("loading train/val features...")
    train = load_features("train")
    val = load_features("val")
    print(f"  train {len(train):,} rows / {train.event_slug.nunique():,} markets")
    print(f"  val   {len(val):,} rows / {val.event_slug.nunique():,} markets")

    print("\nfitting...")
    model = fit(train, val, num_boost_round=args.rounds, verbose=args.verbose)
    print(f"  best iteration {model.best_iteration} of {args.rounds}")

    p_val = model.predict(val)
    report = forecast_report(val, p_val)
    _show("Forecast quality (validation)", report["comparison"])
    _show("Is the model really better than the market? (bootstrap over markets)", report["significance"])
    _show("Calibration (validation)", report["calibration"])
    _show("AUC by time remaining (validation)", report["auc_by_horizon"])

    imp = model.booster.get_score(importance_type="gain")
    _show(
        "Top features by gain",
        pd.DataFrame(sorted(imp.items(), key=lambda kv: -kv[1])[:12], columns=["feature", "gain"]),
    )

    if args.ablate_market:
        print("\nablation: refitting without the market price...")
        blind = fit(train, val, num_boost_round=args.rounds, drop_features=MARKET_PRICE_FEATURES)
        _show(
            "Forecast quality without p_mkt (validation)",
            compare_to_market(val, blind.predict(val)),
        )

    print("\nsweeping theta on validation PnL after fees...")
    sweep = sweep_theta(
        val, p_val, config=config, min_candles_remaining=args.min_candles_remaining
    )
    cols = [
        "theta", "n_traded", "trade_rate", "pnl_per_1k_deployed",
        "gross_pnl_per_1k_deployed", "fee_per_1k_deployed",
        "sharpe", "win_rate", "max_drawdown", "avg_holding_candles",
    ]
    _show("Threshold sweep (validation)", sweep[cols])

    try:
        theta = best_theta(sweep)
        print(f"\nselected theta = {theta} (>= {MIN_TRADES_FOR_SELECTION} trades required)")
    except ValueError as exc:
        print(f"\nno theta selected: {exc}")
        return

    no_trade = trading_metrics(backtest(val, no_trade_policy, config))
    chosen = trading_metrics(
        backtest(val, threshold_policy(theta, min_candles_remaining=args.min_candles_remaining), config, p_val)
    )
    _show(
        "Baseline comparison (validation)",
        comparison_table({"no_trade": no_trade, f"xgboost_theta_{theta}": chosen}),
    )

    if args.save:
        model.booster.save_model(args.save)
        print(f"\nsaved booster to {args.save}")


if __name__ == "__main__":
    main()
