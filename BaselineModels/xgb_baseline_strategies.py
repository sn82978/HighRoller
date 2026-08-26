"""
XGBoost directional baseline on the no-lookahead feature set.

    python BaselineModels/xgb_baseline.py               # train on train, report on val
    python BaselineModels/xgb_baseline.py --tune         # sweep entry thresholds on val
    python BaselineModels/xgb_baseline.py --split test --allow-test   # the one look

Trains an XGBClassifier on FEATURE_COLUMNS to predict y (market resolves Up),
early stopping on val. Predicted P(Up) then drives the same threshold entry/flip
rule as generate_trades.py's momentum_flip, just using the model's probability
instead of price as the trigger -- so a win here is actually about the model's
signal and not a different backtest setup. Same execution engine as everything
else (sim.execution via sim.evaluation.simulate_market), writes markets.csv in
the same schema so it drops into compare_models.py.
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from BaselineModels.data_loader import load_split
from BaselineModels.features import build_features, feature_columns
from sim.evaluation import results_to_frame, score, simulate_market
from sim.execution import BUY_DOWN, BUY_UP, ExecutionConfig, HOLD, Side

OUT_DIR = os.path.join(ROOT, "BaselineModels/output")


def build_split_features(split: str, *, allow_test: bool = False) -> pd.DataFrame:
    candles = load_split(split, dataset="candles_15s", allow_test=allow_test)
    return build_features(candles, live_only=True, drop_warmup=True)


def train_model(train_feat: pd.DataFrame, val_feat: pd.DataFrame, **params) -> XGBClassifier:
    cols = feature_columns(train_feat)
    default = dict(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=20,
        eval_metric="logloss",
        early_stopping_rounds=30,
        n_jobs=-1,
    )
    default.update(params)
    model = XGBClassifier(**default)
    model.fit(
        train_feat[cols],
        train_feat.y,
        eval_set=[(val_feat[cols], val_feat.y)],
        verbose=False,
    )
    return model


def make_model_policy(threshold: float, allow_flip: bool = True):
    # same as generate_trades.make_momentum_flip but keyed off p_up instead of price
    def decide(row, portfolio, i):
        p_up = row.p_up
        if portfolio.side is Side.FLAT:
            if p_up >= threshold:
                return BUY_UP
            if (1.0 - p_up) >= threshold:
                return BUY_DOWN
            return HOLD
        if not allow_flip:
            return HOLD
        if portfolio.side is Side.UP and (1.0 - p_up) >= threshold:
            return BUY_DOWN
        if portfolio.side is Side.DOWN and p_up >= threshold:
            return BUY_UP
        return HOLD

    return decide


def run_backtest(feat: pd.DataFrame, model: XGBClassifier, threshold: float, split: str, config: ExecutionConfig):
    cols = feature_columns(feat)
    feat = feat.copy()
    feat["p_up"] = model.predict_proba(feat[cols])[:, 1]

    decide = make_model_policy(threshold)
    results, fills = [], []
    for slug, ep in feat.groupby("event_slug", sort=False):
        ep = ep.sort_values("candle_index").reset_index(drop=True)
        if ep.p_mkt.isna().any() or pd.isna(ep.winner.iloc[0]):
            continue
        r = simulate_market(ep, decide, strategy="xgb_baseline", split=split, config=config)
        results.append(r)
        for t in r.portfolio.trades:
            fills.append(
                dict(
                    strategy="xgb_baseline",
                    event_slug=slug,
                    candle_index=t.candle_index,
                    action=t.action,
                    side=t.side.name,
                    shares=t.shares,
                    price=t.price,
                    fee=t.fee,
                    cash_delta=t.cash_delta,
                )
            )
    return results_to_frame(results), pd.DataFrame(fills)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["train", "val", "test"], help="split to report the backtest on")
    ap.add_argument("--allow-test", action="store_true", help="required to evaluate on --split test")
    ap.add_argument("--threshold", type=float, default=0.55, help="entry/flip confidence, matches momentum_flip's default")
    ap.add_argument("--stake", type=float, default=100.0)
    ap.add_argument("--slippage", type=float, default=0.0, help="ExecutionConfig.slippage_frac")
    ap.add_argument("--tune", action="store_true", help="sweep thresholds on val instead of a single backtest")
    ap.add_argument("--out-dir", default=OUT_DIR)
    args = ap.parse_args()

    if args.split == "test" and not args.allow_test:
        raise SystemExit("refusing --split test without --allow-test -- tune on val first")

    print("loading + featurizing train/val")
    train_feat = build_split_features("train")
    val_feat = build_split_features("val")

    print(f"training on {train_feat.event_slug.nunique():,} train markets, "
          f"{len(feature_columns(train_feat))} features")
    model = train_model(train_feat, val_feat)
    print(f"best iteration: {model.best_iteration}")

    config = ExecutionConfig(slippage_frac=args.slippage, stake_dollars=args.stake)

    if args.tune:
        print("\nsweeping entry threshold on val:")
        for t in (0.52, 0.55, 0.58, 0.60, 0.65, 0.70):
            mk, _ = run_backtest(val_feat, model, t, "val", config)
            s = score(mk)
            print(
                f"  threshold {t:.2f}  n_traded={s['n_traded']:>4}  "
                f"total_pnl=${s['total_pnl']:>9,.2f}  avg_return={s['avg_return'] * 100:>7.3f}%  "
                f"t_stat={s['t_stat']:>6.2f}"
            )
        return

    eval_feat = val_feat if args.split == "val" else (
        train_feat if args.split == "train" else build_split_features("test", allow_test=True)
    )
    mk, fills = run_backtest(eval_feat, model, args.threshold, args.split, config)

    os.makedirs(args.out_dir, exist_ok=True)
    mk_path = os.path.join(args.out_dir, "markets.csv")
    mk.to_csv(mk_path, index=False)
    fills.to_csv(os.path.join(args.out_dir, "fills.csv"), index=False)
    print(f"{len(mk):>7,} rows -> {mk_path}")

    s = score(mk)
    for k, v in s.items():
        print(f"  {k:<24} {v}")


if __name__ == "__main__":
    main()
