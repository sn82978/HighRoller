"""Regenerate every baseline number in RESULTS.md, in one command.

    python BaselineModels/run_baselines.py                  # val, writes RESULTS.md
    python BaselineModels/run_baselines.py --split test --allow-test   # once, at the end

Runs all three non-RL policies -- the no-trade floor, buy-and-hold, and XGBoost
paired with its swept threshold -- over the same markets, through the same cost
model, scored by the same function. Writes:

    RESULTS.md                          the report, regenerated from this run
    BaselineModels/output/markets.csv   per-market rows in the shared schema,
                                        so sim/compare_models.py picks them up
    figs/xgb_calibration_<split>.png    the calibration curve RESULTS.md cites

Why this exists. RESULTS.md quoted a buy-and-hold column and a `total_fills`
row, and `xgb_baseline.main()` produced neither -- it never ran buy-and-hold at
all, and no committed code emitted that metric. So the document could not be
regenerated from the repo, and nothing checked whether its numbers still matched
the code underneath them. The baselines track also wrote no markets.csv, so
compare_models.py skipped it entirely and the four-way comparison had a hole in
it exactly where the tuned baseline should have been.
"""

from __future__ import annotations

import argparse
import io
import os
import subprocess
import sys
import warnings

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd

from BaselineModels.backtest import backtest, buy_and_hold_policy, no_trade_policy, threshold_policy
from BaselineModels.features import FEATURE_COLUMNS, feature_columns
from BaselineModels.data_loader import LEAKY_COLUMNS, compute_bounds, market_universe
from BaselineModels.metrics import (
    auc_by_horizon,
    calibration_table,
    comparison_table,
    paired_bootstrap_logloss,
    probability_metrics,
    results_frame,
    trading_metrics,
)
from BaselineModels.xgb_baseline import (
    DEFAULT_ROUNDS,
    DEFAULT_THETAS,
    EARLY_STOPPING,
    MIN_TRADES_FOR_SELECTION,
    best_theta,
    fit,
    load_features,
    sweep_theta,
)
from sim.execution import ExecutionConfig
from sim.metrics import write_markets

OUT_DIR = os.path.join(ROOT, "BaselineModels/output")
FIGS_DIR = os.path.join(ROOT, "figs")
REPORT = os.path.join(ROOT, "RESULTS.md")

#: The one split a hyperparameter may be selected on.
SELECTION_SPLIT = "val"


def selects_on_eval_split(split: str) -> bool:
    """May theta be read off the sweep of the split we are reporting?

    Only when that split is validation. Everywhere else -- train, and above all
    test -- the threshold has to come from a separate val sweep, or the
    "held-out" number is the maximum over eleven thresholds tried on the
    held-out data. Broken out of main() so this stays a testable claim rather
    than one branch inside a 200-line driver.
    """
    return split == SELECTION_SPLIT


def git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
            capture_output=True, text=True, timeout=10,
        ).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _fmt(df: pd.DataFrame) -> str:
    return df.to_string(index=False)


def load_with_warnings(split: str, allow_test: bool) -> tuple[pd.DataFrame, list[str]]:
    """Load a split, capturing build_features' sample-cut warnings verbatim.

    Those warnings are the only record that markets were dropped for an
    insufficient warmup window, and a silent cut reads as full coverage.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        feats = load_features(split, allow_test=allow_test)
    return feats, [f"[{split}] {w.message}" for w in caught]


def fee_denominator_check(named: dict[str, list]) -> pd.DataFrame:
    """Our fee ratio next to the positive-only one, on identical trades.

    The report's Section 3.3 turns on this gap, so it is computed rather than
    quoted. `positive_only_gross` sums the gross PnL of profitable markets only,
    dropping every loss from the denominator while its fees stay in the
    numerator -- which flatters a policy exactly when it is doing worst.
    """
    rows = {}
    for name, results in named.items():
        gross = np.array([r.gross_pnl for r in results], dtype=float)
        fees = float(sum(r.fees for r in results))
        net_gross = float(gross.sum())
        pos_only = float(gross[gross > 0].sum())
        rows[name] = {
            "total_fills": int(sum(r.n_fills for r in results)),
            "net_gross": net_gross,
            "positive_only_gross": pos_only,
            "fee_frac_net_gross (as coded)": fees / net_gross if net_gross > 0 else float("nan"),
            "fee_frac_positive_only (RL style)": fees / pos_only if pos_only > 0 else float("nan"),
        }
    return pd.DataFrame(rows)


def calibration_figure(y, p_hat, split: str) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tbl = calibration_table(y, p_hat)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=1, label="perfect calibration")
    ok = tbl.n > 0
    ax.plot(tbl.mean_predicted[ok], tbl.observed_rate[ok], "o-", color="crimson",
            linewidth=1.6, label="model")
    ax.set_xlabel("mean predicted P(Up)")
    ax.set_ylabel("observed rate")
    ax.set_title(f"Calibration on the {split} split")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(FIGS_DIR, exist_ok=True)
    path = os.path.join(FIGS_DIR, f"xgb_calibration_{split}.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", default="val", choices=["train", "val", "test"],
                    help="split to report the backtest on")
    ap.add_argument("--allow-test", action="store_true",
                    help="required for --split test; the paper budgets one such run")
    ap.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    ap.add_argument("--slippage", type=float, default=0.25)
    ap.add_argument("--tie", default="down", choices=["down", "up", "skip"],
                    help="buy-and-hold's tie-break at exactly 0.500")
    ap.add_argument("--report", default=REPORT, help="markdown output path")
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--no-report", action="store_true", help="skip writing the markdown")
    args = ap.parse_args(argv)

    if args.split == "test" and not args.allow_test:
        raise SystemExit(
            "refusing --split test without --allow-test. Tune on val; the test "
            "split is budgeted for exactly one run, at the very end."
        )

    config = ExecutionConfig(slippage_frac=args.slippage)
    out = io.StringIO()
    w = lambda s="": print(s, file=out)

    # -- data ------------------------------------------------------------
    print("loading train/val features...")
    train, train_warns = load_with_warnings("train", False)
    evalset, eval_warns = load_with_warnings(args.split, args.allow_test)
    if args.split == "train":
        evalset, eval_warns = train, []
    universe = market_universe()
    bounds = compute_bounds(universe)

    # -- fit -------------------------------------------------------------
    print("fitting...")
    fit_val = evalset if args.split == "val" else load_with_warnings("val", False)[0]
    model = fit(train, fit_val, num_boost_round=args.rounds)
    p_hat = model.predict(evalset)
    y = evalset.y.to_numpy()

    # -- policies --------------------------------------------------------
    print("sweeping theta...")
    # Theta is a hyperparameter, so it is chosen on validation and then frozen.
    # This used to select it from the evaluation split's own sweep, which is
    # fine while that split IS val and silently ruinous on --split test: the
    # threshold would be picked by looking at the held-out data, and the number
    # reported as held-out would be the best of eleven thresholds tried on it.
    # sweep_theta's own docstring says "run this on validation"; now it does.
    sweep = sweep_theta(evalset, p_hat, config=config)
    if selects_on_eval_split(args.split):
        selection_sweep = sweep
    else:
        print("  selecting theta on val (never on the reported split)...")
        selection_sweep = sweep_theta(fit_val, model.predict(fit_val), config=config)
    try:
        theta = best_theta(selection_sweep)
        theta_note = (
            f"{theta} (>= {MIN_TRADES_FOR_SELECTION} trades required; "
            f"selected on val)"
        )
    except ValueError as exc:
        theta, theta_note = None, f"none selected: {exc}"

    print("backtesting policies...")
    policies = {
        "no_trade": (no_trade_policy, None),
        "buy_and_hold": (buy_and_hold_policy(tie=args.tie), None),
    }
    if theta is not None:
        policies[f"xgboost_theta_{theta}"] = (threshold_policy(theta), p_hat)

    results = {
        name: backtest(evalset, pol, config, preds) for name, (pol, preds) in policies.items()
    }
    metrics = {name: trading_metrics(r) for name, r in results.items()}

    # -- markets.csv, so compare_models.py can see this track ------------
    start_ts = dict(zip(universe.event_slug, universe.start_ts))
    frames = [
        results_frame(r, strategy=name, split=args.split, start_ts=start_ts)
        for name, r in results.items()
    ]
    os.makedirs(args.out_dir, exist_ok=True)
    mk_path = os.path.join(args.out_dir, "markets.csv")
    kept = write_markets(mk_path, pd.concat(frames, ignore_index=True), args.split,
                         slippage_frac=config.slippage_frac)
    print(f"  wrote {mk_path}"
          + (f"  (kept {kept:,} rows from other splits)" if kept else ""))

    fig_path = calibration_figure(y, p_hat, args.split)
    print(f"  wrote {fig_path}")

    if args.no_report:
        return

    # -- report ----------------------------------------------------------
    feats = feature_columns(train)
    leaked = [c for c in feats if c in LEAKY_COLUMNS]
    split_rows = []
    for s in ("train", "val", "test"):
        lo, hi = bounds.range_for(s)
        sel = universe[(universe.start_ts >= lo) & (universe.start_ts < hi)]
        sel = sel[sel.winner.isin(("Up", "Down"))]
        built = {"train": train, args.split: evalset}.get(s)
        split_rows.append({
            "split": s,
            "markets_in_universe": len(sel),
            "markets_after_features": built.event_slug.nunique() if built is not None else "—",
            "feature_rows": len(built) if built is not None else "—",
            "start": pd.to_datetime(lo, unit="s", utc=True).date(),
            "end": pd.to_datetime(hi, unit="s", utc=True).date(),
        })

    w("# Baseline results")
    w()
    w(f"Regenerated by `python BaselineModels/run_baselines.py --split {args.split}` "
      f"at commit `{git_sha()}`.")
    w("Every number below comes from that run. Do not hand-edit this file.")
    w()
    w(f"Cost model: taker fee `0.07*p*(1-p)`, adverse slippage "
      f"`{config.slippage_frac}` of the fill candle's high-low range, "
      f"${config.stake_dollars:g} per entry.")
    w()
    # The two documents disagree on purpose, and a reader who spots it without
    # this note will assume one of them is wrong.
    w("**Scope: this file scores the baselines on every market this track "
      "covers.** `comparison.csv` scores the same policies on the smaller set "
      "*every* track covers, so its totals for `buy_and_hold` and the XGBoost "
      "policy are slightly different numbers for the same policy. Use "
      "`comparison.csv` for anything that puts these models next to the rule "
      "strategies or the RL agent; use this file for anything about the "
      "forecaster on its own.")
    w()

    w("## Setup")
    w()
    w("```")
    w(_fmt(pd.DataFrame(split_rows)))
    w("```")
    w()
    unread = [s for s in ("train", "val", "test") if s not in ("train", args.split)]
    w(f"Splits not read in this run: {', '.join(unread) or 'none'}. "
      "`—` means the split was never loaded, so no feature rows were built.")
    w()
    w(f"### Features consumed ({len(feats)})")
    w()
    w("```")
    for i in range(0, len(feats), 4):
        w("  " + "".join(f"{c:<28}" for c in feats[i:i + 4]).rstrip())
    w("```")
    w()
    w(f"- leak/label/settlement column in the feature list: **{bool(leaked)}**"
      + (f" ({leaked})" if leaked else ""))
    w(f"- `volume` present on the frame but excluded by the allowlist: "
      f"**{'volume' in train.columns and 'volume' not in feats}**")
    w()
    w("### Boosting")
    w()
    w(f"- rounds requested (cap): {args.rounds}")
    w(f"- early-stopping patience: {EARLY_STOPPING}")
    w(f"- best iteration (0-indexed): **{model.best_iteration}** "
      f"(predictions use trees 0..{model.best_iteration})")
    w(f"- early stopping fired: **{model.best_iteration + 1 < args.rounds}**")
    w()

    w(f"## Forecast quality ({args.split})")
    w()
    forecasters = {
        "market": probability_metrics(y, evalset.p_mkt.to_numpy()),
        "model": probability_metrics(y, p_hat),
        "always_0.5": probability_metrics(y, np.full(len(evalset), 0.5)),
    }
    cmp_df = pd.DataFrame(forecasters).T.rename_axis("forecaster").reset_index()
    w("```")
    w(_fmt(cmp_df))
    w("```")
    w()
    beats = {
        m: (forecasters["model"][m] < forecasters["market"][m]) if m != "auc"
        else (forecasters["model"][m] > forecasters["market"][m])
        for m in ("log_loss", "brier", "ece", "auc")
    }
    w("| metric | market | model | model better? |")
    w("|---|---|---|---|")
    for m in ("log_loss", "brier", "ece", "auc"):
        w(f"| {m} | {forecasters['market'][m]:.6f} | {forecasters['model'][m]:.6f} | "
          f"**{'yes' if beats[m] else 'no'}** |")
    w()
    w(f"Model beats the market on **{sum(beats.values())} of 4** metrics. "
      "The bar is the market price, not a coin flip: `p_mkt` is an input "
      "feature, so a model that simply echoes it scores well without learning "
      "anything.")
    w()

    boot = paired_bootstrap_logloss(
        y, p_hat, evalset.p_mkt.to_numpy(), evalset.event_slug.to_numpy()
    )
    w("### Paired bootstrap on the log-loss difference")
    w()
    w("Resampled over **markets**, not rows: the 60 candles in a market share one")
    w("label, so a row-level interval comes out about sqrt(60) too narrow.")
    w()
    w("```")
    for k, v in boot.items():
        w(f"  {k:<20} {v}")
    w("```")
    w()
    verdict = ("does NOT beat" if boot["mean_improvement"] <= 0 else "beats")
    w(f"Improvement (market log loss minus model, so positive = better) is "
      f"**{boot['mean_improvement']:+.6f}**, 95% CI "
      f"**[{boot['ci_lo']:+.6f}, {boot['ci_hi']:+.6f}]**, model ahead in "
      f"**{boot['p_model_better']:.1%}** of resamples. The model **{verdict}** "
      "the market price.")
    w()

    w("### AUC by time remaining")
    w()
    horizon = auc_by_horizon(y, p_hat, evalset.candles_remaining.to_numpy())
    mkt_horizon = auc_by_horizon(
        y, evalset.p_mkt.to_numpy(), evalset.candles_remaining.to_numpy()
    )
    horizon = horizon.rename(columns={"auc": "model_auc"})
    horizon["market_auc"] = mkt_horizon.auc
    w("```")
    w(_fmt(horizon))
    w("```")
    w()
    w("### Calibration (10 bins)")
    w()
    cal = calibration_table(y, p_hat)
    cal["gap"] = cal.observed_rate - cal.mean_predicted
    w("```")
    w(_fmt(cal))
    w("```")
    w()
    # Forward slashes: this string is read on Linux and macOS as often as here,
    # and os.path.relpath hands back backslashes on Windows.
    w(f"Figure: `{os.path.relpath(fig_path, ROOT).replace(os.sep, '/')}` "
      f"(see FIGURES.md for the full index)")
    w()

    w(f"## Threshold sweep ({args.split}, after fees and slippage)")
    w()
    cols = ["theta", "n_traded", "trade_rate", "total_pnl", "pnl_per_1k_deployed",
            "gross_pnl_per_1k_deployed", "fee_per_1k_deployed", "win_rate", "sharpe",
            "avg_holding_candles"]
    if args.split != "val":
        w(f"This sweep is **diagnostic only** — theta was selected on val and "
          f"frozen before {args.split} was scored. Nothing here chose anything; "
          f"reading a better theta off this table and reporting it would make "
          f"the {args.split} number a tuned one.")
        w()
    w("```")
    w(_fmt(sweep[cols].round(4)))
    w("```")
    w()
    w(f"- minimum trades required: **{MIN_TRADES_FOR_SELECTION}**")
    w(f"- selected theta: **{theta_note}**")
    eligible = selection_sweep[selection_sweep.n_traded >= MIN_TRADES_FOR_SELECTION]
    w(f"- eligible (on the val selection sweep): {eligible.theta.tolist()}")
    w(f"- excluded (too few trades): "
      f"{selection_sweep[selection_sweep.n_traded < MIN_TRADES_FOR_SELECTION].theta.tolist()}")
    if args.split != "val":
        w()
        w("Val selection sweep, for the record:")
        w("```")
        w(_fmt(selection_sweep[cols].round(4)))
        w("```")
    w()
    if len(eligible) > 1:
        monotone = eligible.pnl_per_1k_deployed.is_monotonic_decreasing
        if monotone:
            w("PnL is **monotone decreasing** across the eligible grid, so the "
              "selector returned the smallest candidate. That is a boundary "
              "selection, not a tuned optimum: there is no interior peak and no "
              "plateau. Do not describe it as tuned.")
            w()

    w(f"## Trading performance ({args.split})")
    w()
    table = comparison_table(metrics)
    show = ["policy", "n_markets", "n_traded", "total_pnl", "pnl_per_1k_deployed",
            "gross_pnl_per_1k_deployed", "fee_per_1k_deployed", "win_rate", "sharpe",
            "max_drawdown", "turnover", "total_fees", "fee_fraction_gross_pnl",
            "avg_holding_candles", "total_fills"]
    w("```")
    w(_fmt(table[show].round(4)))
    w("```")
    w()
    w("### Fee-fraction denominator check")
    w()
    w("```")
    w(fee_denominator_check(results).to_string())
    w("```")
    w()
    w("`fee_frac_net_gross` divides by realised PnL with fees added back, losing "
      "markets contributing their actual negative PnL, and is NaN when that is "
      "<= 0. The positive-only convention drops every loss from the denominator "
      "while its fees stay in the numerator, so it reads best exactly when a "
      "policy is doing worst. Both are shown so the gap is visible; the paper "
      "must pick one for all models or the comparison means nothing.")
    w()

    w("## Warnings emitted")
    w()
    all_warns = train_warns + eval_warns
    if all_warns:
        w("```")
        for line in all_warns:
            w(line)
        w("```")
        w()
        w("These are deliberate sample cuts. Report the effective market counts, "
          "not the universe counts.")
    else:
        w("None.")
    w()

    w("## Test-split confirmation")
    w()
    if args.split == "test":
        w("**This run READ the test split** (`--allow-test` was passed). That is "
          "the single budgeted look; any further tuning invalidates it.")
    else:
        w(f"`load_split('test', ...)` was not called in this run; the reported "
          f"split is `{args.split}`.")
        w()
        w("One caveat, stated rather than buried: `market_universe()` reads "
          "`event_slug`, `start_ts`, `winner` and `truncated` from all daily "
          "files in order to compute the split boundaries, so test-period "
          "*labels* are read into memory by the bounds calculation. They never "
          "reach features, fitting, threshold selection, or any reported metric "
          "-- the test row above shows dates and counts only -- but \"the test "
          "bytes were never touched\" would be false.")
    w()

    with open(args.report, "w", encoding="utf-8") as fh:
        fh.write(out.getvalue())
    print(f"  wrote {args.report}")

    print("\n" + _fmt(table[["policy", "total_pnl", "pnl_per_1k_deployed",
                             "win_rate", "sharpe"]].round(4)))


if __name__ == "__main__":
    main()
