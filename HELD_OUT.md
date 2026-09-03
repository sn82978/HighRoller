# The held-out pass

The test split was scored **once**, on 2026-08-28, at commit `ec75bb2`. This
file is the record of that run. It is not regenerated as a matter of course:
re-running any of these commands spends a budget that only existed once.

```bash
python strategies/generate_trades.py  --split test --allow-test
python BaselineModels/run_baselines.py --split test --allow-test --report RESULTS_test.md
python QLearning/evaluate_split.py     --split test --allow-test
python sim/compare_models.py           --split test --allow-test --out comparison_test.csv
```

Cost model: taker fee `0.07·p·(1−p)`, adverse slippage 0.25 of the fill candle's
high-low range, $100 per entry, settlement fee-free. Verified identical across
all four tracks — every market row records the slippage it was simulated under
and `compare_models.py` refuses to table rows that disagree.

Test window: 2026-02-13 → 2026-02-26. 1,329 markets, the set all four tracks
cover.

## Disclosure: what "held out" does and does not mean here

**For `buy_and_hold`, `no_trade`, `xgboost_theta_0.01` and the Q-learning agent,
this is a genuine first look.** Nothing was fitted, selected, or inspected on
test:

- XGBoost was fitted on train, early-stopped on val.
- θ was swept on **val** and frozen. The test sweep in `RESULTS_test.md` is
  printed as a diagnostic and selected nothing. (Reading a better θ off it
  would make the number a tuned one; the code no longer permits it.)
- The 30 Q-tables are the ones trained on train and reported on val. They were
  replayed here, not retrained.

**For `momentum_flip` and `buy_and_hold_down` it is not.** Those two were
previously run over the test markets by an earlier version of
`strategies/generate_trades.py`, which defaulted to `--split all` and reached
the test block through a hardcoded `allow_test=True`. Nothing was tuned on the
result — these are fixed rules with no fitted parameters, and the 0.55 threshold
predates the exposure — but the split was read, and the progress report states
in two places that it never was. That is corrected here rather than papered
over. See the warning at the top of `strategies/README.md`.

## Result

Sorted best to worst. Per \$1k of capital at risk.

| policy | traded | PnL/\$1k | gross/\$1k | fees/\$1k | win rate | Sharpe | holding |
|---|---|---|---|---|---|---|---|
| **no_trade** | 0 | **0.00** | 0.00 | 0.00 | — | — | — |
| xgboost θ=0.01 | 1,329 | −38.60 | −7.73 | 30.86 | 0.555 | −8.03 | 58.1 |
| buy_and_hold | 1,329 | −72.88 | −41.94 | 30.94 | 0.536 | −15.24 | 59.0 |
| qlearning (mean of 30) | 239 | −90.68 | −64.95 | 25.72 | 0.537 | −6.04 | 2.8 |
| buy_and_hold_down | 1,329 | −103.56 | −69.79 | 33.76 | 0.484 | −19.96 | 59.0 |
| momentum_flip | 1,328 | −274.64 | −156.46 | 118.18 | 0.492 | −51.49 | 57.1 |

**The no-trade floor wins on the held-out split, by a wider margin than on
validation.** Every policy that trades loses money, and every one of them loses
money *before* fees as well as after.

## What changed from validation to test

| policy | PnL/\$1k val → test | gross/\$1k val → test |
|---|---|---|
| momentum_flip | −272.70 → −274.64 | −155.84 → −156.46 |
| buy_and_hold_down | −36.49 → −103.56 | −2.57 → **−69.79** |
| no_trade | 0.00 → 0.00 | 0.00 → 0.00 |
| buy_and_hold | −9.93 → −72.88 | **+20.89 → −41.94** |
| xgboost θ=0.01 | −32.80 → −38.60 | −1.56 → −7.73 |
| qlearning (mean of 30) | −59.58 → −90.68 | −36.08 → −64.95 |

Two things are worth stating plainly.

**The one positive result on validation does not survive.** On val,
`buy_and_hold` was the only policy with a positive gross edge (+\$20.89 per
\$1k) and the natural reading was "there is a real edge, and fees eat it". On
test its gross edge is **−\$41.94** per \$1k. The edge was a property of the
validation fortnight, not of the market. The fee argument stands — fees are
~\$31 per \$1k for every hold-to-settlement policy, which is more than enough
to sink a positive edge — but on this data there was no positive edge left to
sink. The honest claim is the weaker and more robust one: **at these fees, none
of these policies clears the no-trade floor, and the two effects are separable
only on validation.**

**`momentum_flip` is the only policy that barely moves** (−272.70 → −274.64).
It trades 3.8× turnover and pays \$118 per \$1k in fees; the fee drag so
dominates its P&L that the underlying market regime barely registers. Its
stability is a symptom, not a strength.

`xgboost_theta_0.01` degrades least among the trading policies (−32.80 →
−38.60) and has the best held-out P&L of any of them, but it still trades every
market at θ=0.01 and still loses to doing nothing by \$38.60 per \$1k.

## Reading the Q-learning row

The collapsed row is the mean of 30 per-seed scores, and the seeds do not agree.
Across-seed spread on test (`comparison_test_spread.csv`):

| metric | mean | sd | median | min | max |
|---|---|---|---|---|---|
| n_traded | 239.3 | 195.6 | 236 | 1 | 637 |
| total_pnl | −836.5 | 1,680.8 | −328.4 | −5,602.6 | +1,050.1 |
| pnl_per_1k_deployed | −90.7 | 123.3 | −57.8 | −493.3 | +20.2 |
| win_rate | 0.537 | 0.358 | 0.514 | 0.00 | 1.00 |
| sharpe | −6.04 | 24.51 | −8.92 | −41.66 | +55.81 |
| avg_holding_candles | 2.76 | 2.45 | 2.18 | 1.0 | 13.0 |

7 of 30 seeds finished positive. The sd exceeds the mean on every headline
metric. At 9,000 states, 4 actions and 6,789 episodes over 5,915 training
markets, the policy is dominated by which markets a given seed happened to
visit, and one run of this agent tells you almost nothing.

Two numbers in this row must not be quoted as they stand:

- **`profit_factor` reads 9.85. Do not use it.** It is a ratio of sums, so its
  mean across seeds is a mean of ratios. Three seeds happened to have almost no
  losing markets and scored 207.16, 53.68 and 13.02; the median is **0.457** and
  23 of 30 seeds lost money. `compare_models.py` now prints a warning whenever a
  ratio metric's mean and median diverge like this.
- **`fee_fraction_gross_pnl` is defined for only 8 of the 30 seeds** — it is NaN
  wherever gross P&L is ≤ 0. Any mean of it describes the profitable minority,
  not the sweep.

## The report's holding-period claim, on held-out data

Progress report §4.2 states the agent held "59.9 of a maximum 60 candles",
concluding that "the learned policy almost always rides a position to forced
settlement rather than exiting early with SELL".

On test the agent holds **2.76 candles** (median 2.18) and exits early in
**54.8%** of the markets it trades. On val it was 2.39 and 45.6%. The report's
figure was episode length, which is ~60 by construction and would read the same
for an agent that never opened a position. The behavioural claim built on it is
false in both directions: the agent round-trips in well under a minute and pays
two taker fees for the privilege.

## Files

| file | what |
|---|---|
| `comparison_test.csv` | the table above |
| `comparison_test_spread.csv` | mean/sd/median/min/max across the 30 seeds |
| `comparison_figs/headline_bars_test.png` | six-panel bar chart, sd whiskers on the sweep |
| `comparison_figs/equity_curves_test.png` | cumulative P&L, seeds as a band |
| `RESULTS_test.md` | the baselines track's own report, incl. calibration and the diagnostic θ sweep |
| `figs/xgb_calibration_test.png` | held-out reliability diagram |
| `QLearning/metrics/qlearning_test.csv` | per-seed scores |
