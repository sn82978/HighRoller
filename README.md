# HighRoller

CS 175, summer 2026. Can a model predict Polymarket's **BTC Up or Down 15m**
markets well enough to trade them profitably after fees?

**Short answer: no — and on held-out data, not even before fees.** Doing nothing
beats every policy we built. `HELD_OUT.md` is the result; this file is the map.

## The result

Test split (2026-02-13 → 2026-02-26), 1,329 markets every track covers, \$100
per entry, taker fee `0.07·p·(1−p)`, adverse slippage 0.25 of the fill candle's
range. Per \$1k of capital at risk:

| policy | traded | PnL/\$1k | gross/\$1k | fees/\$1k | Sharpe |
|---|---|---|---|---|---|
| **no_trade** | 0 | **0.00** | 0.00 | 0.00 | — |
| xgboost θ=0.01 | 1,329 | −38.60 | −7.73 | 30.86 | −8.03 |
| buy_and_hold | 1,329 | −72.88 | −41.94 | 30.94 | −15.24 |
| qlearning (mean of 30 seeds) | 239 | −90.68 | −64.95 | 25.72 | −6.04 |
| buy_and_hold_down | 1,329 | −103.56 | −69.79 | 33.76 | −19.96 |
| momentum_flip | 1,328 | −274.64 | −156.46 | 118.18 | −51.49 |

The interesting part is what changed between validation and test. On val,
`buy_and_hold` was the one policy with a **positive** gross edge (+\$20.89 per
\$1k) that fees turned negative — a clean "there is an edge, and the fee eats
it" story. On test its gross edge is **−\$41.94**. The edge belonged to the
validation fortnight, not to the market.

Fees are real and large — about \$31 per \$1k for any policy that holds to
settlement, comfortably enough to sink a genuine edge. But on this data there
was no edge left to sink, so the claim has to be the weaker and more robust one:
**at these fees, none of these policies clears the no-trade floor, and the
"edge minus fees" decomposition only holds on validation.**

## Layout

| directory | what |
|---|---|
| `data/` | the fetchers and the dataset — 8,608 markets, 2025-11-28 → 2026-02-26, 15s OHLCV rebuilt from 44.9M individual trades. See [data/README.md](data/README.md). |
| `sim/` | the shared simulator. `execution.py` is fills/fees/slippage, `metrics.py` is the single definition of every trading metric, `evaluation.py` loads splits, `compare_models.py` builds the cross-model table. |
| `BaselineModels/` | XGBoost forecaster plus the no-trade and buy-and-hold floors. `run_baselines.py` regenerates `RESULTS.md`. |
| `strategies/` | two hand-written rules, `momentum_flip` and `buy_and_hold_down`. See [strategies/README.md](strategies/README.md). |
| `QLearning/` | tabular Q-learning over `(price, time, position, pnl)` — 9,000 states, 4 actions. |
| `tests/` | 185 tests. `python -m pytest tests -q`. |

Every track scores through `sim/metrics.py` and prices fills through
`sim/execution.py`, so the comparison is arithmetic rather than translation.

## Reproducing it

```bash
python strategies/generate_trades.py   --split val   # rule strategies
python BaselineModels/run_baselines.py --split val   # forecaster + floors, writes RESULTS.md
python QLearning/training.py                         # 30 seeded runs (slow)
python QLearning/analysis.py                         # across-run histograms
python sim/compare_models.py           --split val   # the table
```

Every track must run at the same `--slippage` (all default to 0.25). Each market
row records the cost model it was simulated under, and `compare_models.py`
refuses to build a table whose rows disagree.

[FIGURES.md](FIGURES.md) indexes every figure and the command that rebuilds it.

### The held-out split

`test` is budgeted for **one** scored pass and every entry point refuses it
without `--allow-test`. That pass has been spent — see [HELD_OUT.md](HELD_OUT.md)
for the commands, the numbers, and what "held out" does and does not mean for
each policy. Re-running those commands does not un-spend it.

## Method notes worth knowing before reading any number

- **Splits are temporal**, 70/15/15 on `start_ts`. No shuffling; a market never
  trains on its own future.
- **No lookahead.** A signal read off candle *c*'s close fills at candle
  *c+1*'s **open**, everywhere. Candle 59 cannot be traded on.
- **Slippage is adverse-only**, sized from the fill candle's own high-low range,
  so it can never manufacture edge.
- **Settlement is fee-free** ($1 to the winner, $0 to the loser); every other
  fill pays the taker fee.
- **Returns are per market allotment**, not per sum-of-entries. A strategy that
  re-enters twelve times still had one \$100 bankroll at risk.
- **The RL sweep does not converge.** Across 30 seeds, markets traded ranges
  from 0 to 598 and the standard deviation exceeds the mean on every headline
  metric. One run of that agent tells you very little; read the spread in
  `comparison_spread.csv`, not the mean alone.

## Documents

| file | what |
|---|---|
| [HELD_OUT.md](HELD_OUT.md) | the one scored pass over the test split |
| [RESULTS.md](RESULTS.md) | baselines on validation — calibration, θ sweep, trading table. Generated; do not hand-edit |
| [RESULTS_test.md](RESULTS_test.md) | the same, on test |
| [FIGURES.md](FIGURES.md) | every figure, and the command that rebuilds it |
| [VERIFICATION.md](VERIFICATION.md) | adversarial audit of the progress report against the code, plus where each finding now stands |
| `docs/` | the proposal and progress report themselves |

`VERIFICATION.md` is worth reading before the progress report. Several claims in
the report do not hold — most notably §4.2's "the learned policy almost always
rides a position to forced settlement", which measured episode length rather
than holding period. The agent actually holds about 2.8 candles.
