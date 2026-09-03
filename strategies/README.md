# Rule-based baselines on the 15s candle tape

Two hand-written strategies to benchmark the learned agents against.

```bash
python strategies/generate_trades.py --split val     # writes fills / markets CSVs
python strategies/analyze_trades.py --split val      # scores them, prints the metrics
python strategies/sweep_slippage.py --split val      # re-runs both across fill assumptions
```

`--slippage` defaults to **0.25**, matching every other track. It used to default
to `0.0` here while this file's own help text said 0.25, so the documented
command priced these fills differently from the models these strategies are
tabled against --- worth 196 per \$1k on `momentum_flip`, and a sign flip on its
gross edge. `sim/compare_models.py` now refuses to build a table whose tracks
disagree on the cost model; each market row records the slippage it was
simulated under.

`markets.csv` keeps every split it has been given, so scoring one does not erase
another --- which is why `analyze_trades.py` now needs `--split` to say which one
to score.

Fills/fees/slippage go through `sim.execution.Portfolio` via `sim.evaluation.simulate_market`
-- same engine as `BaselineModels/xgb_baseline.py` and `QLearning/`, so P&L numbers are
comparable across models. Market universe comes from `sim.evaluation.load_universe_candles`
(`--split {train,val,test}`, `dev` for train+val, or `all` for the whole tape). See
`sim/compare_models.py` to put this side by side with the other models.

> **These two strategies have seen the held-out test split, and the progress
> report says they haven't.** `--split all` used to be the default here, and it
> built its universe by calling `load_split("test", allow_test=True)` with the
> flag hardcoded, so the ordinary invocation read all 1,332 test markets. The
> progress report's Tables 6 and 7 come from that run, and the report states in
> two places that the test split has never been read.
>
> Nothing was tuned on the result --- these are fixed rules with no fitted
> parameters, and the 0.55 threshold predates the exposure --- but the split was
> read, so the held-out numbers for `momentum_flip` and `buy_and_hold_down` are
> not a genuine first look and are disclosed as such in
> [HELD_OUT.md](../HELD_OUT.md). The other four policies are clean.
>
> `--split all` and `--split test` now refuse to run without an explicit
> `--allow-test`, and the default is `val`. The tables below are regenerated on
> val; the contaminated ones they replaced are described at the end of this
> file so they stay identifiable if they turn up in a draft.

Note: an older version of this script charged slippage but never charged Polymarket's
7% taker fee, so if you've seen a `momentum_flip` number like `+$28,920` floating around,
that's from before the fee was added. Numbers below include it.

## The strategies

**`momentum_flip`** — wait until either side trades at **≥ 0.55**, buy it. Whenever
the *other* side crosses 0.55, sell the whole position and roll the proceeds into
that side. Whatever is held at the bell goes to resolution.

**`buy_and_hold_down`** — buy Down ("No" on Up) at the open of candle 0 and hold to
resolution regardless of what happens. `--hold-side Up` for the other leg.

## Conventions

- The candle tape is one consolidated series quoted **Up-equivalent**, so a side's
  price is `Up: close`, `Down: 1 - close` (see [data/README.md](../data/README.md)).
- Only the live window is traded — `candle_index` 0…59, no pre-open tape.
- **No lookahead.** A signal is read off candle *i*'s close and filled at candle
  *i+1*'s **open**. Candle 59's close therefore cannot be traded on.
- Settlement pays $1/share to the winning side, $0 to the loser, **fee-free**
  (see `sim/execution.py`); every other fill pays Polymarket's taker fee.
- $100 bankroll per market (`ExecutionConfig.stake_dollars`), not carried between
  markets; a flip closes the old leg and rolls the proceeds into the new one.
- 53 of 8,608 markets are skipped for an incomplete or unresolved live window.

## Results

Validation (2026-01-30 → 2026-02-13, 1,332 markets), \$100 per entry, slippage
0.25, taker fee `0.07·p·(1−p)`. Regenerate with the commands at the top;
`sim/compare_models.py` puts these beside the forecaster and the RL agent.

| | momentum_flip | buy_and_hold_down |
|---|---|---|
| markets traded | 1,331 | 1,332 |
| total P&L | **−\$36,296** | **−\$4,861** |
| P&L per \$1k at risk | −\$272.70 | −\$36.49 |
| — of which gross | −\$155.84 | −\$2.57 |
| — of which fees | \$116.86 | \$33.93 |
| avg return per market | −27.25% | −3.65% |
| 95% CI (bootstrap) | [−32.35%, −22.02%] | [−8.84%, +1.42%] |
| win rate | 50.0% | 51.7% |
| profit factor | 0.46 | 0.93 |
| Sharpe (annualised) | −51.26 | −6.99 |
| t-stat vs zero edge | −9.99 | −1.36 |
| max drawdown | \$36,314 | \$8,608 |
| turnover | 3.74× | 1.00× |

Both lose money, and both lose money **before** fees as well as after —
`momentum_flip` gives up \$155.84 per \$1k on execution alone, then pays another
\$116.86 in fees on top. Its 3.74× turnover is the whole story: it pays roughly
four taker fees per market to earn a coin-flip win rate.

`buy_and_hold_down` is the milder failure. Its gross edge is −\$2.57 per \$1k —
statistically indistinguishable from zero (the CI spans it, t = −1.36) — and it
is the \$33.93 fee that makes the loss. That is the clean version of this
project's thesis, and it is the one policy here where it holds.

On the held-out split neither improves and `buy_and_hold_down` gets much worse:

| | momentum_flip | buy_and_hold_down |
|---|---|---|
| P&L per \$1k, val → test | −272.70 → −274.64 | −36.49 → −103.56 |
| gross per \$1k, val → test | −155.84 → −156.46 | −2.57 → **−69.79** |

`momentum_flip` barely moves because fee drag dominates it so completely that
the market regime hardly registers — stability as a symptom, not a strength.
`buy_and_hold_down`'s gross edge collapses, which is the same lesson the
`buy_and_hold` (Up) leg teaches in `HELD_OUT.md`: the fortnight-level edge does
not generalise.

## It only gets worse with slippage

Slippage is `ExecutionConfig.slippage_frac`, a fraction of the candle's
high-low range (median high-low on this book is 0.04), not a flat price-unit
number. Validation, total P&L:

| slippage_frac | momentum_flip | buy_and_hold_down |
|---|---|---|
| 0.00 | −\$10,308 | −\$161 |
| 0.10 | −\$21,008 | −\$2,097 |
| **0.25** | **−\$36,296** | **−\$4,861** |
| 0.50 | −\$57,592 | −\$9,135 |
| 0.75 | −\$75,573 | −\$13,047 |
| 1.00 | −\$88,503 | −\$16,648 |

No slippage level makes either strategy profitable, including zero. At
`slippage_frac 0.00` — a free fill at mid, which nobody gets —
`buy_and_hold_down` still loses \$161 purely to the taker fee. The fee is the
binding constraint; execution quality only decides how much worse it gets.

Regenerate with `python strategies/sweep_slippage.py --split val`
(`output/slippage_sweep.csv`).

### The table this replaced

Earlier revisions of this file reported `--split all` at zero slippage:
`momentum_flip` −\$54,317 and `buy_and_hold_down` −\$28,250 over 8,548 markets,
with `momentum_flip` at a 58.5% win rate. Those numbers covered train + val +
**test**, for the reason in the warning at the top, and the progress report's
Tables 6 and 7 come from them. They also predate two fixes: same-candle
execution, and an average return divided by the sum of every entry's notional
rather than by the one \$100 bankroll actually at risk — which reported
`momentum_flip` at **+8.28%** average return on a run whose total was
−\$36,296. Do not cite them.

## Output files

| file | grain |
|---|---|
| `output/fills_<split>.csv` | every fill, flattened from `sim.execution.Portfolio.trades` |
| `output/markets.csv` | one row per market per strategy, in `sim.metrics.MARKET_RECORD_FIELDS`, plus the `slippage_frac` it was simulated under. Holds **every split scored so far** |
| `output/summary.csv` | one column per strategy, every metric from `sim.metrics.score_records` |
| `output/equity_curve.csv` | cumulative P&L and compounded equity per market |
| `output/slippage_sweep.csv` | P&L vs assumed adverse fill, scored by `score_records` |

`output/` is gitignored — these are intermediate per-market rows, and the
scored results live in `comparison.csv` and `RESULTS.md` at the repo root.
Regenerate with the commands at the top.

## Useful flags

```bash
--split {train,val,test,dev,all}  # market universe; 'dev' is train+val. Default val
--allow-test                     # required for 'test' or 'all'; one run, at the very end
--threshold 0.6                  # move the entry/flip trigger (default 0.55)
--slippage 0.25                  # ExecutionConfig.slippage_frac, fraction of candle H-L range.
                                 # Default 0.25 -- keep it matched to the other tracks
--days 30                        # last N days within the split, for a quick run
--hold-side Up                   # flip the buy-and-hold leg
```

`analyze_trades.py --fraction` sets the bankroll fraction for the compounded view.
It defaults to 0.10 because rolling the **full** bankroll is guaranteed ruin: any
market resolving against you is a −100% return and zeroes the account.
