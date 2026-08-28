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

> **The numbers in this file were computed over the held-out test split.**
> `--split all` used to be the default here, and it built its universe by calling
> `load_split("test", allow_test=True)` with the flag hardcoded, so the ordinary
> invocation read all 1,332 test markets. Every figure in the tables below --- and
> in the progress report's Tables 6 and 7, which come from them --- is therefore
> computed over train + val + test, while the report states in two places that the
> test split has never been read.
>
> `--split all` and `--split test` now refuse to run without an explicit
> `--allow-test`, and the default is `val`. Use `dev` (train + val) to iterate.
> The tables below are stale and are regenerated on `dev` before they are cited.

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

## Results, `--split all`, zero slippage

**Stale and test-contaminated --- see the warning at the top.** This table covers
train + val + test. It is kept here only so the contaminated figures are
identifiable if they turn up in a draft; regenerate on `dev` before citing
anything. The `max drawdown` signs are also from the old `sim.evaluation.score`,
which reported drawdown as a negative number; it is a positive magnitude now.

| | momentum_flip | buy_and_hold_down |
|---|---|---|
| markets traded | 8,548 (99.9%) | 8,555 (100%) |
| total P&L on $855,500 staked | **−$54,317** | **−$28,250** |
| avg return per market | −6.35% | −3.30% |
| 95% CI (bootstrap) | [−8.15%, −4.59%] | [−5.44%, −1.34%] |
| win rate | 58.5% | 50.1% |
| profit factor | 0.82 | 0.94 |
| Sharpe per market | −0.074 | −0.032 |
| t-stat vs zero edge | −6.88 | −3.01 |
| max drawdown | −$55,143 (551 stakes) | −$29,729 (297 stakes) |

Both lose money even at zero slippage once the 7% taker fee is charged on every
non-settlement fill. `momentum_flip` has a noticeably higher win rate (58.5% vs 50.1%,
so there is real signal in the tape) but trades often enough that fees eat past the edge.

## It only gets worse with slippage

Slippage is `ExecutionConfig.slippage_frac`, a fraction of the candle's
high-low range (median high-low on this book is 0.04), not a flat price-unit
number:

| slippage_frac | momentum_flip | buy_and_hold_down |
|---|---|---|
| 0.00 | −$54,317 | −$28,250 |
| 0.10 | −$117,741 | −$39,988 |
| 0.25 | −$206,455 | −$56,718 |
| 0.50 | −$335,974 | −$82,534 |
| 1.00 | −$536,151 | −$127,799 |

No slippage level makes either strategy profitable here -- the fee is the binding
constraint, not execution quality.

## Output files

| file | grain |
|---|---|
| `output/fills_<split>.csv` | every fill, flattened from `sim.execution.Portfolio.trades` |
| `output/markets.csv` | one row per market per strategy, in `sim.evaluation.MARKET_RECORD_FIELDS` |
| `output/summary.csv` | one column per strategy, every metric from `sim.evaluation.score` |
| `output/equity_curve.csv` | cumulative P&L and compounded equity per market |
| `output/slippage_sweep.csv` | P&L vs assumed adverse fill |

Outputs are gitignored (`*.csv`) — regenerate with the commands above.

## Useful flags

```bash
--split {train,val,test,dev,all}  # market universe; 'dev' is train+val
--allow-test                    # required for 'test' or 'all'; one run, at the very end
--threshold 0.6                 # move the entry/flip trigger
--slippage 0.25                 # ExecutionConfig.slippage_frac (fraction of candle H-L range)
--days 30                       # last N days within the split, for a quick run
--hold-side Up                  # flip the buy-and-hold leg
```

`analyze_trades.py --fraction` sets the bankroll fraction for the compounded view.
It defaults to 0.10 because rolling the **full** bankroll is guaranteed ruin: any
market resolving against you is a −100% return and zeroes the account.
