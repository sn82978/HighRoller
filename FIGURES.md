# Figure index

Every figure in the repository, where it lives, and the one command that
rebuilds it. Cite paths from this file — the progress report's `Figs/7T2E/` and
`Figs/8T2E/` are wrong as written and have never existed under those names (see
[Archived figures](#archived-figures) below).

Every path here is written from the repository root. Every command is run from
the repository root.

---

## Rebuilding everything, in order

The three model tracks are independent; the comparison reads all three.

```bash
python strategies/generate_trades.py --split val     # -> strategies/output/markets.csv (no figures)
python BaselineModels/run_baselines.py --split val   # -> figs/, RESULTS.md, BaselineModels/output/
python QLearning/training.py                         # -> QLearning/figs/, QLearning/metrics/
python QLearning/analysis.py                         # -> QLearning/metrics/figs/qlearning/
python sim/compare_models.py --split val             # -> comparison_figs/, comparison*.csv
```

`QLearning/training.py` is the long one: 30 seeded runs × 6,789 episodes. It
takes a lockfile, so a second concurrent invocation refuses rather than
interleaving its appends into the same CSV.

**Every track must run at the same `--slippage`.** They all default to `0.25`
now, and `compare_models.py` refuses to build a table whose rows disagree — each
market row records the cost model it was simulated under. This is not
hypothetical: `generate_trades.py` used to default `0.0` while everything else
defaulted `0.25`, so running each track's own documented command produced a
table that silently priced the rule strategies differently from the models
beside them. It moved `momentum_flip` from −77 to −273 per \$1k and flipped the
sign of its gross edge.

### Scoring the held-out split

Every entry point refuses `test` unless `--allow-test` is spelled out.

```bash
python strategies/generate_trades.py --split test --allow-test
python BaselineModels/run_baselines.py --split test --allow-test
python QLearning/evaluate_split.py --split test --allow-test
python sim/compare_models.py --split test --allow-test
```

The Q-learning step is `evaluate_split.py`, **not** `training.py`. It replays
the 30 already-trained Q-tables from `QLearning/models/` greedily over the test
markets. Retraining would produce 30 different agents, and the val→test
comparison would then be two unrelated runs rather than a generalisation gap.

`markets.csv` keeps every split it has been given, so scoring test does not
destroy the val rows and re-scoring one split replaces rather than doubles it.

---

## Current figures

### `figs/` — XGBoost calibration

| file | what it shows | rebuilt by |
|---|---|---|
| `xgb_calibration_<split>.png` | predicted probability vs realised frequency, decile bins, against the diagonal | `python BaselineModels/run_baselines.py --split <split>` |

Cited by `RESULTS.md`. The reliability diagram behind the reported ECE.

### `comparison_figs/` — the four-way comparison

| file | what it shows | rebuilt by |
|---|---|---|
| `headline_bars_<split>.png` | six panels — total P&L, avg return, win rate, profit factor, Sharpe, max drawdown — every policy side by side | `python sim/compare_models.py --split <split>` |
| `equity_curves_<split>.png` | cumulative P&L by market, chronological | same |

These are the proposal's headline deliverable and the only place all five
policies plus the no-trade floor appear on identical markets under identical
fees. Read them with two things in mind:

- **A collapsed sweep is drawn as a band, not a line.** The Q-learning row is
  the mean of 30 seeds; on the equity curves each seed is a faint grey line
  under a bold mean, and on the bar chart it carries a ±1 sd whisker across
  seeds. Those whiskers cross zero on win rate, profit factor and Sharpe. That
  is the finding, not a plotting artefact.
- **Whiskers mean two different things** and the subtitle says which: bootstrap
  95% CI over markets on the single-run rows' `avg_return`, ±1 sd across seeds
  on the sweep row. They are not comparable to each other.

The numbers behind them are `comparison.csv` and `comparison_spread.csv` (the
across-seed sd, same columns). Both are committed alongside the PNGs.

### `QLearning/figs/` — per-run training diagnostics

120 files, 4 per seed, `qlearning_seed<NN>_<kind>.png` for `NN` in `00`–`29`.

| kind | what it shows |
|---|---|
| `_rewards` | per-episode reward in dollars, under a 100-episode moving average |
| `_epsilon` | the exploration schedule actually followed, episode by episode |
| `_action_frequencies` | per-episode count of each action — HOLD, BUY_UP, BUY_DOWN, SELL — one line each, over training |
| `_eval_action_sequences` | heatmap of the greedy policy on the validation markets: one row per episode, one column per step, coloured by action |

Rebuilt by `python QLearning/training.py`. The seed is in the filename, so a
figure names the run that produced it, and re-running overwrites in place
rather than accumulating a second generation beside the first.

### `QLearning/metrics/figs/qlearning/` — across-run distributions

50 PNGs (25 metrics × `Training` / `Validation`) plus `Training_summary.csv`
and `Validation_summary.csv` (runs, mean, sd, min, max per metric).

Rebuilt by `python QLearning/analysis.py`, which reads
`QLearning/metrics/qlearning_<split>.csv` and writes one histogram per numeric
column, each with its mean drawn as a dashed line. Every metric name is one
`sim/metrics.py` defines, so these are directly comparable to the other tracks.

Two of them need a caption if used:

- **`*_avg_holding_candles`** centres near 2.4, not 60. The progress report
  §4.2 says the agent holds ~59.9 of a maximum 60 candles and "almost always
  rides a position to forced settlement". That number was episode length, not
  holding period. Corrected, the agent round-trips within about 40 seconds and
  exits early in roughly 46% of the markets it trades.
- **`*_fee_fraction_gross_pnl`** is NaN wherever gross P&L is ≤ 0, which on
  validation is 25 of the 30 seeds. Its mean is over the five positive runs, so
  it must not be quoted as "fees are 15% of gross" for the sweep.

### Not produced by any script

The `strategies/` track writes no figures — only `strategies/output/markets.csv`
and the console tables in `strategies/analyze_trades.py`. If the writeup needs a
picture of `momentum_flip` or `buy_and_hold_down`, take it from
`comparison_figs/`, where both appear.

---

## Archived figures

`Figs/7T2E/` and `Figs/8T2E/` do not exist and never did. The report's citations
point at a layout that was assembled by hand and that no code path can rebuild.

| report says | actually was | now at |
|---|---|---|
| `Figs/7T2E/` | `QLearning/figs/0.7T_0.2E/` | `QLearning/figs/archive_pre_refactor/0.7T_0.2E/` |
| `Figs/8T2E/` | `QLearning/figs/` — loose at the top level, unsuffixed | `QLearning/figs/archive_pre_refactor/0.8T_0.2E/` |
| §4.2 histograms | `QLearning/metrics/figs/{0.7T,0.8T}_0.2E/` | `QLearning/metrics/figs/archive_pre_refactor/` |

**Do not cite these.** They are kept only so the report's claims stay traceable
to something. They came from unseeded runs, on a random day-file split rather
than the temporal one, in an environment that filled on the same candle it read
state from and charged no slippage; and three of the eleven metrics they plot do
not measure what their titles say. The READMEs in each archive directory give
the full account, and the replacement figure for each is listed above.
