# Progress-report verification

Adversarial check of the progress-report claims against the code that actually
executes. Docstrings and comments were **not** accepted as evidence anywhere in
this document; every verdict cites executing code or measured output.

**Revision checked:** `origin/main` @ `6f65365` (local `HEAD` was 6 behind at the
time of writing; all `QLearning/*` line numbers refer to `origin/main`).

> **This document is a snapshot, and it has since been acted on.** Everything
> below describes the code as it was at `6f65365` and is left unedited, because
> it is the record the fixes were planned from and the report's claims are still
> checked against it. Line numbers, file contents and measured numbers in the
> body are therefore all *pre-fix*. See
> [What was done about it](#what-was-done-about-it) for where each finding
> stands now.

**How the empirical numbers were produced.** The committed metrics CSVs do not
record fills, entry prices, or time-in-position, so I reconstructed the 80/20
split with the same seed and logic as `split_data.py`, trained one agent with
the unmodified `q_learning_agent.py` / `make_environment.py`, and evaluated it
with counters attached. Nothing in the repo was modified. My replication landed
at `fee_fraction_gross_pnl` 0.252 vs the committed 0.254, `avg_holding_period`
59.90 vs 59.93, win rate 66.9% vs 64.4% — close enough to treat the behavioural
findings as representative.

One caveat on that reconstruction: it produced **6,796 train / 1,698 eval**
markets against the committed **6,789 / 1,705**. Same 8,494 total, different
partition, because `split_data.py:26` shuffles the output of `os.listdir()`,
whose order is filesystem-dependent. The split is therefore *not* reproducible
across machines even with `random.seed(42)`.

---

## Verdict summary

| # | Claim | Verdict |
|---|---|---|
| 1 | State `(price,time,position,pnl)` = 10/60/3/5, Q-table `10x60x3x5x4` | **CONFIRMED** |
| 2 | Invalid actions masked at selection AND in the backup | **CONFIRMED** |
| 3 | Ties broken uniformly at random | **CONFIRMED** |
| 4 | epsilon 1.0, x0.995/episode, floor 0.02 | **CONFIRMED** |
| 5 | alpha 0.1, gamma 0.99 | **CONFIRMED** |
| 6 | Bootstrap dropped on terminal transitions | **CONFIRMED** |
| 7 | Reward net of taker fee `0.07*p*(1-p)` | **CONFIRMED** |
| 8 | Fee charged on open | **CONFIRMED** |
| 9 | SELL = price change minus exit fee | **CONFIRMED** |
| 10 | Settlement pays $1/$0 with no fee | **CONFIRMED** |
| 11 | 6,789 episodes x 30 runs x 2 configs | **PARTIAL / CAN'T TELL** |
| 12 | Episode loader returns everything | **CONFIRMED (bug fixed)** |
| 13 | Split is strictly temporal | **CONTRADICTED** |
| 14 | 70/20: remaining 10% unused | **CONFIRMED** |
| 15 | Test split never read | **CONFIRMED** |
| 16 | 53 NaN-close markets handled | **CONFIRMED (with a hidden effect)** |
| 17 | Fills at candle close, no slippage/spread | **CONFIRMED (worse than stated)** |
| 18 | `win_rate` per market | **CONFIRMED** |
| 19 | `avg_holding_period` from trade events | **CONTRADICTED** |
| 20 | `turnover` from trade events | **CONTRADICTED** |
| 21 | `fee_fraction_gross_pnl` denominator | **CONTRADICTED** |
| 22 | `pnl_per_1k_capital` normalised to $1,000 | **CONTRADICTED** |
| 23 | `sharpe_ratio` per-market, unannualised, no rf | **CONFIRMED** |
| 24 | Reported 80/20 numbers | **CONFIRMED** |
| 25 | Reported 70/20 numbers | **CONFIRMED (fragile — see 25)** |
| 26 | "Holds to settlement 59.9/60 candles" | **CONTRADICTED** |
| 27 | Figure paths `Figs/7T2E/`, `Figs/8T2E/` | **CONTRADICTED** |

---

## Algorithm claims

### 1. State space and Q-table shape — CONFIRMED

`make_environment.py:46-48` sets `N_PRICE_BUCKETS = 10`, `N_PNL_BUCKETS = 5`,
`MAX_TIME_BUCKET = 59`. `make_environment.py:122-124` returns
`(10, 60, 3, 5)`; `q_learning_agent.py:5` builds
`np.zeros(state_shape + (n_actions,))` with `n_actions=4`. Product is
10x60x3x5 = **9,000 states**, 36,000 Q-cells. The tuple returned at
`make_environment.py:120` is exactly `(price_bucket, time_bucket, position,
pnl_bucket)`.

### 2. Invalid-action masking, both sites — CONFIRMED

Selection: `q_learning_agent.py:18-19` samples the exploration action from
`valid_actions`, and `:22-24` sets every non-valid entry to `-inf` before the
argmax. Backup: `q_learning_agent.py:40` takes
`np.max(next_q_vals[next_valid_actions])`. The caller supplies the mask at both
sites — `training.py:131-132` (selection) and `training.py:137-139` (backup).

`training.py:137` calls `env.get_valid_actions()` *after* `env.step()`, so the
mask reflects the position the agent transitioned **into**. That is correct.

### 3. Random tie-breaking — CONFIRMED

`q_learning_agent.py:27-29` computes the max over masked values, collects every
valid action equal to it, and calls `np.random.choice(best_actions)`. It does
not fall through to index 0. This matters at initialisation, where the table is
all zeros (`:5`) and every valid action ties.

### 4-6. Hyperparameters and terminal handling — CONFIRMED

`q_learning_agent.py:4` — `alpha=0.1, gamma=0.99, epsilon=1.0,
epsilon_decay=0.995, min_epsilon=0.02`, and `training.py:114` constructs the
agent without overriding any of them. Decay at `:46-47`. Terminal handling at
`:36-41`: `target = reward`, and the `gamma * max_next_q` term is added only
`if not done`.

**Surprise worth stating in the report:** `0.995^n = 0.02` at n = 780, so
epsilon reaches its floor at episode 781. In a 6,789-episode run **88.5% of
training happens at epsilon = 0.02**. The decay schedule is doing very little
work; the run is nearly all exploitation.

---

## Reward and fees

### 7-10. All CONFIRMED

- Fee function: `make_environment.py:53-55`, `size * 0.07 * p * (1-p)` with
  `TAKER_FEE_RATE = 0.07` at `:51` and `size` defaulting to 1.0.
- Open: `:162` and `:172` compute the fee; `:164` and `:174` set
  `reward = -fee`.
- SELL: `:181-183` (UP) and `:186-188` (DOWN) — `gross = price - entry_price`,
  `reward = gross - fee`.
- Settlement: `:193-201`. Payout is 1.0/0.0 at `:197`, `reward += gross` at
  `:200`, and **no fee variable is touched anywhere in that block**.

### Hand-traced episode

Real market `btc-updown-15m-1768176000` (winner **Down**, 60 candles), run
through the unmodified env twice.

**Path A — buy at step 3, hold to settlement.** Entry price p = 0.280000.

```
predicted   entry fee = 0.07 * 0.28 * 0.72          = 0.014112
            reward@3  = -0.014112
            reward@59 = payout - entry = 0.0 - 0.28 = -0.280000   (no fee)
            total                                    = -0.294112

observed
step act        price_up    entry       fee     gross     reward        cum
   3 Buy Up       0.2800   0.0000   0.01411   0.00000   -0.01411   -0.01411
  59 Hold         0.0020   0.2800   0.00000   0.00000   -0.28000   -0.29411
```

The fee at step 59 is **0.00000** — settlement is free, as claimed.

**Path B — buy at step 3, sell at step 40.** Exit price p = 0.060000.

```
predicted   exit fee  = 0.07 * 0.06 * 0.94          = 0.003948
            reward@40 = (0.06 - 0.28) - 0.003948    = -0.223948
            total                                    = -0.238060

observed
   3 Buy Up       0.2800   0.0000   0.01411   0.00000   -0.01411   -0.01411
  40 Sell         0.0600   0.2800   0.00395   0.00000   -0.22395   -0.23806
  59 Hold         0.0020   0.0000   0.00000   0.00000    0.00000   -0.23806
```

Path A pays one fee, path B pays two. The fee demonstrably enters the reward.

**Note the `gross` column: 0.00000 on both losing exits.** That is
`make_environment.py:203` wrapping the value in `max(0.0, gross_pnl)`. It is
the root of problem 21 below.

---

## Training and data

### 11. Episodes and runs — PARTIAL

`training.py:242-245` sets `base_modelname = "0.8T_0.2E_qtable"`,
`for i in range(30)`, `NUM_EPISODES=6789`. So **30 runs x 6,789 episodes is
confirmed for the 80/20 configuration only**.

For 70/20 the committed code no longer contains the invocation, and
`NUM_EPISODES` is not written to the CSV, so the episode count is
**unverifiable**. The run count is *not* 30 in the file — see 25.

Coincidence worth knowing: 6,789 is also exactly the number of training markets
in the 80/20 split, so "6,789 episodes" means each market is visited ~once on
average, in random order (`make_environment.py:140`).

### Real market counts (from the CSVs' own `num_markets` column)

| config | train markets | eval markets | held out (never read) |
|---|---|---|---|
| 70/20 | **5,932** | **1,711** | 10 of 91 files |
| 80/20 | **6,789** | **1,705** | 1 of 91 file |

The split is by **file (day)**, not by market: `split_data.py:36-41`. With 91
files, 80/20 gives `train_end = int(91*0.8) = 72`, `val_end = 72 + int(91*0.2)
= 90`, leaving `files[90:]` = **1 file** for test. The "80/20" label describes
train/val; there is effectively no test set in that configuration.

### 12. Episode loader — CONFIRMED, bug is fixed

`data_preparation.py:51`, `:65`, `:79` all now `return all_episodes`. The
previous `return episodes` (last-file-only) bug is gone. The CSV corroborates
it: `num_markets` is 6,789 for training, not ~96.

### 13. Temporal split — CONTRADICTED

`split_data.py:32-33`:

```python
random.seed(42)
random.shuffle(csv_files)
```

The split is **random by day file**. `start_ts` is never consulted anywhere in
`split_data.py`. This directly contradicts the proposal's commitment to a
strictly temporal split.

**How big a deal:** significant, and it is the kind of thing a grader looks for
specifically because the proposal called it out. Shuffling days means the model
trains on February and evaluates on December. It is not per-market lookahead —
a market never straddles the boundary — but any regime or drift structure
leaks across the split, and results are optimistic by an unmeasured amount.
Note also that the baseline pipeline (`BaselineModels/data_loader.py`) *is*
temporal at 70/15/15, so the two model families are currently evaluated on
incompatible splits and their numbers are not directly comparable.

### 14. The remaining ~10% in the 70/20 config — CONFIRMED unused

`train_end = int(91*0.7) = 63`, `val_end = 63 + int(91*0.2) = 81`, so
`test_files = files[81:]` = **10 files** (11.0%, roughly 965 markets). They are
copied to `QLearning/data/test` and never loaded.

### 15. Test split never read — CONFIRMED

The only call site is `training.py:236`, which is **commented out**:

```python
    # test_episodes = get_test_data()
```

`get_test_data` is imported at `training.py:5` but never invoked. Nothing else
in the repo references `TEST_DIR` outside `data_preparation.py:12,69,73`. The
test split is clean.

### 16. The 53 NaN-close markets — CONFIRMED, with an effect the report omits

Measured directly on the dataset: of 8,608 markets, **53** have at least one
NaN `close` inside the live window. They do **not** crash the run.
`data_preparation.py:22` drops those rows:

```python
ep_df = ep_df.dropna(subset=["close"]).copy()
```

But the 53 split two ways, and only one is visible:

- **18 markets** are entirely NaN in the live window. They become empty and are
  skipped at `data_preparation.py:26-28`, which prints the bare string
  `"empty"` — I saw exactly 18 of those in my run.
- **35 markets** are *partially* NaN. Their bad rows are silently deleted, which
  **shortens the episode**. Those episodes have fewer than 60 steps, and the
  forced-settlement branch (`make_environment.py:194`) fires at whatever the
  last surviving row is rather than at candle 59.

That second group is why `avg_holding_period` is 59.9 rather than exactly 60.0.

### 17. Fills — CONFIRMED, and worse than the claim

`data_preparation.py:30-31` sets `price_up = close`, `price_down = 1 - close`.
`make_environment.py:149-150` reads `row = self._ep.iloc[self._i]` and prices
off that same row. No spread, no slippage, no fill-price adjustment anywhere.

Beyond the claim: `_get_state()` (`:86-87`) reads **the same row index** the
trade then executes against. The agent observes candle *c*'s close and
transacts at candle *c*'s close. You cannot trade at a price you have already
watched settle. This is a mild lookahead at the execution layer, independent of
the split issue, and it biases results optimistically.

---

## Metric definitions

### 18. `win_rate` — per MARKET (episode), CONFIRMED

`training.py:55`: `win_rate = float((pnls > 0).mean() * 100)` where `pnls` is
one entry per episode (`:49`). It is the share of **markets** whose summed
reward was positive — not the share of profitable trades. The CSV comment at
`training.py:85` says "percentage of trades", which is wrong; the code is
per-market.

With ~4 fills per market this distinction is large. Report it as "share of
markets ending profitable".

### 19. `avg_holding_period` — episode step count, CONTRADICTED

```
training.py:34   steps = 0
training.py:43   steps += 1          # every step, regardless of position
training.py:50   holding_periods.append(steps)
training.py:68   avg_holding_period = float(np.mean(holding_periods))
```

`steps` increments on every environment step whether the agent holds a
position, sits flat, or trades. It is **episode length**, pinned near 60 by
construction, and it carries no information about the policy. Its only
variation comes from the 35 shortened episodes in 16.

### 20. `turnover` — same quantity, CONTRADICTED

`training.py:69`: `turnover = float(np.sum(holding_periods))` — the sum of the
same step counts. For 80/20 eval it is 102,175, which is exactly
1,705 x 59.9267. It is `num_markets * avg_holding_period` and nothing else. It
measures no trading activity whatsoever.

### 21. `fee_fraction_gross_pnl` — denominator is POSITIVE-only gross, CONTRADICTED

`training.py:70`: `total_fees / gross_pnl`. The numerator accumulates
`info.fee` (`:46`). The denominator accumulates `info.gross_pnl` (`:47`), and
that field is set at `make_environment.py:203`:

```python
info = StepInfo(..., gross_pnl=max(0.0, gross_pnl))
```

So **every losing trade contributes 0 to the denominator while its fee still
counts in the numerator.** The denominator is the sum of the *winning* trades'
gross PnL. It is neither "sum of absolute rewards" nor net gross PnL.

Measured on my replication (eval, 1,698 markets):

```
total fees                                   14.9835
POSITIVE-only gross  (the coded denominator) 59.4092  -> 25.2%
NET gross  (= total_pnl + fees)               4.8570  -> 308.5%
```

Same split, same fees, two definitions, a 12x difference in the headline.

### 22. `pnl_per_1k_capital` — CONTRADICTED

`training.py:59`:

```python
pnl_per_1k_capital = (total_pnl / 1000.0)
```

This is total PnL divided by one thousand. It is **not** a normalisation to
$1,000 of deployed capital — no capital, position size, or stake appears
anywhere in the calculation. There is no position size in this codebase at all:
the env trades one notional unit (`calculate_polymarket_fee(price, size=1.0)`,
`make_environment.py:53`) and rewards are per-share price differences.

Confirm by inspection of the CSV: `pnl_per_1k_capital` equals `total_pnl/1000`
in every row of all four files. Do not describe this as a capital-normalised
return.

### 23. `sharpe_ratio` — CONFIRMED

`training.py:60`:

```python
sharpe_ratio = float(avg_pnl_per_market / np.std(pnls)) if np.std(pnls) > 0 else 0.0
```

Per-market mean over per-market population std (`ddof=0`), **no risk-free rate,
no annualisation**. Describe it exactly that way — an unannualised per-market
Sharpe of -0.027 is not comparable to any conventionally reported Sharpe.

---

## Reproducing the reported numbers

### 24. 80/20 — CONFIRMED, exact

All 30 rows of `0.8T_0.2E_qtable_{eval,training}.csv`:

| metric | reported | computed |
|---|---|---|
| win rate | 64.4% | 64.4438% |
| avg PnL/market | -$0.0074 | -0.007389 |
| total PnL | -$12.60 | -12.5988 |
| Sharpe | -0.049 | -0.04936 |
| fees % of gross | 25.4% | 25.379% |
| runs positive | 0/30 | 0/30 |
| training avg PnL/market | -$0.0054 | -0.005441 |
| avg holding | 59.9 | 59.9267 |

### 25. 70/20 — numbers CONFIRMED, but the file is booby-trapped

The reported figures match **the last 30 rows** of the CSV exactly:

| metric | reported | last 30 | all 35 |
|---|---|---|---|
| win rate | 65.9% | 65.8543% | 66.3004% |
| avg PnL/market | -$0.0039 | -0.00395 | -0.00377 |
| total PnL | -$6.75 | **-6.75151** | -6.44823 |
| Sharpe | -0.027 | -0.02696 | -0.02574 |
| fees % of gross | 24.6% | 0.24600 | 0.24525 |
| runs positive | 3/30 | 3/30 | 3/35 |
| training avg PnL/market | -$0.0048 | -0.00482 | -0.00491 |
| avg holding | 59.9 | 59.9059 | 59.9059 |

**The file contains 35 rows, not 30.** The `iteration` column reads:

```
[0,1,2,3,4, 0,1,2,3,4,5,...,29]
```

An aborted 5-run batch was appended to (`training.py:101` opens with
`mode="a"`), then the real 30-run batch ran. Your reported numbers are correct
because they used the real batch. But anyone — a teammate, a grader, or you in
three weeks — who opens that CSV and takes a plain mean gets **-6.45 instead of
-6.75** and **3/35 instead of 3/30**. The claim "30 runs" is also literally
false about the file as committed.

---

## The reconciliation you asked for

**Your instinct was right that the two claims cannot both mean what they
appear to. The holding-period claim is the false one.**

Measured over 1,698 eval markets with the real agent and env:

```
entries (BUY fills)     3442  ->  2.027 per market
early SELLs             3428  ->  2.019 per market
held to settlement        14  ->  0.008 per market   (0.8%)
total fills             6870  ->  4.046 per market
mean entry price                   0.9602   (median 0.9700)
total fees            14.9835  ->  0.008824 per market
```

Training set, 6,796 markets: 4.071 fills/market, 54 markets held to settlement
(**0.79%**), mean entry price 0.9597.

So:

1. **The agent almost never holds to settlement — it does so in 0.8% of
   markets.** It round-trips roughly twice per market and pays about **four**
   taker fees, not one. The "59.9 of 60 candles" figure is `avg_holding_period`,
   which is episode length (19 above), not time in position. It would read ~60
   even for an agent that never traded at all.

2. **Your fee arithmetic was right in method but wrong in both inputs.** You
   assumed one fill at p ~ 0.66 → 1.6c. Actual is ~4 fills at p ~ 0.96, and
   because `p(1-p)` collapses at the extremes each fill costs only
   `0.07 * 0.96 * 0.04` ~ 0.27c. Four of those is ~1.1c, and the measured
   average is **0.88c per market** — *lower* than your estimate, not higher.

3. **The 25% comes from the denominator, not the fees.** Positive-only gross is
   just 0.035 per market (59.41/1698), so 0.0088/0.035 = 25.2%. Against **net**
   gross PnL of 0.0029 per market, the same fees are **308%**.

**The honest sentence for the report:** *the agent round-trips about twice per
market, paying roughly four taker fees at entry prices averaging 0.96; fees
total about 0.9 cents per market, which is roughly three times the entire net
gross edge the policy generates, and is why every configuration loses money.*

That is a stronger finding than the current framing, and it is defensible.

---

## Figures

**`Figs/7T2E/` and `Figs/8T2E/` do not exist.** Nothing in the repository
matches those names. Actual locations:

| report says | actually, when this was written | since moved to |
|---|---|---|
| `Figs/7T2E/` | `QLearning/figs/0.7T_0.2E/` (120 files, ITER 0-29) | `QLearning/figs/archive_pre_refactor/0.7T_0.2E/` |
| `Figs/8T2E/` | `QLearning/figs/` — **unsuffixed, top level** (120 files, ITER 0-29) | `QLearning/figs/archive_pre_refactor/0.8T_0.2E/` |

> Both sets were archived in the figure pass, along with the §4.2 histograms
> from `QLearning/metrics/figs/{0.7T,0.8T}_0.2E/`. Each archive directory has a
> README; `FIGURES.md` indexes what replaced them. The diagnosis below stands as
> written — it is the reason they were archived rather than regenerated.

Per-figure existence:

| figure | 0.7T | 0.8T | notes |
|---|---|---|---|
| `epsilon` | yes | yes | `training.py:188` |
| `rewards` | yes | yes | `training.py:178` |
| `action_frequencies` | yes | yes | `training.py:202` |
| `eval_action_sequences` | yes | yes | `training.py:232` |
| `Evaluation_total_pnl` | yes | yes | `analysis.py:33-34` |
| `Evaluation_win_rate` | yes | yes | |
| `Training_avg_pnl_per_market` | yes | yes | |
| `Evaluation_avg_pnl_per_market` | yes | yes | |

All eight exist for both configs. Three problems:

- **The directory layout was created by hand, not by the code.**
  `training.py:10` writes to a flat `figs/`, and `analysis.py:6,33` writes to a
  flat `metrics/figs/`. Neither ever creates a per-config subdirectory. The
  `0.7T_0.2E/` and `0.8T_0.2E/` subdirs were introduced by moving files in
  commit `c4adf71`. The code as committed cannot regenerate the layout the
  report cites.
- **`analysis.py` can no longer produce the 0.7T figures at all.** Line 44:
  `if '0.8T_0.2E' not in metrics_csv: continue`. Re-running it regenerates only
  the 0.8T set, and drops them in the flat directory.
- **The 0.7T metric histograms are very likely stale relative to your reported
  numbers.** `analysis.py:48` reads the whole CSV with no row filter, and the
  figure subdirectories were committed in `c4adf71`, *after* the CSV reached 35
  rows in `7b3018d`. So those histograms almost certainly bin **all 35 runs**,
  while your reported statistics use the last 30. I cannot confirm this from
  the PNGs themselves — hence not a hard CONTRADICTED — but the figures and the
  numbers in the report are probably describing different sample sets.

Only 30 runs' worth of training figures exist for 0.7T (ITER 0-29), so the 5
orphan runs have no figures — consistent with them being an aborted batch.

---

## Things a reader of the report would be surprised by

1. **The agent trades constantly and almost never holds to settlement** (0.8%),
   while the report says the opposite.
2. **It buys at an average price of 0.96** — it is overwhelmingly betting on
   near-certain outcomes late in the market, collecting small edges. That is a
   substantive behavioural finding the report does not mention at all.
3. **88.5% of training runs at the minimum epsilon.** The decay schedule
   effectively finishes in the first 12% of the run.
4. **There is no position sizing or capital anywhere.** All PnL is per-share
   price difference on one notional unit, so every dollar figure in the report
   is "dollars per share", not dollars on an account.
5. **The 80/20 config holds out exactly one day file** as test — 1.1%, not 20%.
6. **Runs are not reproducible.** No seed is set for `np.random` (used in
   `q_learning_agent.py:18,19,29`) or for `make_environment.py:84`'s
   `default_rng()`. `split_data.py:32` seeds only the file shuffle, and even
   that depends on `os.listdir` order.
7. **18 markets vanish with a bare `print("empty")`** and no count in any
   output artefact.
8. **Trades execute at the same candle close the state was read from.**

## Things the report describes that the code does not do

- A temporal split (13).
- Holding to settlement (26).
- Capital-normalised PnL (22).
- A trade-based holding period or turnover (19, 20).
- `Figs/7T2E/` and `Figs/8T2E/` (27).
- "30 runs" for 70/20 — the file has 35 (25).
- "per trade" win rate, per the CSV's own column comment (18).

---

## Ranked by how likely this is to cost you with a grader

| # | Problem | Why it matters |
|---|---|---|
| 1 | **Random-by-day split, not temporal** | The proposal explicitly committed to temporal and named lookahead as the top risk. A grader who reads the proposal will check this line first. It also makes the RL and baseline numbers non-comparable. |
| 2 | **"Holds to settlement" is false — 0.8%, ~4 fills/market** | A central qualitative claim about the learned policy, contradicted by the policy's actual behaviour. Cheap to check, embarrassing if found. |
| 3 | **`avg_holding_period` and `turnover` are episode length** | Two of the proposal's named metrics measure nothing. Both are ~60 and ~60x N by construction and would look identical for an agent that never traded. |
| 4 | **`fee_fraction_gross_pnl` discards all losses** | Makes the headline cost figure 12x too flattering (25% vs 308%). A grader who asks "gross of what?" finds it immediately. |
| 5 | **`pnl_per_1k_capital` is `total_pnl/1000`** | The name asserts a normalisation that does not exist. Trivially checkable against the CSV. |
| 6 | **70/20 CSV has 35 rows, reported as 30** | Your numbers are right, but nobody can reproduce them from the file without knowing to drop the first five rows. Undocumented, and it is exactly the kind of thing that reads as cherry-picking even though it isn't. |
| 7 | **Figure paths in the report do not exist; 0.7T figures unreproducible and probably stale** | Citation integrity. Fast for a grader to spot and it undermines everything around it. |
| 8 | **Same-candle execution** | Real methodological flaw, but subtle and easy to state as a limitation. |
| 9 | **Effectively no test set in 80/20 (1 file)** | Fine if you never claim a test result — and you never do (15). Only a problem if the report implies otherwise. |
| 10 | **No RNG seeding; split not portable** | Reproducibility, not correctness. Worth one sentence in limitations. |
| 11 | **88.5% of training at epsilon floor** | Tuning observation, not an error. |
| 12 | **18 markets silently dropped** | Small sample cut, one line to disclose. |

---

## What was done about it

The diagnosis above was the plan. This section records where each finding
stands; the body of the document is deliberately not updated to match.

### The ranked list

| # | Problem | Status |
|---|---|---|
| 1 | Random-by-day split, not temporal | **Fixed** (`38d8f03`). `data_preparation.py` reads the canonical 70/15/15 temporal split through `sim.evaluation.load_split_candles`, the same universe every other track uses. `split_data.py` is marked SUPERSEDED and nothing imports it. |
| 2 | "Holds to settlement" is false | **Measured and reported.** The corrected agent holds **2.39** candles on val and **2.76** on test, exiting early in 46% / 55% of the markets it trades. The report's 59.9 was episode length. See `HELD_OUT.md`. |
| 3 | `avg_holding_period` and `turnover` are episode length | **Fixed** (`6bf7d09`). Both are computed in `sim/metrics.py` from trade events — `avg_holding_candles` from entry/exit candles, `turnover` from notional traded over capital. |
| 4 | `fee_fraction_gross_pnl` discards all losses | **Fixed** (`6bf7d09`, `9456edd`). The `max(0, ...)` clamp is gone from the RL reward path and fee drag is computed once, on net gross PnL. It is NaN rather than 0.0 when gross ≤ 0 — see the caveat below. |
| 5 | `pnl_per_1k_capital` is `total_pnl/1000` | **Fixed** (`6bf7d09`). Replaced by `pnl_per_1k_deployed`, normalised by the per-market allotment actually at risk. |
| 6 | 70/20 CSV has 35 rows, reported as 30 | **Retired** (`7a3743d`). Both "configurations" were filename prefixes over a split that no longer exists. Replaced by 30 seeded runs on the canonical split; the old CSVs are in `QLearning/metrics/archive_pre_refactor/` with a README. |
| 7 | Figure paths do not exist; 0.7T figures unreproducible | **Fixed** (`7c6cf54`). Both sets archived with provenance READMEs; `FIGURES.md` indexes every current figure and the command that rebuilds it. |
| 8 | Same-candle execution | **Fixed** (`eb3cc84`). A signal on candle *c* fills on candle *c+1*'s open, repo-wide, and trades are stamped with the fill candle. |
| 9 | Effectively no test set in 80/20 | **Resolved** (`38d8f03`, `65ac748`). The configs are gone; the canonical split has a real 1,332-market test block, scored exactly once. |
| 10 | No RNG seeding; split not portable | **Fixed** (`7a3743d`). Agent and environment take a seed; the sweep is 30 seeded runs and reproduces to the last decimal across re-runs. |
| 11 | 88.5% of training at the epsilon floor | **Not fixed — not a defect.** The schedule still finishes in roughly the first 12% of a run. Worth a sentence in limitations. |
| 12 | 18 markets silently dropped | **Fixed** (`7a3743d`). An empty episode list now raises instead of printing; every track prints its sample cut. |

### The CONTRADICTED verdicts

13 (temporal split), 17 (fills), 19 and 20 (holding period, turnover), 21 (fee
denominator), 22 (capital normalisation), 26 (holds to settlement) and 27
(figure paths) are all addressed above.

**15 needs correcting, not closing.** It reads CONFIRMED — "test split never
read" — and that was true of the Q-learning track. It was not true of the rule
strategies: `generate_trades.py` defaulted to `--split all` and reached the test
block through a hardcoded `allow_test=True`, so the ordinary invocation read all
1,332 test markets. That is disclosed in `HELD_OUT.md` and at the top of
`strategies/README.md`.

### Things this audit did not catch

Found later, by running the tracks end to end rather than by reading them:

- **The tracks were not running the same cost model.** `generate_trades.py`
  defaulted `--slippage 0.0` while every other track defaulted `0.25`, and while
  its own help text said 0.25. Worth 196 per \$1k on `momentum_flip`, and a sign
  flip on its gross edge. Now stamped per market row and checked (`ec75bb2`).
- **The tracks overwrote each other's `markets.csv`.** Scoring a second split
  deleted the first, and a track with no rows reads as a skip, not an error.
- **Theta was selected on the split being reported**, which would have made the
  held-out number the best of eleven thresholds tried on held-out data.
- **`xgb_baseline_strategies.py` wrote to the committed driver's output file**,
  so running the exploratory script replaced what `RESULTS.md` is scored from.
- **A mean of ratios is not a ratio.** The collapsed sweep row reported
  `profit_factor` 9.85 on test, against a median of 0.457 with 23 of 30 seeds
  losing money (`65ac748`).

### Caveats that survive every fix

- **`fee_fraction_gross_pnl` is undefined more often than not.** It is NaN
  wherever gross PnL ≤ 0 — 25 of 30 seeds on val, 22 of 30 on test. Any mean of
  it describes the profitable minority. Do not quote it as the sweep's fee drag.
- **The Q-learning agent does not converge.** Across 30 seeds on val, `n_traded`
  ranges 0 to 598 and the sd exceeds the mean on every headline metric. One run
  of this agent tells you almost nothing; the report's own §3.3 predicted this.
- **The rule strategies' prior test exposure cannot be undone.** It is disclosed
  rather than repaired.
