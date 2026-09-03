# Pre-refactor metric histograms — kept for provenance, not for citing

These 44 PNGs are the across-run histograms the progress report shows in
Section 4.2. Each bins one metric over the runs recorded in the matching
archived CSV.

| directory | files | binned from |
|---|---|---|
| `0.7T_0.2E/` | 22 (11 metrics × Training/Evaluation) | `../../archive_pre_refactor/0.7T_0.2E_qtable_{training,eval}.csv` |
| `0.8T_0.2E/` | 22 | `../../archive_pre_refactor/0.8T_0.2E_qtable_{training,eval}.csv` |

**None of them can be regenerated, and three of the eleven metrics they plot no
longer mean what their titles say.** The full account of why the underlying
runs are unreproducible is in
[`../../archive_pre_refactor/README.md`](../../archive_pre_refactor/README.md).
What matters for reading the pictures themselves:

- `Training_avg_holding_period` / `Evaluation_avg_holding_period` measure
  **episode length, not holding period**. They sit at ~59.9 of a maximum 60 by
  construction, and would look identical for an agent that never opened a
  position. The report's "the learned policy almost always rides a position to
  forced settlement" reads that number as behaviour; it is not. Measured
  properly, the corrected agent holds 2.39 candles.
- `*_turnover` is likewise episode length × N, not notional traded over capital.
- `*_fee_fraction_gross_pnl` was computed against a `max(0, ...)`-clamped gross
  PnL that dropped every losing step while keeping its fees. It reads ~25% where
  the unclamped figure on the same trades is ~308%.
- `*_pnl_per_1k_capital` is `total_pnl / 1000`. There was no capital in the
  simulator to normalise by.

The 0.7T histograms probably bin all 35 rows of a CSV whose reported statistics
use only the last 30 — see the note in
[`../../../figs/archive_pre_refactor/README.md`](../../../figs/archive_pre_refactor/README.md).

## What replaces them

```
python QLearning/training.py     # 30 seeded runs -> metrics/qlearning_{train,val}.csv
python QLearning/analysis.py     # -> metrics/figs/qlearning/<Split>_<metric>.png
```

The replacement set is `../qlearning/`: 25 metrics × Training/Validation, every
one of them a name `sim/metrics.py` actually defines, each histogram carrying a
mean line and backed by a `<Split>_summary.csv` of mean/sd/min/max.
