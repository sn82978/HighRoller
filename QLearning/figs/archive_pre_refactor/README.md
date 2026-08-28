# Pre-refactor training figures — kept for provenance, not for citing

These 240 PNGs are the per-run training diagnostics behind the progress
report's Q-learning section. They are archived rather than deleted because the
report cites them, so they need to stay identifiable — but **no current code
path produces them, and they describe an environment that has since been
corrected.**

| directory | files | the report calls it |
|---|---|---|
| `0.7T_0.2E/` | 120 (ITER 0–29 × 4) | `Figs/7T2E/` |
| `0.8T_0.2E/` | 120 (ITER 0–29 × 4) | `Figs/8T2E/` |

Four figures per run: `_rewards`, `_epsilon`, `_action_frequencies`,
`_eval_action_sequences`.

## The cited paths never existed

`Figs/7T2E/` and `Figs/8T2E/` appear nowhere in the repository's history. The
report's paths are not a rename — they are wrong as written, in both directions:

- The directory layout was **created by hand**, not by the code. The
  pre-refactor `training.py` wrote every PNG into a flat `QLearning/figs/`; the
  `0.7T_0.2E/` subdirectory was made by moving files in commit `c4adf71`, and
  the 0.8T set was never moved at all — it sat loose at the top of
  `QLearning/figs/`, mixed in with everything else, until this commit.
- So the code as committed could not regenerate the layout the report cites,
  and re-running it would have overwritten one configuration with the other.

## Why they can't be reproduced

Same reasons the metrics CSVs can't — see
[`../../metrics/archive_pre_refactor/README.md`](../../metrics/archive_pre_refactor/README.md)
for the full account. In short: the two "configurations" were only filename
prefixes over a random day-file split, the runs were unseeded, and the
environment filled on the same candle it read state from.

One thing specific to the figures: the `_rewards` and `_action_frequencies`
panels are plotted in the units of the old reward function, which had no
capital and no slippage. Their y-axes are per-share price differences on one
notional unit, not dollars on an account, so they are not on the same scale as
anything produced today.

## Staleness that could not be confirmed

The 0.7T metrics CSV holds 35 rows while the report's statistics use the last
30, and the analysis script that drew the histograms read the whole file with
no row filter. The 0.7T histograms in `../../metrics/figs/archive_pre_refactor/`
therefore probably bin all 35 runs. This is a deduction from the archived code,
not something readable off the PNGs — recorded here as unconfirmed.

## What replaces them

```
python QLearning/training.py          # writes QLearning/figs/qlearning_seed<NN>_*.png
python QLearning/analysis.py          # writes QLearning/metrics/figs/qlearning/*.png
```

30 seeded runs on the canonical temporal split, through the shared cost model.
The seed is in the filename, so a figure names the run that produced it and
re-running overwrites in place instead of accumulating. See
[`../../../FIGURES.md`](../../../FIGURES.md) for the full index.
