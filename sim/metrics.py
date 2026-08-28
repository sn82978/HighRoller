"""The one definition of every trading metric in this project.

Both scoring paths -- ``BaselineModels.metrics.trading_metrics`` (which starts
from ``MarketResult`` objects) and ``sim.evaluation.score`` (which starts from a
committed ``markets.csv``) -- delegate here, so the four-way comparison the
proposal asks for is computed by one piece of code rather than two that happen
to use the same words.

They did not previously agree. ``max_drawdown`` came out positive from one and
negative from the other; Sharpe used a different set of markets in each; fee
drag and turnover existed in one and not the other. Numbers that disagree by
definition cannot be put in the same table, which is exactly what the report's
remaining goal requires.

The contested calls, settled
----------------------------
**Per-market return, and therefore Sharpe, is taken over every market in the
split** -- a market the policy sat out contributes a return of exactly 0.0, not
a missing value. This is the call that matters most. Scoring a selective policy
only on the markets it chose flatters it against an always-on policy like
buy-and-hold: an agent that trades 40 markets out of 1,343 and happens to win
them looks superb, while the fact that its capital was idle for the other 1,303
disappears. Over the full sequence, sitting out is a real outcome with a real
(zero) return. Both current baselines trade 100% of markets, so this changes
none of the published baseline numbers -- it makes the definition safe for the
Q-learning agent, which does not.

**Win rate is taken over traded markets only.** A market you sat out is neither
a win nor a loss, and counting it as a loss would make the no-trade floor score
0% rather than undefined. So win rate answers "when this policy committed
capital, how often was it right" while Sharpe answers "what did running this
policy do to a portfolio". Different questions; both are standard; the pairing
is only confusing if you do not say so, so it is said here.

**Max drawdown is a positive dollar figure measured from a zero baseline.** The
running peak starts at 0 rather than at the first market's PnL, so a policy that
loses from the very first market has that loss counted. Reported as a magnitude
(a drawdown *of* $4,423), never as a percentage.

**Fee drag divides by net gross PnL** -- realised PnL with fees added back,
losing markets contributing their actual negative PnL -- and is NaN, never 0.0,
when that quantity is <= 0. The alternative convention of summing only
profitable markets makes the number look best exactly when the strategy is
worst: on the same buy-and-hold trades the two differ by a factor of 21.5
(6.9% against positive-only, 147.3% against net gross). A hard 0.0 here is also
what a fee accumulator that was never wired up looks like, so it stays NaN.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

#: 15-minute markets in a year: 4/hour * 24 * 365. Sharpe means nothing without
#: stating what period it was scaled from, so it is named rather than inlined.
MARKETS_PER_YEAR = 35_040

#: The interchange schema. Every track writes these columns to its markets.csv
#: and :func:`score_records` reads exactly these, so a model is comparable to
#: the others by construction rather than by convention.
#:
#: ``fees`` through ``early_exit`` are what the old CSV schema was missing, and
#: their absence is why ``sim.evaluation.score`` could not compute fee drag,
#: turnover or holding period at all -- the numbers were not lost in the
#: arithmetic, they were never written down.
MARKET_RECORD_FIELDS: tuple[str, ...] = (
    "strategy",
    "event_slug",
    "start_ts",
    "split",
    "stake",
    "pnl",
    "fees",
    "stake_deployed",
    "notional_traded",
    "n_trades",
    "n_fills",
    "entry_candle",
    "exit_candle",
    "early_exit",
    "winner",
)


#: Column stamping the cost model each row was simulated under. Not part of
#: MARKET_RECORD_FIELDS -- it describes the run, not the market -- but written
#: alongside so "identical costs" is checkable from the artefacts instead of
#: being a convention held in someone's head.
COST_MODEL_FIELD = "slippage_frac"


def write_markets(
    path: str, frame: pd.DataFrame, split: str, *, slippage_frac: float | None = None
) -> int:
    """Write one split's rows into a track's markets.csv, keeping the others.

    The tracks used to overwrite this file wholesale. That is invisible while
    everything runs on val, and destructive the moment a second split is
    scored: `run_baselines.py --split test` deleted every val row, and the next
    `compare_models.py --split val` then had nothing to read for that track --
    silently, since a track with no rows for a split is a `[skip]`, not an
    error.

    Replacing only the rows whose `split` matches keeps every split already
    scored and makes re-running one split idempotent rather than doubling it.
    Returns the number of rows kept from the previous file.
    """
    if slippage_frac is not None:
        frame = frame.copy()
        frame[COST_MODEL_FIELD] = float(slippage_frac)

    kept = 0
    if os.path.exists(path):
        old = pd.read_csv(path)
        if "split" in old.columns:
            old = old[old.split != split]
            kept = len(old)
            if kept:
                frame = pd.concat([old, frame], ignore_index=True)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    frame.to_csv(path, index=False)
    return kept


def _capital_basis(mk: pd.DataFrame) -> np.ndarray:
    """Capital a policy had at risk in each market -- the return denominator.

    This is ``stake``, the per-market allotment (one $100 bankroll), **not**
    ``stake_deployed``, which sums the notional of every entry. Those differ
    whenever a policy re-enters: momentum_flip rolls one position through up to
    12 flips, and summing them charges it $1,200 of capital for a market where
    $100 was ever at risk.

    That distinction is not cosmetic. Dividing each market's PnL by the summed
    figure shrinks precisely the markets that flipped most -- which are the ones
    that lost most -- so the mean of the per-market ratios came out at +8.3% on
    a val run whose dollar total was -$36,296, with an annualised Sharpe of
    +38.8. A losing strategy read as a spectacular one.

    Falls back to ``stake_deployed`` when ``stake`` is absent or zero, which is
    the single-entry case where the two are equal anyway.
    """
    deployed = mk.stake_deployed.to_numpy(dtype=float)
    if "stake" not in mk.columns:
        return deployed
    stake = pd.to_numeric(mk.stake, errors="coerce").to_numpy(dtype=float)
    return np.where(np.isfinite(stake) & (stake > 0), stake, deployed)


def _max_drawdown_dollars(pnl: np.ndarray) -> float:
    """Worst peak-to-trough of the cumulative PnL path, as a positive number.

    The peak sequence is seeded with 0.0 so a policy that is underwater from its
    very first market has that drawdown counted rather than measured from the
    hole it already dug.
    """
    if len(pnl) == 0:
        return 0.0
    cum = np.cumsum(pnl)
    peak = np.maximum.accumulate(np.concatenate([[0.0], cum]))[1:]
    return float(np.max(peak - cum))


def _bootstrap_ci(x: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    """Percentile interval on the mean. Seeded, so a rerun reproduces it."""
    if len(x) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def score_records(
    mk: pd.DataFrame,
    markets_per_year: int = MARKETS_PER_YEAR,
    *,
    order_by: str = "start_ts",
) -> dict[str, float]:
    """Score one policy on one split. ``mk`` carries :data:`MARKET_RECORD_FIELDS`.

    One row per market, including markets the policy declined to trade -- their
    presence is what makes the return series comparable across policies with
    different participation rates.
    """
    missing = {"pnl", "fees", "stake_deployed", "notional_traded", "n_trades"} - set(mk.columns)
    if missing:
        raise ValueError(f"market records are missing columns: {sorted(missing)}")
    n = len(mk)
    if n == 0:
        raise ValueError("no markets to score")

    # Drawdown follows the order the markets actually resolved in, so sort here
    # rather than trusting the caller to have done it.
    if order_by in mk.columns:
        mk = mk.sort_values(order_by, kind="stable")

    pnl = mk.pnl.to_numpy(dtype=float)
    fees = float(mk.fees.sum())
    notional = float(mk.notional_traded.sum())
    gross = float(pnl.sum()) + fees

    traded_mask = mk.n_trades.to_numpy(dtype=float) > 0
    n_traded = int(traded_mask.sum())

    # Capital at risk per market, and the total actually committed: a market the
    # policy sat out deployed nothing, so it contributes 0 to the denominator of
    # the per-$1,000 figures while still contributing a 0.0 return to the series.
    basis = _capital_basis(mk)
    deployed = float(basis[traded_mask].sum())

    # Per-market return over EVERY market: untraded markets return 0.0. See the
    # module docstring -- this is what keeps a selective policy comparable to an
    # always-on one.
    with np.errstate(divide="ignore", invalid="ignore"):
        rets = np.where(basis > 0, pnl / np.where(basis > 0, basis, 1.0), 0.0)

    sd = float(rets.std(ddof=1)) if n > 1 else 0.0
    if sd > 0:
        sharpe_per_market = float(rets.mean() / sd)
        sharpe = sharpe_per_market * float(np.sqrt(markets_per_year))
        t_stat = sharpe_per_market * float(np.sqrt(n))
    else:
        sharpe_per_market = sharpe = t_stat = float("nan")

    ci_lo, ci_hi = _bootstrap_ci(rets)

    traded_pnl = pnl[traded_mask]
    wins = traded_pnl[traded_pnl > 0]
    losses = traded_pnl[traded_pnl < 0]

    holding = np.full(n, np.nan)
    if {"entry_candle", "exit_candle"} <= set(mk.columns):
        entry = pd.to_numeric(mk.entry_candle, errors="coerce").to_numpy(dtype=float)
        exit_ = pd.to_numeric(mk.exit_candle, errors="coerce").to_numpy(dtype=float)
        holding = exit_ - entry

    early = mk.early_exit.to_numpy(dtype=bool) if "early_exit" in mk.columns else np.zeros(n, bool)

    return {
        "n_markets": n,
        "n_traded": n_traded,
        "trade_rate": n_traded / n,
        # Entries plus early closes; settlement is a redemption, not a fill.
        "total_fills": int(mk.n_fills.sum()) if "n_fills" in mk.columns else int(mk.n_trades.sum()),
        "total_pnl": float(pnl.sum()),
        "gross_pnl": gross,
        "total_fees": fees,
        "capital_deployed": deployed,
        "avg_pnl_per_market": float(pnl.mean()),
        "median_pnl": float(np.median(pnl)),
        "pnl_per_1k_deployed": float(pnl.sum() / deployed * 1000.0) if deployed > 0 else 0.0,
        "gross_pnl_per_1k_deployed": float(gross / deployed * 1000.0) if deployed > 0 else 0.0,
        "fee_per_1k_deployed": float(fees / deployed * 1000.0) if deployed > 0 else 0.0,
        # NaN, never 0.0, when there are no profits for fees to be a share of.
        "fee_fraction_gross_pnl": float(fees / gross) if gross > 0 else float("nan"),
        "avg_return": float(rets.mean()),
        "return_ci95_lo": ci_lo,
        "return_ci95_hi": ci_hi,
        # Over traded markets only; NaN for a policy that never took a position.
        "win_rate": float((traded_pnl > 0).mean()) if n_traded else float("nan"),
        "profit_factor": (
            float(wins.sum() / abs(losses.sum())) if losses.size and losses.sum() < 0
            else float("nan")
        ),
        "sharpe": sharpe,
        "sharpe_per_market": sharpe_per_market,
        "t_stat": t_stat,
        "max_drawdown": _max_drawdown_dollars(pnl),
        "turnover": float(notional / deployed) if deployed > 0 else 0.0,
        "avg_holding_candles": (
            float(np.nanmean(holding[traded_mask])) if n_traded and not np.all(np.isnan(holding[traded_mask]))
            else float("nan")
        ),
        "early_exit_rate": float(early[traded_mask].mean()) if n_traded else 0.0,
    }


def comparison_table(named: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Stack several policies' scores into one table for the paper."""
    return pd.DataFrame(named).T.rename_axis("policy").reset_index()
