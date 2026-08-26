"""Scoring for the baselines. Two separate questions:

1. Is the forecast any good? log loss, Brier, AUC, calibration. Scored against
   the market price, not against 0.5 -- the market gets log loss 0.442 / AUC
   0.870 on train and costs nothing, so that's the bar.
2. Does trading on it make money? PnL after fees, Sharpe, drawdown, and how
   much of the gross PnL the fees eat. A model can do well on (1) and still
   lose on (2), which is more or less the point of the project.

One thing to watch out for: rows aren't independent. Every candle in a market
carries the same label, so 354k rows is really ~5.9k observations. Anything
averaged per row looks about 60x more precise than it is. That's why
paired_bootstrap_logloss resamples markets instead of rows.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# The trading metrics themselves live in sim.metrics, which is the single
# definition shared with the strategies and Q-learning tracks. Re-exported here
# so existing `from BaselineModels.metrics import ...` call sites keep working.
from sim.metrics import MARKETS_PER_YEAR, comparison_table, score_records

__all__ = [
    "log_loss",
    "brier",
    "expected_calibration_error",
    "calibration_table",
    "probability_metrics",
    "auc_by_horizon",
    "paired_bootstrap_logloss",
    "MarketResult",
    "results_frame",
    "trading_metrics",
    "comparison_table",
    "MARKETS_PER_YEAR",
]

_EPS = 1e-15


# -- probability quality -------------------------------------------------
def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    """Mean negative log likelihood. Lower is better; 0.693 is a coin flip."""
    p = np.clip(np.asarray(p, dtype=float), _EPS, 1 - _EPS)
    y = np.asarray(y, dtype=float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(y: np.ndarray, p: np.ndarray) -> float:
    """Mean squared error of the probability. 0.25 is a coin flip."""
    return float(np.mean((np.asarray(p, dtype=float) - np.asarray(y, dtype=float)) ** 2))


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """Weighted average gap between predicted confidence and what happened.

    Calibration matters more than AUC for us, since the trade rule compares a
    probability to a price. A model can rank well and still be untradeable if
    its probabilities are skewed. One-number version of calibration_table().
    """
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        n = int(m.sum())
        if n:
            total += n * abs(p[m].mean() - y[m].mean())
    return float(total / len(y)) if len(y) else float("nan")


def calibration_table(y: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Reliability curve as a table: predicted vs actual, per probability bin.

    This is the calibration figure for the writeup. If the forecaster is well
    calibrated, mean_predicted and observed_rate sit on the diagonal.
    """
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    rows = []
    for b in range(bins):
        m = idx == b
        n = int(m.sum())
        rows.append(
            {
                "bin_lo": edges[b],
                "bin_hi": edges[b + 1],
                "n": n,
                "mean_predicted": float(p[m].mean()) if n else float("nan"),
                "observed_rate": float(y[m].mean()) if n else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def probability_metrics(y: np.ndarray, p: np.ndarray, bins: int = 10) -> dict[str, float]:
    """Every forecast metric at once, for one row of the comparison table."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    # AUC is undefined when a split contains a single class.
    auc = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")
    return {
        "n": int(len(y)),
        "log_loss": log_loss(y, p),
        "brier": brier(y, p),
        "auc": auc,
        "ece": expected_calibration_error(y, p, bins),
        "base_rate": float(y.mean()),
    }


def auc_by_horizon(
    y: np.ndarray, p: np.ndarray, candles_remaining: np.ndarray, edges=(0, 4, 8, 12, 16, 20, 30, 40, 60)
) -> pd.DataFrame:
    """AUC bucketed by how much time is left in the market.

    A working prediction market should get sharper as it nears settlement. If
    it's near-certain at every horizon instead, something is leaking -- so this
    doubles as a sanity check whenever someone adds a feature.
    """
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    rem = np.asarray(candles_remaining, dtype=float)
    rows = []
    for lo, hi in zip(edges, edges[1:]):
        m = (rem >= lo) & (rem < hi)
        n = int(m.sum())
        rows.append(
            {
                "candles_remaining": f"{lo}-{hi}",
                "n": n,
                "auc": float(roc_auc_score(y[m], p[m]))
                if n and len(np.unique(y[m])) > 1
                else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def paired_bootstrap_logloss(
    y: np.ndarray,
    p_model: np.ndarray,
    p_market: np.ndarray,
    groups: np.ndarray,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, float]:
    """Is the model's log loss actually below the market's, or is that noise?

    Resamples whole markets with replacement. Candles inside a market are
    heavily dependent (same label), so resampling rows would shrink the
    interval by roughly sqrt(60) and make noise look like a result.

    Returns mean improvement (market minus model, so positive = model better)
    with a 95% interval and how often the model wins across resamples.
    """
    y = np.asarray(y, dtype=float)
    p_model = np.clip(np.asarray(p_model, dtype=float), _EPS, 1 - _EPS)
    p_market = np.clip(np.asarray(p_market, dtype=float), _EPS, 1 - _EPS)

    def _ll(yy, pp):
        return -(yy * np.log(pp) + (1 - yy) * np.log(1 - pp))

    # Per-row loss difference, then averaged within each market so every market
    # contributes one number regardless of how many candles it has.
    diff = _ll(y, p_market) - _ll(y, p_model)
    per_market = pd.Series(diff).groupby(np.asarray(groups)).mean().to_numpy()

    rng = np.random.default_rng(seed)
    n = len(per_market)
    draws = rng.integers(0, n, size=(n_boot, n))
    boots = per_market[draws].mean(axis=1)
    return {
        "n_markets": int(n),
        "mean_improvement": float(per_market.mean()),
        "ci_lo": float(np.percentile(boots, 2.5)),
        "ci_hi": float(np.percentile(boots, 97.5)),
        "p_model_better": float((boots > 0).mean()),
    }


# -- trading performance -------------------------------------------------
@dataclass
class MarketResult:
    """What happened when one policy played one market.

    pnl is net cash after settlement, so fees and slippage are already in it.
    gross_pnl adds the fees back, which is what makes the fee ratio mean
    anything.
    """

    event_slug: str
    pnl: float = 0.0
    fees: float = 0.0
    #: Capital allotted to this market -- the denominator for per-market return.
    #: Left 0.0 by callers that only ever take one entry, where it equals
    #: stake_deployed and sim.metrics falls back to that.
    stake: float = 0.0
    stake_deployed: float = 0.0
    notional_traded: float = 0.0
    n_trades: int = 0
    entry_candle: int | None = None
    exit_candle: int | None = None
    winner: str | None = None
    early_exit: bool = False

    @property
    def traded(self) -> bool:
        return self.n_trades > 0

    @property
    def gross_pnl(self) -> float:
        return self.pnl + self.fees

    @property
    def holding_candles(self) -> float:
        """Candles between entry and exit. NaN when the policy never entered."""
        if self.entry_candle is None or self.exit_candle is None:
            return float("nan")
        return float(self.exit_candle - self.entry_candle)

    @property
    def capital_basis(self) -> float:
        """Capital at risk in this market: the allotment, else the entry notional."""
        return self.stake if self.stake > 0 else self.stake_deployed

    @property
    def market_return(self) -> float:
        """PnL as a fraction of the capital this market actually risked."""
        basis = self.capital_basis
        return self.pnl / basis if basis > 0 else 0.0


def results_frame(
    results: list[MarketResult],
    *,
    strategy: str = "",
    split: str = "",
    start_ts: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Per-market results in the interchange schema of :mod:`sim.metrics`.

    Carries :data:`sim.metrics.MARKET_RECORD_FIELDS` so the frame can be scored
    by :func:`sim.metrics.score_records`, written as a markets.csv, and compared
    against the other tracks without any renaming in between. ``gross_pnl``,
    ``traded``, ``holding_candles`` and ``market_return`` are kept alongside as
    derived conveniences for plots.

    ``start_ts`` maps event_slug -> market start, used to order the cumulative
    PnL path. Without it the column is 0 and the caller's own ordering stands.
    """
    start_ts = start_ts or {}
    return pd.DataFrame(
        [
            {
                "strategy": strategy,
                "event_slug": r.event_slug,
                "start_ts": int(start_ts.get(r.event_slug, 0)),
                "split": split,
                "stake": r.stake,
                "pnl": r.pnl,
                "fees": r.fees,
                "stake_deployed": r.stake_deployed,
                "notional_traded": r.notional_traded,
                "n_trades": r.n_trades,
                "entry_candle": r.entry_candle,
                "exit_candle": r.exit_candle,
                "early_exit": r.early_exit,
                "winner": r.winner,
                # derived, for plotting -- score_records recomputes what it needs
                "gross_pnl": r.gross_pnl,
                "traded": r.traded,
                "holding_candles": r.holding_candles,
                "market_return": r.market_return,
            }
            for r in results
        ]
    )


def trading_metrics(
    results: list[MarketResult], markets_per_year: int = MARKETS_PER_YEAR
) -> dict[str, float]:
    """The trading table from the proposal, for one policy.

    A thin adapter now: it turns ``MarketResult`` objects into the interchange
    schema and hands them to :func:`sim.metrics.score_records`, which is the one
    place any of these quantities is defined. The strategies and Q-learning
    tracks reach the same function from the other side, starting from a
    committed markets.csv, so all three are scored by the same code rather than
    by three implementations that happen to use the same words.

    Read :mod:`sim.metrics` for the definitions themselves, in particular why
    Sharpe is taken over every market in the split while win rate is taken over
    traded markets only.
    """
    if not results:
        raise ValueError("no markets to score")
    return score_records(results_frame(results), markets_per_year)
