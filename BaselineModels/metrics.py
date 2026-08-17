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

# 15-minute markets in a year: 4/hour * 24 * 365. Used to annualise the
# per-market Sharpe. Worth stating explicitly in the writeup since Sharpe means
# nothing without knowing what period it was scaled from.
MARKETS_PER_YEAR = 35_040

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
    def market_return(self) -> float:
        """PnL as a fraction of the capital this market actually used."""
        return self.pnl / self.stake_deployed if self.stake_deployed > 0 else 0.0


def results_frame(results: list[MarketResult]) -> pd.DataFrame:
    """Per-market results as a frame. Use for drawdown plots and error bars."""
    return pd.DataFrame(
        [
            {
                "event_slug": r.event_slug,
                "pnl": r.pnl,
                "gross_pnl": r.gross_pnl,
                "fees": r.fees,
                "stake_deployed": r.stake_deployed,
                "notional_traded": r.notional_traded,
                "n_trades": r.n_trades,
                "traded": r.traded,
                "holding_candles": r.holding_candles,
                "market_return": r.market_return,
                "early_exit": r.early_exit,
                "winner": r.winner,
            }
            for r in results
        ]
    )


def trading_metrics(
    results: list[MarketResult], markets_per_year: int = MARKETS_PER_YEAR
) -> dict[str, float]:
    """The trading table from the proposal, for one policy.

    Two definitions that are easy to get wrong:

    pnl_per_1k_deployed divides by the capital the policy actually committed,
    not by some notional $1,000 account. A policy that trades 2 markets out of
    5,900 and makes $1 isn't comparable to one that trades all of them unless
    the denominator moves too.

    fee_fraction_gross_pnl is NaN when gross PnL <= 0, since "fees ate 140% of
    a negative number" doesn't mean anything. Don't change this to return 0 --
    a hard zero here is what a fee accumulator that was never wired up looks
    like, and we want those to be obvious.
    """
    n = len(results)
    if n == 0:
        raise ValueError("no markets to score")

    traded = [r for r in results if r.traded]
    pnl = np.array([r.pnl for r in results], dtype=float)
    fees = float(sum(r.fees for r in results))
    deployed = float(sum(r.stake_deployed for r in results))
    notional = float(sum(r.notional_traded for r in results))
    gross = float(pnl.sum()) + fees

    # Only count markets we actually took a position in. Markets we sat out
    # carry no risk, and including them as zeros would deflate the vol.
    rets = np.array([r.market_return for r in traded], dtype=float)
    if len(rets) > 1 and rets.std(ddof=1) > 0:
        sharpe = float(rets.mean() / rets.std(ddof=1) * np.sqrt(markets_per_year))
    else:
        sharpe = float("nan")

    # Drawdown along the cumulative PnL path, in the order markets resolved.
    cum = np.cumsum(pnl)
    peak = np.maximum.accumulate(cum)
    max_dd = float(np.max(peak - cum)) if n else 0.0

    holding = np.array([r.holding_candles for r in traded], dtype=float)

    return {
        "n_markets": n,
        "n_traded": len(traded),
        "trade_rate": len(traded) / n,
        "total_pnl": float(pnl.sum()),
        "gross_pnl": gross,
        "capital_deployed": deployed,
        "pnl_per_1k_deployed": float(pnl.sum() / deployed * 1000.0) if deployed > 0 else 0.0,
        "sharpe": sharpe,
        "win_rate": float((np.array([r.pnl for r in traded]) > 0).mean()) if traded else float("nan"),
        "max_drawdown": max_dd,
        "turnover": float(notional / deployed) if deployed > 0 else 0.0,
        "total_fees": fees,
        "fee_fraction_gross_pnl": float(fees / gross) if gross > 0 else float("nan"),
        # These two are always defined, unlike the ratio above, which goes NaN
        # exactly when we lose money (i.e. the cases we most need to explain).
        # Put fee_per_1k next to pnl_per_1k and the gap between them is the
        # gross edge. That's the comparison the writeup wants.
        "fee_per_1k_deployed": float(fees / deployed * 1000.0) if deployed > 0 else 0.0,
        "gross_pnl_per_1k_deployed": float(gross / deployed * 1000.0) if deployed > 0 else 0.0,
        "avg_holding_candles": float(np.nanmean(holding)) if len(holding) else float("nan"),
        "early_exit_rate": float(np.mean([r.early_exit for r in traded])) if traded else 0.0,
    }


def comparison_table(named: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Stack a few policies' trading_metrics into one table for the paper."""
    return pd.DataFrame(named).T.rename_axis("policy").reset_index()
