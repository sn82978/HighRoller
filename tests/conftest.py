"""Shared fixtures: synthetic candle tapes that mimic the real dataset's shape.

Synthetic rather than sampled from disk so the feature tests run without the
15s candle dataset built, and so a market's price path can be constructed to
make a leak visible if one exists.
"""

import numpy as np
import pandas as pd
import pytest

PRE_OPEN = -40
LAST = 59


def make_market(
    slug: str,
    start_ts: int,
    prices: np.ndarray | None = None,
    winner: str = "Up",
    truncated: bool = False,
    seed: int = 0,
) -> pd.DataFrame:
    """One market's dense 100-candle tape, in the real schema."""
    n = LAST - PRE_OPEN + 1
    rng = np.random.default_rng(seed)
    if prices is None:
        steps = rng.normal(0, 0.01, n)
        prices = np.clip(0.5 + np.cumsum(steps), 0.02, 0.98)
    prices = np.asarray(prices, dtype=float)
    assert len(prices) == n, f"need {n} prices, got {len(prices)}"

    spread = rng.uniform(0.005, 0.03, n)
    trades = rng.integers(0, 12, n)
    has_trades = trades > 0
    return pd.DataFrame(
        {
            "event_slug": slug,
            "condition_id": f"0x{abs(hash(slug)):x}",
            "start_ts": start_ts,
            "end_ts": start_ts + 900,
            "candle_index": np.arange(PRE_OPEN, LAST + 1),
            "t": start_ts + 15 * np.arange(PRE_OPEN, LAST + 1),
            "timestamp": pd.to_datetime(
                start_ts + 15 * np.arange(PRE_OPEN, LAST + 1), unit="s", utc=True
            ),
            "open": prices,
            "high": prices + spread,
            "low": prices - spread,
            "close": prices,
            "vwap": np.where(has_trades, prices, np.nan),
            "volume_shares": np.where(has_trades, rng.uniform(10, 500, n), 0.0),
            "volume_usdc": np.where(has_trades, rng.uniform(5, 250, n), 0.0),
            "trades": trades,
            "has_trades": has_trades,
            "truncated": truncated,
            "winner": winner,
            # Market TOTAL volume -- constant across every row, including
            # pre-open. This is the leaky column.
            "volume": 12345.6,
        }
    )


@pytest.fixture
def candles() -> pd.DataFrame:
    """Three markets with different outcomes and price paths."""
    return pd.concat(
        [
            make_market("btc-updown-15m-1000000000", 1_000_000_000, winner="Up", seed=1),
            make_market("btc-updown-15m-1000000900", 1_000_000_900, winner="Down", seed=2),
            make_market("btc-updown-15m-1000001800", 1_000_001_800, winner="Up", seed=3),
        ],
        ignore_index=True,
    )


@pytest.fixture
def spike_candles() -> pd.DataFrame:
    """A market that is flat until candle 50, then jumps to near-certainty.

    If any feature leaks the future, rows before candle 50 will know about the
    jump and the truncation check in `assert_no_lookahead` will fire.
    """
    n = LAST - PRE_OPEN + 1
    prices = np.full(n, 0.50)
    prices[(50 - PRE_OPEN) :] = 0.97
    return make_market("btc-updown-15m-1000002700", 1_000_002_700, prices=prices, winner="Up")
